from functools import reduce
import json
import os
import pickle
import re

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.repocard import metadata_load
import pandas as pd
from tqdm.autonotebook import tqdm

from utils.model_size import get_model_parameters_memory
from envs import LEADERBOARD_CONFIG, MODEL_META, REPO_ID, RESULTS_REPO, API


MODEL_CACHE = {}
TASKS_CONFIG = LEADERBOARD_CONFIG["tasks"]
BOARDS_CONFIG = LEADERBOARD_CONFIG["boards"]

TASKS = list(TASKS_CONFIG.keys())
PRETTY_NAMES = {
    "InstructionRetrieval": "Retrieval w/Instructions",
    "PairClassification": "Pair Classification",
    "BitextMining": "Bitext Mining",
}

TASK_TO_METRIC = {k: [v["metric"]] for k, v in TASKS_CONFIG.items()}
# Add legacy metric names
TASK_TO_METRIC["STS"].append("cos_sim_spearman")
TASK_TO_METRIC["STS"].append("cosine_spearman")
TASK_TO_METRIC["Summarization"].append("cos_sim_spearman")
TASK_TO_METRIC["Summarization"].append("cosine_spearman")
TASK_TO_METRIC["PairClassification"].append("cos_sim_ap")
TASK_TO_METRIC["PairClassification"].append("cosine_ap")


EXTERNAL_MODELS = {k for k,v in MODEL_META["model_meta"].items() if v.get("is_external", False)}
EXTERNAL_MODEL_TO_LINK = {k: v["link"] for k,v in MODEL_META["model_meta"].items() if v.get("link", False)}
EXTERNAL_MODEL_TO_DIM = {k: v["dim"] for k,v in MODEL_META["model_meta"].items() if v.get("dim", False)}
EXTERNAL_MODEL_TO_SEQLEN = {k: v["seq_len"] for k,v in MODEL_META["model_meta"].items() if v.get("seq_len", False)}
EXTERNAL_MODEL_TO_SIZE = {k: v["size"] for k,v in MODEL_META["model_meta"].items() if v.get("size", False)}
PROPRIETARY_MODELS = {k for k,v in MODEL_META["model_meta"].items() if v.get("is_proprietary", False)}
TASK_DESCRIPTIONS = {k: v["task_description"] for k,v in TASKS_CONFIG.items()}
TASK_DESCRIPTIONS["Overall"] = "Overall performance across MTEB tasks."
SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS = {k for k,v in MODEL_META["model_meta"].items() if v.get("is_sentence_transformers_compatible", False)}
MODELS_TO_SKIP = MODEL_META["models_to_skip"]
CROSS_ENCODERS = MODEL_META["cross_encoders"]
BI_ENCODERS = [k for k, _ in MODEL_META["model_meta"].items() if k not in CROSS_ENCODERS + ["bm25"]]



TASK_TO_TASK_TYPE = {task_category: [] for task_category in TASKS}
for board_config in BOARDS_CONFIG.values():
    for task_category, task_list in board_config["tasks"].items():
        TASK_TO_TASK_TYPE[task_category].extend(task_list)


## Don't cache this because we want to re-compute every time
# model_infos_path = "model_infos.json"
MODEL_INFOS = {}
# if os.path.exists(model_infos_path):
#     with open(model_infos_path) as f:
#         MODEL_INFOS = json.load(f)

def add_rank(df):
    cols_to_rank = [col for col in df.columns if col not in ["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens"]]
    if len(cols_to_rank) == 1:
        df.sort_values(cols_to_rank[0], ascending=False, inplace=True)
    else:
        df.insert(len(df.columns) - len(cols_to_rank), "Average", df[cols_to_rank].mean(axis=1, skipna=False))
        df.sort_values("Average", ascending=False, inplace=True)
    df.insert(0, "Rank", list(range(1, len(df) + 1)))
    df = df.round(2)
    # Fill NaN after averaging
    df.fillna("", inplace=True)
    return df


def make_clickable_model(model_name, link=None):
    if link is None:
        link = "https://huggingface.co/" + model_name
    # Remove user from model name
    return (
        f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name.split("/")[-1]}</a>'
    )


def add_lang(examples):
    if not(examples["eval_language"]):
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"]
    else:
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"] + f' ({examples["eval_language"]})'
    return examples

def norm(names): return set([name.split(" ")[0] for name in names])

def add_task(examples):
    # Could be added to the dataset loading script instead
    task_name = examples["mteb_dataset_name"]
    task_type = None
    for task_category, task_list in TASK_TO_TASK_TYPE.items():
        if task_name in norm(task_list):
            task_type = task_category
            break
    if task_type is not None:
        examples["mteb_task"] = task_type
    else:
        print("WARNING: Task not found for dataset", examples["mteb_dataset_name"])
        examples["mteb_task"] = "Unknown"
    return examples

def filter_metric_external(x, task, metrics):
    # This is a hack for the passkey and needle retrieval test, which reports ndcg_at_1 (i.e. accuracy), rather than the ndcg_at_10 that is commonly used for retrieval tasks. 
    if x['mteb_dataset_name'] in ['LEMBNeedleRetrieval', 'LEMBPasskeyRetrieval']:
        return x["mteb_task"] == task and x['metric'] == 'ndcg_at_1'
    else:
        return x["mteb_task"] == task and x["metric"] in metrics

def filter_metric_fetched(name, metric, expected_metrics):
    # This is a hack for the passkey and needle retrieval test, which reports ndcg_at_1 (i.e. accuracy), rather than the ndcg_at_10 that is commonly used for retrieval tasks. 
    return metric == 'ndcg_at_1' if name in ['LEMBNeedleRetrieval', 'LEMBPasskeyRetrieval'] else metric in expected_metrics


def get_dim_seq_size(model):
    filenames = [sib.rfilename for sib in model.siblings]
    dim, seq = "", ""
    for filename in filenames:
        if re.match("\d+_Pooling/config.json", filename):
            st_config_path = hf_hub_download(model.modelId, filename=filename)
            dim = json.load(open(st_config_path)).get("word_embedding_dimension", "")
            break
    for filename in filenames:
        if re.match("\d+_Dense/config.json", filename):
            st_config_path = hf_hub_download(model.modelId, filename=filename)
            dim = json.load(open(st_config_path)).get("out_features", dim)
    if "config.json" in filenames:
        config_path = hf_hub_download(model.modelId, filename="config.json")
        config = json.load(open(config_path))
        if not dim:
            dim = config.get("hidden_dim", config.get("hidden_size", config.get("d_model", "")))
        seq = config.get("n_positions", config.get("max_position_embeddings", config.get("n_ctx", config.get("seq_length", ""))))
    
    if dim == "" or seq == "":
        raise Exception(f"Could not find dim or seq for model {model.modelId}")
    
    # Get model file size without downloading. Parameters in million parameters and memory in GB
    parameters, memory = get_model_parameters_memory(model)
    return dim, seq, parameters, memory


def get_external_model_results():
    if os.path.exists("EXTERNAL_MODEL_RESULTS.json"):
        with open("EXTERNAL_MODEL_RESULTS.json") as f:
            EXTERNAL_MODEL_RESULTS = json.load(f)
        # Update with models not contained
        models_to_run = []
        for model in EXTERNAL_MODELS:
            if model not in EXTERNAL_MODEL_RESULTS:
                models_to_run.append(model)
                EXTERNAL_MODEL_RESULTS[model] = {k: {v[0]: []} for k, v in TASK_TO_METRIC.items()}

    ## only if we want to re-calculate all instead of using the cache... it's likely they haven't changed
    ## but if your model results have changed, delete it from the "EXTERNAL_MODEL_RESULTS.json" file
    else:
        EXTERNAL_MODEL_RESULTS = {model: {k: {v[0]: []} for k, v in TASK_TO_METRIC.items()} for model in EXTERNAL_MODELS}
        models_to_run = EXTERNAL_MODELS

    pbar = tqdm(models_to_run, desc="Fetching external model results")
    for model in pbar:
        pbar.set_description(f"Fetching external model results for {model!r}")
        ds = load_dataset(RESULTS_REPO, model, trust_remote_code=True, download_mode='force_redownload', verification_mode="no_checks")
        ds = ds.map(add_lang)
        ds = ds.map(add_task)
        base_dict = {"Model": make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))}

        for task, metrics in TASK_TO_METRIC.items():
            ds_dict = ds.filter(lambda x: filter_metric_external(x, task, metrics))["test"].to_dict()
            ds_dict = {k: round(v, 2) for k, v in zip(ds_dict["mteb_dataset_name_with_lang"], ds_dict["score"])}
            # metrics[0] is the main name for this metric; other names in the list are legacy for backward-compat
            EXTERNAL_MODEL_RESULTS[model][task][metrics[0]].append({**base_dict, **ds_dict})

    # Save & cache EXTERNAL_MODEL_RESULTS
    with open("EXTERNAL_MODEL_RESULTS.json", "w") as f:
        json.dump(EXTERNAL_MODEL_RESULTS, f, indent=4)

    return EXTERNAL_MODEL_RESULTS


def download_or_use_cache(modelId):
    global MODEL_CACHE
    if modelId in MODEL_CACHE:
        return MODEL_CACHE[modelId]
    try:
        readme_path = hf_hub_download(modelId, filename="README.md", etag_timeout=30)
    except Exception:
        print(f"ERROR: Could not fetch metadata for {modelId}, trying again")
        readme_path = hf_hub_download(modelId, filename="README.md", etag_timeout=30)
    meta = metadata_load(readme_path)
    MODEL_CACHE[modelId] = meta
    return meta


def get_mteb_data(tasks=["Clustering"], langs=[], datasets=[], fillna=True, add_emb_dim=True, task_to_metric=TASK_TO_METRIC, rank=True):
    global MODEL_INFOS

    with open("EXTERNAL_MODEL_RESULTS.json", "r") as f:
        external_model_results = json.load(f)

    api = API
    models = list(api.list_models(filter="mteb"))
    # Legacy names changes; Also fetch the old results & merge later
    if ('MLSUMClusteringP2P (fr)' in datasets):
        datasets.append('MLSUMClusteringP2P')
    if ('MLSUMClusteringS2S (fr)' in datasets):
        datasets.append('MLSUMClusteringS2S')
    # Initialize list to models that we cannot fetch metadata from
    df_list = []
    for model in external_model_results:
        results_list = []
        for task in tasks:
            # Not all models have InstructionRetrieval, other new tasks
            if task not in external_model_results[model]: continue
            results_list += external_model_results[model][task][task_to_metric[task][0]]
        
        if len(datasets) > 0:
            res = {k: v for d in results_list for k, v in d.items() if (k == "Model") or any([x in k for x in datasets])}
        elif langs:
            # Would be cleaner to rely on an extra language column instead
            langs_format = [f"({lang})" for lang in langs]
            res = {k: v for d in results_list for k, v in d.items() if any([k.split(" ")[-1] in (k, x) for x in langs_format])}
        else:
            res = {k: v for d in results_list for k, v in d.items()}
        # Model & at least one result
        if len(res) > 1:
            if add_emb_dim:
                res["Model Size (Million Parameters)"] = EXTERNAL_MODEL_TO_SIZE.get(model, "")
                res["Memory Usage (GB, fp32)"] = round(res["Model Size (Million Parameters)"] * 1e6 * 4 / 1024**3, 2) if res["Model Size (Million Parameters)"] != "" else ""
                res["Embedding Dimensions"] = EXTERNAL_MODEL_TO_DIM.get(model, "")
                res["Max Tokens"] = EXTERNAL_MODEL_TO_SEQLEN.get(model, "")
            df_list.append(res)

    pbar = tqdm(models, desc="Fetching model metadata")
    for model in pbar:
        if model.modelId in MODELS_TO_SKIP: continue
        pbar.set_description(f"Fetching {model.modelId!r} metadata")
        meta = download_or_use_cache(model.modelId)
        MODEL_INFOS[model.modelId] = {
            "metadata": meta
        }
        if "model-index" not in meta:
            continue
        # meta['model-index'][0]["results"] is list of elements like:
        # {
        #    "task": {"type": "Classification"},
        #    "dataset": {
        #        "type": "mteb/amazon_massive_intent",
        #        "name": "MTEB MassiveIntentClassification (nb)",
        #        "config": "nb",
        #        "split": "test",
        #    },
        #    "metrics": [
        #        {"type": "accuracy", "value": 39.81506388702084},
        #        {"type": "f1", "value": 38.809586587791664},
        #    ],
        # },
        # Use "get" instead of dict indexing to skip incompat metadata instead of erroring out
        if len(datasets) > 0:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks) and any([x in sub_res.get("dataset", {}).get("name", "") for x in datasets])]
        elif langs:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks) and (sub_res.get("dataset", {}).get("config", "default") in ("default", *langs))]
        else:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks)]
        try:
            out = [{res["dataset"]["name"].replace("MTEB ", ""): [round(score["value"], 2) for score in res["metrics"] if filter_metric_fetched(res["dataset"]["name"].replace("MTEB ", ""), score["type"], task_to_metric.get(res["task"]["type"]))][0]} for res in task_results]
        except Exception as e:
            print("ERROR", model.modelId, e)
            continue
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = make_clickable_model(model.modelId)
        # Model & at least one result
        if len(out) > 1:
            if add_emb_dim:
                # The except clause triggers on gated repos, we can use external metadata for those
                try:
                    MODEL_INFOS[model.modelId]["dim_seq_size"] = list(get_dim_seq_size(model))
                except:
                    name_without_org = model.modelId.split("/")[-1]
                    # EXTERNAL_MODEL_TO_SIZE[name_without_org] refers to millions of parameters, so for memory usage
                    # we multiply by 1e6 to get just the number of parameters, then by 4 to get the number of bytes
                    # given fp32 precision (4 bytes per float), then divide by 1024**3 to get the number of GB
                    MODEL_INFOS[model.modelId]["dim_seq_size"] = (
                        EXTERNAL_MODEL_TO_DIM.get(name_without_org, ""),
                        EXTERNAL_MODEL_TO_SEQLEN.get(name_without_org, ""),
                        EXTERNAL_MODEL_TO_SIZE.get(name_without_org, ""),
                        round(EXTERNAL_MODEL_TO_SIZE[name_without_org] * 1e6 * 4 / 1024**3, 2) if name_without_org in EXTERNAL_MODEL_TO_SIZE else "",
                    )
                out["Embedding Dimensions"], out["Max Tokens"], out["Model Size (Million Parameters)"], out["Memory Usage (GB, fp32)"] = tuple(MODEL_INFOS[model.modelId]["dim_seq_size"])
            df_list.append(out)
        if model.library_name == "sentence-transformers" or "sentence-transformers" in model.tags or "modules.json" in {file.rfilename for file in model.siblings}:
            SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS.add(out["Model"])

    # # Save & cache MODEL_INFOS
    # with open("model_infos.json", "w") as f:
    #     json.dump(MODEL_INFOS, f)

    df = pd.DataFrame(df_list)
    # If there are any models that are the same, merge them
    # E.g. if out["Model"] has the same value in two places, merge & take whichever one is not NaN else just take the first one
    df = df.groupby("Model", as_index=False).first()
    # Put 'Model' column first
    cols = sorted(list(df.columns))
    base_columns = ["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens"]
    if len(datasets) > 0:
        # Update legacy column names to be merged with newer ones
        # Update 'MLSUMClusteringP2P (fr)' with values from 'MLSUMClusteringP2P'
        if ('MLSUMClusteringP2P (fr)' in datasets) and ('MLSUMClusteringP2P' in cols):
            df['MLSUMClusteringP2P (fr)'] = df['MLSUMClusteringP2P (fr)'].fillna(df['MLSUMClusteringP2P'])
            datasets.remove('MLSUMClusteringP2P')
        if ('MLSUMClusteringS2S (fr)' in datasets) and ('MLSUMClusteringS2S' in cols):
            df['MLSUMClusteringS2S (fr)'] = df['MLSUMClusteringS2S (fr)'].fillna(df['MLSUMClusteringS2S'])
            datasets.remove('MLSUMClusteringS2S')
        # Filter invalid columns
        cols = [col for col in cols if col in base_columns + datasets]
    i = 0
    for column in base_columns:
        if column in cols:
            cols.insert(i, cols.pop(cols.index(column)))
            i += 1
    df = df[cols]
    if rank:
        df = add_rank(df)       
    if fillna:
        df.fillna("", inplace=True)
    return df


# Get dict with a task list for each task category
# E.g. {"Classification": ["AmazonMassiveIntentClassification (en)", ...], "PairClassification": ["SprintDuplicateQuestions", ...]}
def get_mteb_average(task_dict: dict):
    all_tasks = reduce(lambda x, y: x + y, task_dict.values())
    DATA_OVERALL = get_mteb_data(
        tasks=list(task_dict.keys()),
        datasets=all_tasks,
        fillna=False,
        add_emb_dim=True,
        rank=False,
    )
    # Debugging:
    # DATA_OVERALL.to_csv("overall.csv")
    
    DATA_OVERALL.insert(1, f"Average ({len(all_tasks)} datasets)", DATA_OVERALL[all_tasks].mean(axis=1, skipna=False))
    for i, (task_category, task_category_list) in enumerate(task_dict.items()):
        DATA_OVERALL.insert(i+2, f"{task_category} Average ({len(task_category_list)} datasets)", DATA_OVERALL[task_category_list].mean(axis=1, skipna=False))
    DATA_OVERALL.sort_values(f"Average ({len(all_tasks)} datasets)", ascending=False, inplace=True)
    # Start ranking from 1
    DATA_OVERALL.insert(0, "Rank", list(range(1, len(DATA_OVERALL) + 1)))

    DATA_OVERALL = DATA_OVERALL.round(2)

    DATA_TASKS = {}
    for task_category, task_category_list in task_dict.items():
        DATA_TASKS[task_category] = add_rank(DATA_OVERALL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] + task_category_list])
        DATA_TASKS[task_category] = DATA_TASKS[task_category][DATA_TASKS[task_category].iloc[:, 4:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL.fillna("", inplace=True)

    data_overall_rows = ["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens", f"Average ({len(all_tasks)} datasets)"]
    for task_category, task_category_list in task_dict.items():
        data_overall_rows.append(f"{task_category} Average ({len(task_category_list)} datasets)")

    DATA_OVERALL = DATA_OVERALL[data_overall_rows]
    DATA_OVERALL = DATA_OVERALL[DATA_OVERALL.iloc[:, 5:].ne("").any(axis=1)]

    return DATA_OVERALL, DATA_TASKS


def refresh_leaderboard():
    """
    The main code to refresh and calculate results for the leaderboard. It does this by fetching the results from the
        external models and the models in the leaderboard, then calculating the average scores for each task category.

    Returns:
        dict: A dictionary containing the overall leaderboard and the task category leaderboards.
    """

    # get external model results and cache them
    external_results = get_external_model_results()

    boards_data = {}
    all_data_tasks = []
    pbar_tasks = tqdm(BOARDS_CONFIG.items(), desc="Fetching leaderboard results for ???", total=len(BOARDS_CONFIG), leave=True)
    for board, board_config in pbar_tasks:
        boards_data[board] = {
            "data_overall": None,
            "data_tasks": {}
        }
        pbar_tasks.set_description(f"Fetching leaderboard results for {board!r}")
        pbar_tasks.refresh()
        if board_config["has_overall"]:
            data_overall, data_tasks = get_mteb_average(board_config["tasks"])
            boards_data[board]["data_overall"] = data_overall
            boards_data[board]["data_tasks"] = data_tasks
            all_data_tasks.extend(data_tasks.values())
        else:
            for task_category, task_category_list in board_config["tasks"].items():
                data_task_category = get_mteb_data(tasks=[task_category], datasets=task_category_list)
                data_task_category.drop(columns=["Embedding Dimensions", "Max Tokens"], inplace=True)
                boards_data[board]["data_tasks"][task_category] = data_task_category
                all_data_tasks.append(data_task_category)

    return all_data_tasks, boards_data



if __name__ == "__main__":
    print(f"Refreshing leaderboard statistics...")
    all_data_tasks, boards_data = refresh_leaderboard()

    print(f"Done calculating, saving...")
    # save them so that the leaderboard can use them, as pickles because they're quite complex objects
    with open("all_data_tasks.pkl", "wb") as f:
        pickle.dump(all_data_tasks, f)

    with open("boards_data.pkl", "wb") as f:
        pickle.dump(boards_data, f)