from functools import reduce
import json
import os
import re

from datasets import load_dataset
import gradio as gr
from huggingface_hub import hf_hub_download
from huggingface_hub.repocard import metadata_load
import pandas as pd
from tqdm.autonotebook import tqdm

from utils.model_size import get_model_parameters_memory
from envs import LEADERBOARD_CONFIG, MODEL_META, REPO_ID, RESULTS_REPO, API

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


def make_clickable_model(model_name, link=None):
    if link is None:
        link = "https://huggingface.co/" + model_name
    # Remove user from model name
    return (
        f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name.split("/")[-1]}</a>'
    )

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

PROPRIETARY_MODELS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in PROPRIETARY_MODELS
}
SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS
}
CROSS_ENCODERS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in CROSS_ENCODERS
}
BI_ENCODERS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in BI_ENCODERS
}


TASK_TO_TASK_TYPE = {task_category: [] for task_category in TASKS}
for board_config in BOARDS_CONFIG.values():
    for task_category, task_list in board_config["tasks"].items():
        TASK_TO_TASK_TYPE[task_category].extend(task_list)

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

if os.path.exists("EXTERNAL_MODEL_RESULTS.json"):
    with open("EXTERNAL_MODEL_RESULTS.json") as f:
        EXTERNAL_MODEL_RESULTS = json.load(f)
    # Update with models not contained
    models_to_run = []
    for model in EXTERNAL_MODELS:
        if model not in EXTERNAL_MODEL_RESULTS:
            models_to_run.append(model)
            EXTERNAL_MODEL_RESULTS[model] = {k: {v[0]: []} for k, v in TASK_TO_METRIC.items()}
else:
    EXTERNAL_MODEL_RESULTS = {model: {k: {v[0]: []} for k, v in TASK_TO_METRIC.items()} for model in EXTERNAL_MODELS}
    models_to_run = EXTERNAL_MODELS

pbar = tqdm(models_to_run, desc="Fetching external model results")
for model in pbar:
    pbar.set_description(f"Fetching external model results for {model!r}")
    ds = load_dataset(RESULTS_REPO, model, trust_remote_code=True)
    # For local debugging:
    #, download_mode='force_redownload', verification_mode="no_checks")
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
    json.dump(EXTERNAL_MODEL_RESULTS, f)

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

def make_datasets_clickable(df):
    """Does not work"""
    if "BornholmBitextMining" in df.columns:
        link = "https://huggingface.co/datasets/strombergnlp/bornholmsk_parallel"
        df = df.rename(
            columns={f'BornholmBitextMining': '<a target="_blank" style="text-decoration: underline" href="{link}">BornholmBitextMining</a>',})
    return df

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

model_infos_path = "model_infos.json"
MODEL_INFOS = {}
if os.path.exists(model_infos_path):
    with open(model_infos_path) as f:
        MODEL_INFOS = json.load(f)

def get_mteb_data(tasks=["Clustering"], langs=[], datasets=[], fillna=True, add_emb_dim=True, task_to_metric=TASK_TO_METRIC, rank=True, refresh=True):
    global MODEL_INFOS
    api = API
    models = api.list_models(filter="mteb")
    # Legacy names changes; Also fetch the old results & merge later
    if ('MLSUMClusteringP2P (fr)' in datasets):
        datasets.append('MLSUMClusteringP2P')
    if ('MLSUMClusteringS2S (fr)' in datasets):
        datasets.append('MLSUMClusteringS2S')
    # Initialize list to models that we cannot fetch metadata from
    df_list = []
    for model in EXTERNAL_MODEL_RESULTS:
        results_list = []
        for task in tasks:
            # Not all models have InstructionRetrieval, other new tasks
            if task not in EXTERNAL_MODEL_RESULTS[model]: continue
            results_list += EXTERNAL_MODEL_RESULTS[model][task][task_to_metric[task][0]]
        
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

    for model in models:
        if model.modelId in MODELS_TO_SKIP: continue
        print("MODEL", model.modelId)
        if model.modelId not in MODEL_INFOS or refresh:
            readme_path = hf_hub_download(model.modelId, filename="README.md")
            meta = metadata_load(readme_path)
            MODEL_INFOS[model.modelId] = {
                "metadata": meta
            }
        meta = MODEL_INFOS[model.modelId]["metadata"]
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
        # if model.modelId == "w601sxs/b1ade-embed-kd_3":
        #     import pdb; pdb.set_trace()
        try:
            out = [{res["dataset"]["name"].replace("MTEB ", ""): [round(score["value"], 2) for score in res["metrics"] if filter_metric_fetched(res["dataset"]["name"].replace("MTEB ", ""), score["type"], task_to_metric.get(res["task"]["type"]))][0]} for res in task_results]
        except Exception as e:
            import pdb; pdb.set_trace()
            print("ERROR", model.modelId)
            continue
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = make_clickable_model(model.modelId)
        # Model & at least one result
        if len(out) > 1:
            if add_emb_dim:
                # The except clause triggers on gated repos, we can use external metadata for those
                try:
                    if "dim_seq_size" not in MODEL_INFOS[model.modelId] or refresh:
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

    # Save & cache MODEL_INFOS
    with open("model_infos.json", "w") as f:
        json.dump(MODEL_INFOS, f)

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
def get_mteb_average(task_dict: dict, refresh=True):
    all_tasks = reduce(lambda x, y: x + y, task_dict.values())
    DATA_OVERALL = get_mteb_data(
        tasks=list(task_dict.keys()),
        datasets=all_tasks,
        fillna=False,
        add_emb_dim=True,
        rank=False,
        refresh=refresh
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

boards_data = {}
all_data_tasks = []
for board, board_config in BOARDS_CONFIG.items():
    boards_data[board] = {
        "data_overall": None,
        "data_tasks": {}
    }
    if board_config["has_overall"]:
        data_overall, data_tasks = get_mteb_average(board_config["tasks"], refresh=False)
        boards_data[board]["data_overall"] = data_overall
        boards_data[board]["data_tasks"] = data_tasks
        all_data_tasks.extend(data_tasks.values())
    else:
        for task_category, task_category_list in board_config["tasks"].items():
            data_task_category = get_mteb_data(tasks=[task_category], datasets=task_category_list, refresh=False)
            data_task_category.drop(columns=["Embedding Dimensions", "Max Tokens"], inplace=True)
            boards_data[board]["data_tasks"][task_category] = data_task_category
            all_data_tasks.append(data_task_category)

# Exact, add all non-nan integer values for every dataset
NUM_SCORES = 0
DATASETS = []
MODELS = []
# LANGUAGES = []
for d in all_data_tasks:
    # NUM_SCORES += d.iloc[:, 1:].apply(lambda x: sum([1 for y in x if isinstance(y, float) and not np.isnan(y)]), axis=1).sum()
    cols_to_ignore = 4 if "Average" in d.columns else 3
    # Count number of scores including only non-nan floats & excluding the rank column
    NUM_SCORES += d.iloc[:, cols_to_ignore:].notna().sum().sum()
    # Exclude rank & model name column (first two); Do not count different language versions as different datasets
    DATASETS += [i.split(" ")[0] for i in d.columns[cols_to_ignore:]]
    # LANGUAGES += [i.split(" ")[-1] for i in d.columns[cols_to_ignore:]]
    MODELS += d["Model"].tolist()

NUM_DATASETS = len(set(DATASETS))
# NUM_LANGUAGES = len(set(LANGUAGES))
NUM_MODELS = len(set(MODELS))

# 1. Force headers to wrap
# 2. Force model column (maximum) width
# 3. Prevent model column from overflowing, scroll instead
# 4. Prevent checkbox groups from taking up too much space
css = """
table > thead {
    white-space: normal
}

table {
    --cell-width-1: 250px
}

table > tbody > tr > td:nth-child(2) > div {
    overflow-x: auto
}

.filter-checkbox-group {
    max-width: max-content;
}
"""

"""
Each inner tab can have the following keys:
- language: The language of the leaderboard
- language_long: [optional] The long form of the language
- description: The description of the leaderboard
- credits: [optional] The credits for the leaderboard
- data: The data for the leaderboard
- refresh: The function to refresh the leaderboard
"""

def get_refresh_function(task_category, task_list):
    def _refresh():
        data_task_category = get_mteb_data(tasks=[task_category], datasets=task_list)
        data_task_category.drop(columns=["Embedding Dimensions", "Max Tokens"], inplace=True)
        return data_task_category
    return _refresh


def get_refresh_overall_function(tasks):
    return lambda: get_mteb_average(tasks)[0]


data = {
    "Overall": {"metric": "Various, refer to task tabs", "data": []}
}
for task in TASKS:
    data[task] = {"metric": TASKS_CONFIG[task]["metric_description"], "data": []}

for board, board_config in BOARDS_CONFIG.items():
    init_name = board_config["title"]
    if init_name in PRETTY_NAMES:
        init_name = PRETTY_NAMES[init_name]
    board_pretty_name = f"{init_name} leaderboard"
    acronym = board_config.get("acronym", None)
    board_icon = board_config.get("icon", None)
    if board_icon is None:
        board_icon = ""
    credits = board_config.get("credits", None)
    metric = board_config.get("metric", None)

    if board_config["has_overall"]:
        overall_pretty_name = board_pretty_name
        if acronym is not None:
            overall_pretty_name += f" ({board_config['acronym']})"
        data["Overall"]["data"].append({
            "language": board_config["title"],
            "language_long": board_config["language_long"],
            "description": f"**Overall MTEB {overall_pretty_name}** 🔮{board_icon}",
            "data": boards_data[board]["data_overall"],
            "refresh": get_refresh_overall_function(board_config["tasks"]),
            "credits": credits,
            "metric": metric,
        })
    for task_category, task_category_list in board_config["tasks"].items():
        task_icon = TASKS_CONFIG[task_category]['icon']
        if "special_icons" in board_config and isinstance(board_config["special_icons"], dict):
            task_icon = board_config["special_icons"].get(task_category, task_icon)
        data[task_category]["data"].append({
            "language": board_config["title"],
            "language_long": board_config["language_long"],
            "description": f"**{task_category} {board_pretty_name}** {task_icon}{board_icon}",
            "data": boards_data[board]["data_tasks"][task_category],
            "refresh": get_refresh_function(task_category, task_category_list),
            "credits": credits,
            "metric": metric,
        })

dataframes = []
full_dataframes = []
tabs = []

# The following JavaScript function updates the URL parameters based on the selected task and language
# Additionally, `update_url_task` and `update_url_language` are used to update the current task and language
# The current task and language are stored in the `current_task_language` and `language_per_task` JSON objects
# This is all a bit hacky, but it might be the only way to pass options to a JavaScript function via Gradio
set_window_url_params = """
function(goalUrlObject) {
    const params = new URLSearchParams(window.location.search);
    for (const [key, value] of Object.entries(goalUrlObject)) {
        params.set(key, value);
    };
    const queryString = '?' + params.toString();
    console.log(queryString);
    window.history.replaceState({}, '', queryString);
    return [];
}
"""

def update_url_task(event: gr.SelectData, current_task_language: dict, language_per_task: dict):
    current_task_language["task"] = event.target.id
    # Either use the cached language for this task or the 1st language
    try:
        current_task_language["language"] = language_per_task.get(event.target.id, event.target.children[1].children[0].id)
    except Exception as e: # is Overall tab, no description
        current_task_language["language"] = language_per_task.get(event.target.id, event.target.children[0].children[0].id)

    return current_task_language, language_per_task

def update_url_language(event: gr.SelectData, current_task_language: dict, language_per_task: dict):
    current_task_language["language"] = event.target.id
    if "task" not in current_task_language:
        current_task_language["task"] = "overall"
    language_per_task[current_task_language["task"]] = event.target.id
    return current_task_language, language_per_task

NUMERIC_INTERVALS = {
    "<100M": pd.Interval(0, 100, closed="right"),
    "100M to 250M": pd.Interval(100, 250, closed="right"),
    "250M to 500M": pd.Interval(250, 500, closed="right"),
    "500M to 1B": pd.Interval(500, 1000, closed="right"),
    ">1B": pd.Interval(1000, 1_000_000, closed="right"),
}

MODEL_TYPES = [
    "Open",
    "Proprietary",
    "Sentence Transformers",
    "Cross-Encoders",
    "Bi-Encoders"
]

def filter_data(search_query, model_types, model_sizes, *full_dataframes):
    output_dataframes = []
    for df in full_dataframes:
        # Apply the search query
        if search_query:
            names = df["Model"].map(lambda x: re.match("<a .+?>(.+)</a>", x).group(1))
            masks = []
            for query in search_query.split(";"):
                masks.append(names.str.lower().str.contains(query.lower()))
            df = df[reduce(lambda a, b: a | b, masks)]

        # Apply the model type filtering
        if set(model_types) != set(MODEL_TYPES):
            masks = []
            for model_type in model_types:
                if model_type == "Open":
                    masks.append(~df["Model"].isin(PROPRIETARY_MODELS))
                elif model_type == "Proprietary":
                    masks.append(df["Model"].isin(PROPRIETARY_MODELS))
                elif model_type == "Sentence Transformers":
                    masks.append(df["Model"].isin(SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS))
                elif model_type == "Cross-Encoders":
                    masks.append(df["Model"].isin(CROSS_ENCODERS))
                elif model_type == "Bi-Encoders":
                    masks.append(df["Model"].isin(BI_ENCODERS))
            if masks:
                df = df[reduce(lambda a, b: a | b, masks)]
            else:
                df = pd.DataFrame(columns=df.columns)

        # Apply the model size filtering
        if set(model_sizes) != set(NUMERIC_INTERVALS.keys()):
            numeric_interval = pd.IntervalIndex(sorted([NUMERIC_INTERVALS[model_size] for model_size in model_sizes]))
            sizes = df["Model Size (Million Parameters)"].replace('', 0)
            mask = sizes.apply(lambda size: any(numeric_interval.contains(size)))
            df = df[mask]

        output_dataframes.append(df)
    return output_dataframes


with gr.Blocks(css=css) as block:

    # Store the current task and language for updating the URL. This is a bit hacky, but it works
    # for passing the current task and language to the JavaScript function via Gradio
    current_task_language = gr.JSON(value=dict(), visible=False)
    language_per_task = gr.JSON(value=dict(), visible=False)

    gr.Markdown(f"""
    Massive Text Embedding Benchmark (MTEB) Leaderboard. To submit, refer to the <a href="https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md" target="_blank" style="text-decoration: underline">MTEB GitHub repository</a> 🤗 Refer to the [MTEB paper](https://arxiv.org/abs/2210.07316) for details on metrics, tasks and models.
    """)

    with gr.Row():
        search_bar = gr.Textbox(
            label="Search Bar (separate multiple queries with `;`)",
            placeholder=" 🔍 Search for a model and press enter...",
        )
        filter_model_type = gr.CheckboxGroup(
            label="Model types",
            choices=MODEL_TYPES,
            value=MODEL_TYPES,
            interactive=True,
            elem_classes=["filter-checkbox-group"]
        )
        filter_model_sizes = gr.CheckboxGroup(
            label="Model sizes (in number of parameters)",
            choices=list(NUMERIC_INTERVALS.keys()),
            value=list(NUMERIC_INTERVALS.keys()),
            interactive=True,
            elem_classes=["filter-checkbox-group"],
            scale=2,
        )

    with gr.Tabs() as outer_tabs:
        # Store the tabs for updating them on load based on URL parameters
        tabs.append(outer_tabs)
        for task, task_values in data.items():
            metric = task_values["metric"]
            task_tab_id = task.lower().replace(" ", "-")

            # Overall, Bitext Mining, Classification, etc.
            pretty_task_name = task if task not in PRETTY_NAMES.keys() else PRETTY_NAMES[task]
            with gr.Tab(pretty_task_name, id=task_tab_id) as task_tab:
                # For updating the 'task' in the URL
                task_tab.select(update_url_task, [current_task_language, language_per_task], [current_task_language, language_per_task]).then(None, [current_task_language], [], js=set_window_url_params)
                if "Overall" != task:
                    gr.Markdown(TASK_DESCRIPTIONS[task])
                with gr.Tabs() as task_tabs:
                    # Store the task tabs for updating them on load based on URL parameters
                    tabs.append(task_tabs)

                    for item in task_values["data"]:
                        item_tab_id = item["language"].lower().replace(" ", "-")

                        # English, Chinese, French, etc.
                        with gr.Tab(item["language"], id=item_tab_id) as item_tab:
                            # For updating the 'language' in the URL
                            item_tab.select(update_url_language, [current_task_language, language_per_task], [current_task_language, language_per_task], trigger_mode="always_last").then(None, [current_task_language], [], js=set_window_url_params)

                            specific_metric = metric
                            if item.get("metric", None) is not None:
                                specific_metric = item['metric']
                            
                            with gr.Row():
                                gr.Markdown(f"""
                                {item['description']}

                                - **Metric:** {specific_metric}
                                - **Languages:** {item['language_long'] if 'language_long' in item else item['language']}
                                {"- **Credits:** " + item['credits'] if ("credits" in item and item["credits"] is not None) else ''}
                                """)

                            with gr.Row():
                                datatype = ["number", "markdown"] + ["number"] * len(item["data"])
                                dataframe = gr.Dataframe(item["data"], datatype=datatype, type="pandas", height=500)
                                dataframes.append(dataframe)

                                full_dataframe = gr.Dataframe(item["data"], datatype=datatype, type="pandas", visible=False)
                                full_dataframes.append(full_dataframe)

                            with gr.Row():
                                refresh_button = gr.Button("Refresh")
                                refresh_button.click(item["refresh"], inputs=None, outputs=dataframe, concurrency_limit=20)

    gr.Markdown(f"""
    - **Total Datasets**: {NUM_DATASETS}
    - **Total Languages**: 113
    - **Total Scores**: {NUM_SCORES}
    - **Total Models**: {NUM_MODELS}
    """ + r"""
    Made with ❤️ for NLP. If this work is useful to you, please consider citing:

    ```bibtex
    @article{muennighoff2022mteb,
        doi = {10.48550/ARXIV.2210.07316},
        url = {https://arxiv.org/abs/2210.07316},
        author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
        title = {MTEB: Massive Text Embedding Benchmark},
        publisher = {arXiv},
        journal={arXiv preprint arXiv:2210.07316},  
        year = {2022}
    }
    ```
    """)

    def set_tabs_on_load(request: gr.Request):
        """Set the selected tab based on the URL parameters on load."""
        global tabs
        valid_task_keys = [child.id for child in tabs[0].children]
        return_tabs = [gr.Tabs()] * len(tabs)

        query_params = request.request.query_params
        task_key = query_params.get("task", "overall")
        if task_key not in valid_task_keys:
            task_key = "overall"
        return_tabs[0] = gr.Tabs(selected=task_key)

        tabs_idx = valid_task_keys.index(task_key) + 1
        language_key = query_params.get("language", "english")
        return_tabs[tabs_idx] = gr.Tabs(selected=language_key)
        current_task_language = {"task": task_key, "language": language_key}
        language_per_task = {task_key: language_key}
        return return_tabs + [current_task_language, language_per_task]

    block.load(set_tabs_on_load, inputs=[], outputs=tabs + [current_task_language, language_per_task])

    search_bar.submit(filter_data, inputs=[search_bar, filter_model_type, filter_model_sizes] + full_dataframes, outputs=dataframes)
    filter_model_type.change(filter_data, inputs=[search_bar, filter_model_type, filter_model_sizes] + full_dataframes, outputs=dataframes)
    filter_model_sizes.change(filter_data, inputs=[search_bar, filter_model_type, filter_model_sizes] + full_dataframes, outputs=dataframes)

block.queue(max_size=10)
block.launch()

# Possible changes:
# Could add graphs / other visual content
# Could add verification marks

# Sources:
# https://huggingface.co/spaces/gradio/leaderboard
# https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard
# https://getemoji.com/
