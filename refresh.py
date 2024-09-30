from __future__ import annotations

import json

import os
import re
from functools import reduce
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.repocard import metadata_load
from tqdm.autonotebook import tqdm

from envs import API, LEADERBOARD_CONFIG, MODEL_META, REPO_ID, RESULTS_REPO
from utils.model_size import get_model_parameters_memory

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
TASK_TO_METRIC["STS"].append("spearman")
TASK_TO_METRIC["Summarization"].append("cos_sim_spearman")
TASK_TO_METRIC["Summarization"].append("spearman")
TASK_TO_METRIC["PairClassification"].append("ap")
TASK_TO_METRIC["PairClassification"].append("cos_sim_ap")
TASK_TO_METRIC["PairClassification"].append("cosine_ap")


EXTERNAL_MODELS = {
    k for k, v in MODEL_META["model_meta"].items() if v.get("is_external", False)
}
EXTERNAL_MODEL_TO_LINK = {
    k: v["link"] for k, v in MODEL_META["model_meta"].items() if v.get("link", False)
}
EXTERNAL_MODEL_TO_DIM = {
    k: v["dim"] for k, v in MODEL_META["model_meta"].items() if v.get("dim", False)
}
EXTERNAL_MODEL_TO_SEQLEN = {
    k: v["seq_len"]
    for k, v in MODEL_META["model_meta"].items()
    if v.get("seq_len", False)
}
EXTERNAL_MODEL_TO_SIZE = {
    k: v["size"] for k, v in MODEL_META["model_meta"].items() if v.get("size", False)
}
PROPRIETARY_MODELS = {
    k for k, v in MODEL_META["model_meta"].items() if v.get("is_proprietary", False)
}
TASK_DESCRIPTIONS = {k: v["task_description"] for k, v in TASKS_CONFIG.items()}
TASK_DESCRIPTIONS["Overall"] = "Overall performance across MTEB tasks."
SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS = {
    k
    for k, v in MODEL_META["model_meta"].items()
    if v.get("is_sentence_transformers_compatible", False)
}
MODELS_TO_SKIP = MODEL_META["models_to_skip"]
CROSS_ENCODERS = MODEL_META["cross_encoders"]
BI_ENCODERS = [
    k for k, _ in MODEL_META["model_meta"].items() if k not in CROSS_ENCODERS + ["bm25"]
]
INSTRUCT_MODELS = {
    k for k, v in MODEL_META["model_meta"].items() if v.get("uses_instruct", False)
}
NOINSTRUCT_MODELS = {
    k for k, v in MODEL_META["model_meta"].items() if not v.get("uses_instruct", False)
}


TASK_TO_TASK_TYPE = {task_category: [] for task_category in TASKS}
TASK_TO_SPLIT = {}
for k, board_config in BOARDS_CONFIG.items():
    for task_category, task_list in board_config["tasks"].items():
        TASK_TO_TASK_TYPE[task_category].extend(task_list)
        if "split" in board_config:
            TASK_TO_SPLIT[k] = board_config["split"]


## Don't cache this because we want to re-compute every time
# model_infos_path = "model_infos.json"
MODEL_INFOS = {}
# if os.path.exists(model_infos_path):
#     with open(model_infos_path) as f:
#         MODEL_INFOS = json.load(f)


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_rank = [
        col
        for col in df.columns
        if col
        not in [
            "Model",
            "Model Size (Million Parameters)",
            "Memory Usage (GB, fp32)",
            "Embedding Dimensions",
            "Max Tokens",
        ]
    ]
    if len(cols_to_rank) == 1:
        df.sort_values(cols_to_rank[0], ascending=False, inplace=True)
    else:
        df.insert(
            len(df.columns) - len(cols_to_rank),
            "Average",
            df[cols_to_rank].mean(axis=1, skipna=False),
        )
        df.sort_values("Average", ascending=False, inplace=True)
    df.insert(0, "Rank", list(range(1, len(df) + 1)))
    df = df.round(2)
    # Fill NaN after averaging
    df.fillna("", inplace=True)
    return df


def make_clickable_model(model_name: str, link: None | str = None) -> str:
    if link is None:
        link = "https://huggingface.co/" + model_name
    # Remove user from model name
    model_name = model_name.split("/")[-1]
    model_name = model_name.split("__")[-1]
    return f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name}</a>'


def add_subset(examples):
    if not (examples["hf_subset"]) or (examples["hf_subset"] == "default"):
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"]
    else:
        examples["mteb_dataset_name_with_lang"] = (
            examples["mteb_dataset_name"] + f' ({examples["hf_subset"]})'
        )
    return examples


def norm(names: list[str]) -> set[str]:
    return set([name.split()[0] for name in names])


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


def filter_metric_external(x, task, metrics) -> bool:
    # This is a hack for the passkey and needle retrieval test, which reports ndcg_at_1 (i.e. accuracy), rather than the ndcg_at_10 that is commonly used for retrieval tasks.
    if x["mteb_dataset_name"] in ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]:
        return bool(x["mteb_task"] == task and x["metric"] == "ndcg_at_1")
    elif (x["mteb_dataset_name"].startswith("BrightRetrieval") and (x["split"] == "long")):
        return bool(x["mteb_task"] == task and x["metric"] in ["recall_at_1"])
    elif x["mteb_dataset_name"] == "MIRACLReranking":
        return bool(x["mteb_task"] == task and x["metric"] in ["NDCG@10(MIRACL)"])
    else:
        return bool(x["mteb_task"] == task and x["metric"] in metrics)


def filter_metric_fetched(name: str, metric: str, expected_metrics, split: str) -> bool:
    # This is a hack for the passkey and needle retrieval test, which reports ndcg_at_1 (i.e. accuracy), rather than the ndcg_at_10 that is commonly used for retrieval tasks.
    if name in ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]:
        return bool(metric == "ndcg_at_1")
    elif (name.startswith("BrightRetrieval") and (split == "long")):
        return bool(metric in ["recall_at_1"])
    elif name.startswith("MIRACLReranking"):
        return bool(metric in ["NDCG@10(MIRACL)"])
    else:
        return bool(metric in expected_metrics)


def get_dim_seq_size(model):
    siblings = model.siblings or []
    filenames = [sib.rfilename for sib in siblings]
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
            dim = config.get(
                "hidden_dim", config.get("hidden_size", config.get("d_model", ""))
            )
        seq = config.get(
            "n_positions",
            config.get(
                "max_position_embeddings",
                config.get("n_ctx", config.get("seq_length", "")),
            ),
        )

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
                EXTERNAL_MODEL_RESULTS[model] = {
                    k: {v[0]: []} for k, v in TASK_TO_METRIC.items()
                }

    ## only if we want to re-calculate all instead of using the cache... it's likely they haven't changed
    ## but if your model results have changed, delete it from the "EXTERNAL_MODEL_RESULTS.json" file
    else:
        EXTERNAL_MODEL_RESULTS = {
            model: {k: {v[0]: []} for k, v in TASK_TO_METRIC.items()}
            for model in EXTERNAL_MODELS
        }
        models_to_run = EXTERNAL_MODELS

    pbar = tqdm(models_to_run, desc="Fetching external model results")
    for model in pbar:
        pbar.set_description(f"Fetching external model results for {model!r}")
        try:
            ds = load_dataset(
                RESULTS_REPO,
                model,
                trust_remote_code=True,
                download_mode="force_redownload",
                verification_mode="no_checks",
            )
        except (KeyError, ValueError) as e:
            model_tmp = "__".join([MODEL_META["model_meta"][model]["link"].split("/")[-2], model])
            ds = load_dataset(
                RESULTS_REPO,
                model_tmp,
                trust_remote_code=True,
                download_mode="force_redownload",
                verification_mode="no_checks",
            )
        except ValueError as e:
            print(f"Can't fined model {model} in results repository. Exception: {e}")
            continue

        ds = ds.map(add_subset)
        ds = ds.map(add_task)
        base_dict = {
            "Model": make_clickable_model(
                model,
                link=EXTERNAL_MODEL_TO_LINK.get(
                    model, f"https://huggingface.co/spaces/{REPO_ID}"
                ),
            )
        }

        for task, metrics in TASK_TO_METRIC.items():
            ds_sub = ds.filter(lambda x: filter_metric_external(x, task, metrics))[
                "test"
            ]
            curent_task_metrics = ds_sub.unique("metric")
            for metric in curent_task_metrics:
                ds_dict = ds_sub.filter(lambda x: x["metric"] == metric).to_dict()
                ds_dict = {
                    k: round(v, 2)
                    for k, v in zip(
                        ds_dict["mteb_dataset_name_with_lang"], ds_dict["score"]
                    )
                }
                # metrics[0] is the main name for this metric; other names in the list are legacy for backward-compat
                # except for recall_at_1, which is the main name for BrightRetrieval (Long)
                metric = metrics[0] if metric != "recall_at_1" else metric
                if metric not in EXTERNAL_MODEL_RESULTS[model][task]:
                    EXTERNAL_MODEL_RESULTS[model][task][metric] = []
                EXTERNAL_MODEL_RESULTS[model][task][metric].append(
                    {**base_dict, **ds_dict}
                )
            #ds_dict = ds.filter(lambda x: filter_metric_external(x, task, metrics))[
            #    "test"
            #].to_dict()
            #ds_dict = {
            #    k: round(v, 2)
            #    for k, v in zip(
            #        ds_dict["mteb_dataset_name_with_lang"], ds_dict["score"]
            #    )
            #}
            ## metrics[0] is the main name for this metric; other names in the list are legacy for backward-compat
            #EXTERNAL_MODEL_RESULTS[model][task][metrics[0]].append(
            #    {**base_dict, **ds_dict}
            #)

    # Save & cache EXTERNAL_MODEL_RESULTS
    with open("EXTERNAL_MODEL_RESULTS.json", "w") as f:
        json.dump(dict(sorted(EXTERNAL_MODEL_RESULTS.items())), f, indent=4)

    return EXTERNAL_MODEL_RESULTS


def download_or_use_cache(modelId: str):
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


def simplify_dataset_name(name):
    return name.replace("MTEB ", "").replace(" (default)", "")


def get_mteb_data(
    tasks: list = ["Clustering"],
    langs: list = [],
    datasets: list = [],
    fillna: bool = True,
    add_emb_dim: bool = True,
    task_to_metric: dict = TASK_TO_METRIC,
    rank: bool = True,
) -> pd.DataFrame:
    global MODEL_INFOS

    with open("EXTERNAL_MODEL_RESULTS.json", "r") as f:
        external_model_results = json.load(f)

    api = API
    models = list(api.list_models(filter="mteb", full=True))
    # Legacy names changes; Also fetch the old results & merge later
    if "MLSUMClusteringP2P (fr)" in datasets:
        datasets.append("MLSUMClusteringP2P")
    if "MLSUMClusteringS2S (fr)" in datasets:
        datasets.append("MLSUMClusteringS2S")
    if "PawsXPairClassification (fr)" in datasets:
        datasets.append("PawsX (fr)")
    # Initialize list to models that we cannot fetch metadata from
    df_list = []
    for model in external_model_results:
        results_list = []
        for task in tasks:
            # Not all models have InstructionRetrieval, other new tasks
            if task not in external_model_results[model]: continue
            if task_to_metric[task][0] not in external_model_results[model][task]: continue
            results_list += external_model_results[model][task][task_to_metric[task][0]]

        if len(datasets) > 0:
            res = {
                k: v
                for d in results_list
                for k, v in d.items()
                if (k == "Model") or any([x in k for x in datasets])
            }
        elif langs:
            # Would be cleaner to rely on an extra language column instead
            langs_format = [f"({lang})" for lang in langs]
            res = {
                k: v
                for d in results_list
                for k, v in d.items()
                if any([k.split(" ")[-1] in (k, x) for x in langs_format])
            }
        else:
            res = {k: v for d in results_list for k, v in d.items()}
        # Model & at least one result
        if len(res) > 1:
            if add_emb_dim:
                res["Model Size (Million Parameters)"] = EXTERNAL_MODEL_TO_SIZE.get(
                    model, ""
                )
                res["Memory Usage (GB, fp32)"] = (
                    round(res["Model Size (Million Parameters)"] * 1e6 * 4 / 1024**3, 2)
                    if res["Model Size (Million Parameters)"] != ""
                    else ""
                )
                res["Embedding Dimensions"] = EXTERNAL_MODEL_TO_DIM.get(model, "")
                res["Max Tokens"] = EXTERNAL_MODEL_TO_SEQLEN.get(model, "")
            df_list.append(res)

    pbar = tqdm(models, desc="Fetching model metadata")
    for model in pbar:
        if model.modelId in MODELS_TO_SKIP:
            continue
        pbar.set_description(f"Fetching {model.modelId!r} metadata")
        meta = download_or_use_cache(model.modelId)
        MODEL_INFOS[model.modelId] = {"metadata": meta}
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
            task_results = [
                sub_res
                for sub_res in meta["model-index"][0]["results"]
                if (sub_res.get("task", {}).get("type", "") in tasks)
                and any(
                    [x in sub_res.get("dataset", {}).get("name", "") for x in datasets]
                )
            ]
        elif langs:
            task_results = [
                sub_res
                for sub_res in meta["model-index"][0]["results"]
                if (sub_res.get("task", {}).get("type", "") in tasks)
                and (
                    sub_res.get("dataset", {}).get("config", "default")
                    in ("default", *langs)
                )
            ]
        else:
            task_results = [
                sub_res
                for sub_res in meta["model-index"][0]["results"]
                if (sub_res.get("task", {}).get("type", "") in tasks)
            ]
        try:
            out = [
                {
                    simplify_dataset_name(res["dataset"]["name"]): [
                        round(score["value"], 2)
                        for score in res["metrics"]
                        if filter_metric_fetched(
                            simplify_dataset_name(res["dataset"]["name"]),
                            score["type"],
                            task_to_metric.get(res["task"]["type"]),
                            res["dataset"]["split"],
                        )
                    ][0]
                }
                for res in task_results
            ]
        except Exception as e:
            if 'ILKT' in model.modelId: continue
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
                        round(
                            EXTERNAL_MODEL_TO_SIZE[name_without_org]
                            * 1e6
                            * 4
                            / 1024**3,
                            2,
                        )
                        if name_without_org in EXTERNAL_MODEL_TO_SIZE
                        else "",
                    )
                (
                    out["Embedding Dimensions"],
                    out["Max Tokens"],
                    out["Model Size (Million Parameters)"],
                    out["Memory Usage (GB, fp32)"],
                ) = tuple(MODEL_INFOS[model.modelId]["dim_seq_size"])
            df_list.append(out)
        model_siblings = model.siblings or []
        if (
            model.library_name == "sentence-transformers"
            or "sentence-transformers" in model.tags
            or "modules.json" in {file.rfilename for file in model_siblings}
        ):
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
    base_columns = [
        "Model",
        "Model Size (Million Parameters)",
        "Memory Usage (GB, fp32)",
        "Embedding Dimensions",
        "Max Tokens",
    ]
    if len(datasets) > 0:
        # Update legacy column names to be merged with newer ones
        # Update 'MLSUMClusteringP2P (fr)' with values from 'MLSUMClusteringP2P'
        if ("MLSUMClusteringP2P (fr)" in datasets) and ("MLSUMClusteringP2P" in cols):
            df["MLSUMClusteringP2P (fr)"] = df["MLSUMClusteringP2P (fr)"].fillna(
                df["MLSUMClusteringP2P"]
            )
            datasets.remove("MLSUMClusteringP2P")
        if ("MLSUMClusteringS2S (fr)" in datasets) and ("MLSUMClusteringS2S" in cols):
            df["MLSUMClusteringS2S (fr)"] = df["MLSUMClusteringS2S (fr)"].fillna(
                df["MLSUMClusteringS2S"]
            )
            datasets.remove("MLSUMClusteringS2S")
        if ("PawsXPairClassification (fr)" in datasets) and ("PawsX (fr)" in cols):
            # for the first bit no model has it, hence no column for it. We can remove this in a month or so
            if "PawsXPairClassification (fr)" not in cols:
                df["PawsXPairClassification (fr)"] = df["PawsX (fr)"]
            else:
                df["PawsXPairClassification (fr)"] = df[
                    "PawsXPairClassification (fr)"
                ].fillna(df["PawsX (fr)"])
            # make all the columns the same
            datasets.remove("PawsX (fr)")
            cols.remove("PawsX (fr)")
            df.drop(columns=["PawsX (fr)"], inplace=True)

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
def get_mteb_average(task_dict: dict) -> tuple[Any, dict]:
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
    DATA_OVERALL.insert(
        1,
        f"Average ({len(all_tasks)} datasets)",
        DATA_OVERALL[all_tasks].mean(axis=1, skipna=False),
    )

    for i, (task_category, task_category_list) in enumerate(task_dict.items()):
        DATA_OVERALL.insert(
            i + 2,
            f"{task_category} Average ({len(task_category_list)} datasets)",
            DATA_OVERALL[task_category_list].mean(axis=1, skipna=False),
        )
    DATA_OVERALL.sort_values(
        f"Average ({len(all_tasks)} datasets)", ascending=False, inplace=True
    )
    # Start ranking from 1
    DATA_OVERALL.insert(0, "Rank", list(range(1, len(DATA_OVERALL) + 1)))

    DATA_OVERALL = DATA_OVERALL.round(2)

    DATA_TASKS = {}
    for task_category, task_category_list in task_dict.items():
        DATA_TASKS[task_category] = add_rank(
            DATA_OVERALL[
                ["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens"] + task_category_list
            ]
        )
        DATA_TASKS[task_category] = DATA_TASKS[task_category][
            DATA_TASKS[task_category].iloc[:, 4:].ne("").any(axis=1)
        ]

    # Fill NaN after averaging
    DATA_OVERALL.fillna("", inplace=True)

    data_overall_rows = [
        "Rank",
        "Model",
        "Model Size (Million Parameters)",
        "Memory Usage (GB, fp32)",
        "Embedding Dimensions",
        "Max Tokens",
        f"Average ({len(all_tasks)} datasets)",
    ]
    for task_category, task_category_list in task_dict.items():
        data_overall_rows.append(
            f"{task_category} Average ({len(task_category_list)} datasets)"
        )

    DATA_OVERALL = DATA_OVERALL[data_overall_rows]
    DATA_OVERALL = DATA_OVERALL[DATA_OVERALL.iloc[:, 5:].ne("").any(axis=1)]

    return DATA_OVERALL, DATA_TASKS


def refresh_leaderboard() -> tuple[list, dict]:
    """
    The main code to refresh and calculate results for the leaderboard. It does this by fetching the results from the
        external models and the models in the leaderboard, then calculating the average scores for each task category.
    """

    # get external model results and cache them
    # NOTE: if your model results have changed, use this function to refresh them (see inside for details)
    get_external_model_results()

    boards_data = {}
    all_data_tasks = []
    pbar_tasks = tqdm(
        BOARDS_CONFIG.items(),
        desc="Fetching leaderboard results for ???",
        total=len(BOARDS_CONFIG),
        leave=True,
    )
    for board, board_config in pbar_tasks:
        # Optional fetch only for a specific board
        # if board != "ru": continue
        # Very hacky - should fix this as soon as possible
        if board == "bright_long":
            TASK_TO_METRIC["Retrieval"] = ["recall_at_1"]
        boards_data[board] = {"data_overall": None, "data_tasks": {}}
        pbar_tasks.set_description(f"Fetching leaderboard results for {board!r}")
        pbar_tasks.refresh()
        if board_config["has_overall"]:
            data_overall, data_tasks = get_mteb_average(board_config["tasks"])
            boards_data[board]["data_overall"] = data_overall
            boards_data[board]["data_tasks"] = data_tasks
            all_data_tasks.extend(data_tasks.values())
        else:
            for task_category, task_category_list in board_config["tasks"].items():
                data_task_category = get_mteb_data(
                    tasks=[task_category], datasets=task_category_list
                )
                boards_data[board]["data_tasks"][task_category] = data_task_category
                all_data_tasks.append(data_task_category)
        if board == "bright_long":
            TASK_TO_METRIC["Retrieval"] = ["ndcg_at_10"]
    return all_data_tasks, boards_data

def write_out_results(item: dict, item_name: str) -> None:
    """
    Due to their complex structure, let's recursively create subfolders until we reach the end
        of the item and then save the DFs as jsonl files

    Args:
        item: The item to save
        item_name: The name of the item
    """
    main_folder = item_name

    if isinstance(item, list):
        for i, v in enumerate(item):
            write_out_results(v, os.path.join(main_folder, str(i)))

    elif isinstance(item, dict):
        for key, value in item.items():
            if isinstance(value, dict):
                write_out_results(value, os.path.join(main_folder, key))
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    write_out_results(v, os.path.join(main_folder, key + str(i)))
            else:
                write_out_results(value, os.path.join(main_folder, key))

    elif isinstance(item, pd.DataFrame):
        print(f"Saving {main_folder} to {main_folder}/default.jsonl")
        os.makedirs(main_folder, exist_ok=True)

        if "index" not in item.columns:
            item.reset_index(inplace=True)
        item.to_json(f"{main_folder}/default.jsonl", orient="records", lines=True)

    elif isinstance(item, str):
        print(f"Saving {main_folder} to {main_folder}/default.txt")
        os.makedirs(main_folder, exist_ok=True)
        with open(f"{main_folder}/default.txt", "w") as f:
            f.write(item)

    elif item is None:
        # write out an empty file
        print(f"Saving {main_folder} to {main_folder}/default.txt")
        os.makedirs(main_folder, exist_ok=True)
        with open(f"{main_folder}/default.txt", "w") as f:
            f.write("")

    else:
        raise Exception(f"Unknown type {type(item)}")


def load_results(data_path: str) -> list | dict | pd.DataFrame | str | None:
    """
    Do the reverse of `write_out_results` to reconstruct the item

    Args:
        data_path: The path to the data to load

    Returns:
        The loaded data
    """
    if os.path.isdir(data_path):
        # if the folder just has numbers from 0 to N, load as a list
        all_files_in_dir = list(os.listdir(data_path))
        if set(all_files_in_dir) == set([str(i) for i in range(len(all_files_in_dir))]):
            ### the list case
            return [
                load_results(os.path.join(data_path, str(i)))
                for i in range(len(os.listdir(data_path)))
            ]
        else:
            if len(all_files_in_dir) == 1:
                file_name = all_files_in_dir[0]
                if file_name == "default.jsonl":
                    return load_results(os.path.join(data_path, file_name))
                else:  ### the dict case
                    return {file_name: load_results(os.path.join(data_path, file_name))}
            else:
                return {
                    file_name: load_results(os.path.join(data_path, file_name))
                    for file_name in all_files_in_dir
                }

    elif data_path.endswith(".jsonl"):
        df = pd.read_json(data_path, orient="records", lines=True)
        if "index" in df.columns:
            df = df.set_index("index")
        if "Memory Usage (GB, fp32)" in df.columns:
            df["Memory Usage (GB, fp32)"] = df["Memory Usage (GB, fp32)"].map(lambda value: round(value, 2) if isinstance(value, float) else value)
        return df

    else:
        with open(data_path, "r") as f:
            data = f.read()
        if data == "":
            return None
        else:
            return data


if __name__ == "__main__":
    print("Refreshing leaderboard statistics...")
    all_data_tasks, boards_data = refresh_leaderboard()
    print("Done calculating, saving...")
    # save them so that the leaderboard can use them.  They're quite complex though
    #   but we can't use pickle files because of git-lfs.
    write_out_results(all_data_tasks, "all_data_tasks")
    write_out_results(boards_data, "boards_data")

    # to load them use
    # all_data_tasks = load_results("all_data_tasks")
    # boards_data = load_results("boards_data")
    print("Done saving results!")
