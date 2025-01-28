from functools import reduce
import re

import gradio as gr
import pandas as pd

from envs import REPO_ID
from refresh import BOARDS_CONFIG, TASKS, TASKS_CONFIG, TASK_DESCRIPTIONS, PRETTY_NAMES, load_results, make_clickable_model
from refresh import PROPRIETARY_MODELS, SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS, CROSS_ENCODERS, BI_ENCODERS, INSTRUCT_MODELS, NOINSTRUCT_MODELS, EXTERNAL_MODEL_TO_LINK


PROPRIETARY_MODELS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in PROPRIETARY_MODELS
}
SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS
}
INSTRUCT_MODELS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in INSTRUCT_MODELS
}
NOINSTRUCT_MODELS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in NOINSTRUCT_MODELS
}
CROSS_ENCODERS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in CROSS_ENCODERS
}
BI_ENCODERS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, f"https://huggingface.co/spaces/{REPO_ID}"))
    for model in BI_ENCODERS
}


def make_datasets_clickable(df):
    """Does not work"""
    if "BornholmBitextMining" in df.columns:
        link = "https://huggingface.co/datasets/strombergnlp/bornholmsk_parallel"
        df = df.rename(
            columns={f'BornholmBitextMining': '<a target="_blank" style="text-decoration: underline" href="{link}">BornholmBitextMining</a>',})
    return df


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
- desc: [optional] The description of the leaderboard
- data: The data for the leaderboard
"""

# No more refreshing manually, happens daily
# def get_refresh_function(task_category, task_list):
#     def _refresh():
#         data_task_category = get_mteb_data(tasks=[task_category], datasets=task_list)
#         data_task_category.drop(columns=["Embedding Dimensions", "Max Tokens"], inplace=True)
#         return data_task_category
#     return _refresh


# def get_refresh_overall_function(tasks):
#     return lambda: get_mteb_average(tasks)[0]


# load in the pre-calculated `all_data_tasks` and `boards_data`
print(f"Loading pre-calculated data....")
all_data_tasks = load_results("all_data_tasks")
boards_data = load_results("boards_data")

#### Caclulate Metadata
# Exact, add all non-nan integer values for every dataset
NUM_SCORES = 0
DATASETS = []
MODELS = []
# LANGUAGES = []
for d in all_data_tasks:
    if isinstance(d, list) and len(d) == 0:
        continue
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
    desc = board_config.get("desc", None)

    if board_config["has_overall"]:
        overall_pretty_name = board_pretty_name
        if acronym is not None:
            overall_pretty_name += f" ({board_config['acronym']})"
        data["Overall"]["data"].append({
            "language": board_config["title"],
            "language_long": board_config["language_long"],
            "description": f"**Overall MTEB {overall_pretty_name}** 🔮{board_icon}",
            "data": boards_data[board]["data_overall"],
            # "refresh": get_refresh_overall_function(board_config["tasks"]),
            "credits": credits,
            "metric": metric,
            "desc": desc,
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
            # "refresh": get_refresh_function(task_category, task_category_list),
            "credits": credits,
            "metric": metric,
            "desc": desc,
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
    "Bi-Encoders",
    "Uses Instructions",
    "No Instructions",
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
                elif model_type == "Uses Instructions":
                    masks.append(df["Model"].isin(INSTRUCT_MODELS))
                elif model_type == "No Instructions":
                    masks.append(df["Model"].isin(NOINSTRUCT_MODELS))
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
    Massive Text Embedding Benchmark (MTEB) Leaderboard. To submit, refer to the <a href="https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md" target="_blank" style="text-decoration: underline">MTEB GitHub repository</a> 🤗 Refer to the [MTEB paper](https://arxiv.org/abs/2210.07316) for details on metrics, tasks and models. Also check out [MTEB Arena](https://huggingface.co/spaces/mteb/arena) ⚔️
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
            elem_classes=["filter-checkbox-group"],
            scale=3,
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
                                {"- **Description:** " + item['desc'] if ("desc" in item and item["desc"] is not None) else ''}
                                """)

                            with gr.Row():
                                datatype = ["number", "markdown"] + ["number"] * len(item["data"])
                                dataframe = gr.Dataframe(item["data"], datatype=datatype, type="pandas", height=500)
                                dataframes.append(dataframe)

                                full_dataframe = gr.Dataframe(item["data"], datatype=datatype, type="pandas", visible=False)
                                full_dataframes.append(full_dataframe)

                            # with gr.Row():
                            #     refresh_button = gr.Button("Refresh")
                            #     refresh_button.click(item["refresh"], inputs=None, outputs=dataframe, concurrency_limit=20)

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

# Add model names here so the mteb/leaderboard space shows up on their model page
# from envs import MODEL_META
# print("','".join(MODEL_META["models_to_skip"]))
# print("','".join(list(MODEL_META['model_meta'].keys())))
# print("','".join([x['link'].split("co/")[-1] for x in MODEL_META['model_meta'].values() if (x.get('link', None)) and ("huggingface.co" in x['link'])]))
# from envs import API; print("','".join([x.modelId for x in list(API.list_models(filter="mteb")) if x.modelId not in UNUSED]))
UNUSED = ['michaelfeil/ct2fast-e5-large-v2','McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse','newsrx/instructor-xl','sionic-ai/sionic-ai-v1','lsf1000/bge-evaluation','Intel/bge-small-en-v1.5-sst2','newsrx/instructor-xl-newsrx','McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse','davidpeer/gte-small','goldenrooster/multilingual-e5-large','kozistr/fused-large-en','mixamrepijey/instructor-small','McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised','DecisionOptimizationSystem/DeepFeatEmbeddingLargeContext','Intel/bge-base-en-v1.5-sst2-int8-dynamic','morgendigital/multilingual-e5-large-quantized','BAAI/bge-small-en','ggrn/e5-small-v2','vectoriseai/gte-small','giulio98/placeholder','odunola/UAE-Large-VI','vectoriseai/e5-large-v2','gruber/e5-small-v2-ggml','Severian/nomic','arcdev/e5-mistral-7b-instruct','mlx-community/multilingual-e5-base-mlx','michaelfeil/ct2fast-bge-base-en-v1.5','Intel/bge-small-en-v1.5-sst2-int8-static','jncraton/stella-base-en-v2-ct2-int8','vectoriseai/multilingual-e5-large','rlsChapters/Chapters-SFR-Embedding-Mistral','arcdev/SFR-Embedding-Mistral','McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised','vectoriseai/gte-base','mixamrepijey/instructor-models','GovCompete/e5-large-v2','ef-zulla/e5-multi-sml-torch','khoa-klaytn/bge-small-en-v1.5-angle','krilecy/e5-mistral-7b-instruct','vectoriseai/bge-base-en-v1.5','vectoriseai/instructor-base','jingyeom/korean_embedding_model','rizki/bgr-tf','barisaydin/bge-base-en','jamesgpt1/zzz','Malmuk1/e5-large-v2_Sharded','vectoriseai/ember-v1','Consensus/instructor-base','barisaydin/bge-small-en','barisaydin/gte-base','woody72/multilingual-e5-base','Einas/einas_ashkar','michaelfeil/ct2fast-bge-large-en-v1.5','vectoriseai/bge-small-en-v1.5','iampanda/Test','cherubhao/yogamodel','ieasybooks/multilingual-e5-large-onnx','jncraton/e5-small-v2-ct2-int8','radames/e5-large','khoa-klaytn/bge-base-en-v1.5-angle','Intel/bge-base-en-v1.5-sst2-int8-static','vectoriseai/e5-large','TitanML/jina-v2-base-en-embed','Koat/gte-tiny','binqiangliu/EmbeddingModlebgelargeENv1.5','beademiguelperez/sentence-transformers-multilingual-e5-small','sionic-ai/sionic-ai-v2','jamesdborin/jina-v2-base-en-embed','maiyad/multilingual-e5-small','dmlls/all-mpnet-base-v2','odunola/e5-base-v2','vectoriseai/bge-large-en-v1.5','vectoriseai/bge-small-en','karrar-alwaili/UAE-Large-V1','t12e/instructor-base','Frazic/udever-bloom-3b-sentence','Geolumina/instructor-xl','hsikchi/dump','recipe/embeddings','michaelfeil/ct2fast-bge-small-en-v1.5','ildodeltaRule/multilingual-e5-large','shubham-bgi/UAE-Large','BAAI/bge-large-en','michaelfeil/ct2fast-e5-small-v2','cgldo/semanticClone','barisaydin/gte-small','aident-ai/bge-base-en-onnx','jamesgpt1/english-large-v1','michaelfeil/ct2fast-e5-small','baseplate/instructor-large-1','newsrx/instructor-large','Narsil/bge-base-en','michaelfeil/ct2fast-e5-large','mlx-community/multilingual-e5-small-mlx','lightbird-ai/nomic','MaziyarPanahi/GritLM-8x7B-GGUF','newsrx/instructor-large-newsrx','dhairya0907/thenlper-get-large','barisaydin/bge-large-en','jncraton/bge-small-en-ct2-int8','retrainai/instructor-xl','BAAI/bge-base-en','gentlebowl/instructor-large-safetensors','d0rj/e5-large-en-ru','atian-chapters/Chapters-SFR-Embedding-Mistral','Intel/bge-base-en-v1.5-sts-int8-static','Intel/bge-base-en-v1.5-sts-int8-dynamic','jncraton/GIST-small-Embedding-v0-ct2-int8','jncraton/gte-tiny-ct2-int8','d0rj/e5-small-en-ru','vectoriseai/e5-small-v2','SmartComponents/bge-micro-v2','michaelfeil/ct2fast-gte-base','vectoriseai/e5-base-v2','Intel/bge-base-en-v1.5-sst2','McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised','Research2NLP/electrical_stella','weakit-v/bge-base-en-v1.5-onnx','GovCompete/instructor-xl','barisaydin/text2vec-base-multilingual','Intel/bge-small-en-v1.5-sst2-int8-dynamic','jncraton/gte-small-ct2-int8','d0rj/e5-base-en-ru','barisaydin/gte-large','fresha/e5-large-v2-endpoint','vectoriseai/instructor-large','Severian/embed','vectoriseai/e5-base','mlx-community/multilingual-e5-large-mlx','vectoriseai/gte-large','anttip/ct2fast-e5-small-v2-hfie','michaelfeil/ct2fast-gte-large','gizmo-ai/Cohere-embed-multilingual-v3.0','McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse','Kenknight1999/tungdd7_ft_e5','joteqwork/new_gsev0','vantagediscovery/jina-embeddings-v2-base-en','vantagediscovery/nomic-embed-text-v1','vantagediscovery/nomic-embed-text-v1.5','srikanthmalla/hkunlp-instructor-xl','afrideva/GIST-all-MiniLM-L6-v2-GGUF','nadeem1362/mxbai-embed-large-v1-Q4_K_M-GGUF','agier9/gte-Qwen1.5-7B-instruct-Q5_K_M-GGUF','ekorman-strive/bge-large-en-v1.5','raghavlight/SE_v1','liddlefish/privacyembeddingv2_bge_small','ahmet1338/finetuned_embedder','radia/snowflake-arctic-embed-l-Q4_K_M-GGUF','GregorBiswanger/GritLM-7B-Q4_K_M-GGUF','powermove72/GritLM-7B-Q4_K_M-GGUF','sunzx0810/gte-Qwen2-7B-instruct-Q5_K_M-GGUF','nazimali/gte-Qwen2-7B-instruct-Q6_K-GGUF','nazimali/gte-Qwen2-7B-instruct-Q6_K-GGUF','fishbone64/gte-Qwen2-7B-instruct-Q8_0-GGUF','tobchef/gte-Qwen2-1.5B-instruct-Q4_K_M-GGUF','liddlefish/privacy_embedding_rag','liddlefish/privacy_embedding_rag_10k_tmp','liddlefish/privacy_embedding_bge_small_synthetic','mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF','leonn71/gte-Qwen2-1.5B-instruct-Q6_K-GGUF', 'Baichuan-text-embedding','Cohere-embed-english-v3.0','Cohere-embed-multilingual-light-v3.0','Cohere-embed-multilingual-v3.0','DanskBERT','FollowIR-7B','GritLM-7B','LASER2','LLM2Vec-Llama-2-supervised','LLM2Vec-Llama-2-unsupervised','LLM2Vec-Meta-Llama-3-supervised','LLM2Vec-Meta-Llama-3-unsupervised','LLM2Vec-Mistral-supervised','LLM2Vec-Mistral-unsupervised','LLM2Vec-Sheared-Llama-supervised','LLM2Vec-Sheared-Llama-unsupervised','LaBSE','OpenSearch-text-hybrid','SFR-Embedding-Mistral','all-MiniLM-L12-v2','all-MiniLM-L6-v2','all-mpnet-base-v2','allenai-specter','bert-base-10lang-cased','bert-base-15lang-cased','bert-base-25lang-cased','bert-base-multilingual-cased','bert-base-multilingual-uncased','bert-base-swedish-cased','bert-base-uncased','bge-base-zh-v1.5','bge-large-en-v1.5','bge-large-zh-noinstruct','bge-large-zh-v1.5','bge-m3','bge-small-zh-v1.5','bm25','camembert-base','camembert-large','contriever-base-msmarco','cross-en-de-roberta-sentence-transformer','dfm-encoder-large-v1','dfm-sentence-encoder-large-1','distilbert-base-25lang-cased','distilbert-base-en-fr-cased','distilbert-base-en-fr-es-pt-it-cased','distilbert-base-fr-cased','distilbert-base-uncased','distiluse-base-multilingual-cased-v2','e5-base-4k','e5-base-v2','e5-base','e5-large-v2','e5-large','e5-mistral-7b-instruct','e5-small','electra-small-nordic','electra-small-swedish-cased-discriminator','elser-v2','flan-t5-base','flan-t5-large','flaubert_base_cased','flaubert_base_uncased','flaubert_large_cased','gbert-base','gbert-large','gelectra-base','gelectra-large','glove.6B.300d','google-gecko-256.text-embedding-004','google-gecko.text-embedding-004','gottbert-base','gte-Qwen1.5-7B-instruct','gte-Qwen2-7B-instruct','gtr-t5-base','gtr-t5-large','gtr-t5-xl','gtr-t5-xxl','herbert-base-retrieval-v2','instructor-base','instructor-large','instructor-xl','jina-embeddings-v2-base-en','komninos','llama-2-7b-chat','luotuo-bert-medium','m3e-base','m3e-large','mistral-7b-instruct-v0.2','mistral-embed','monobert-large-msmarco','monot5-3b-msmarco-10k','monot5-base-msmarco-10k','msmarco-bert-co-condensor','multi-qa-MiniLM-L6-cos-v1','multilingual-e5-base','multilingual-e5-large','multilingual-e5-small','nb-bert-base','nb-bert-large','nomic-embed-text-v1','nomic-embed-text-v1.5-128','nomic-embed-text-v1.5-256','nomic-embed-text-v1.5-512','nomic-embed-text-v1.5-64','norbert3-base','norbert3-large','paraphrase-multilingual-MiniLM-L12-v2','paraphrase-multilingual-mpnet-base-v2','sentence-bert-swedish-cased','sentence-camembert-base','sentence-camembert-large','sentence-croissant-llm-base','sentence-t5-base','sentence-t5-large','sentence-t5-xl','sentence-t5-xxl','silver-retriever-base-v1','st-polish-paraphrase-from-distilroberta','st-polish-paraphrase-from-mpnet','sup-simcse-bert-base-uncased','text-embedding-3-large','text-embedding-3-large-256','text-embedding-3-small','text-embedding-ada-002','text-search-ada-001','text-search-ada-doc-001','text-search-ada-query-001','text-search-babbage-001','text-search-curie-001','text-search-davinci-001','text-similarity-ada-001','text-similarity-babbage-001','text-similarity-curie-001','text-similarity-davinci-001','tart-dual-contriever-msmarco','tart-full-flan-t5-xl','text2vec-base-chinese','text2vec-base-multilingual','text2vec-large-chinese','titan-embed-text-v1','udever-bloom-1b1','udever-bloom-560m','universal-sentence-encoder-multilingual-3','universal-sentence-encoder-multilingual-large-3','unsup-simcse-bert-base-uncased','use-cmlm-multilingual','voyage-2','voyage-code-2','voyage-large-2-instruct','voyage-law-2','voyage-lite-01-instruct','voyage-lite-02-instruct','voyage-multilingual-2','xlm-roberta-base','xlm-roberta-large','NV-Retriever-v1','NV-Embed-v1','Linq-Embed-Mistral','Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit','Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit','Muennighoff/SGPT-125M-weightedmean-nli-bitfit','Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit','Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit','Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit','DMetaSoul/sbert-chinese-general-v1','bigscience-data/sgpt-bloom-1b7-nli','bigscience/sgpt-bloom-7b1-msmarco','aari1995/German_Semantic_STS_V2','intfloat/e5-small','hkunlp/instructor-large','hkunlp/instructor-base','hkunlp/instructor-xl','intfloat/e5-base','intfloat/e5-large','Shimin/yiyouliao','vprelovac/universal-sentence-encoder-multilingual-large-3','vprelovac/universal-sentence-encoder-multilingual-3','vprelovac/universal-sentence-encoder-4','vprelovac/universal-sentence-encoder-large-5','ManiShankar-AlpesAi/paraphrase-multilingual-mpnet-base-v2-KE_Sieve','nickprock/mmarco-bert-base-italian-uncased','intfloat/e5-small-v2','intfloat/e5-base-v2','intfloat/e5-large-v2','intfloat/multilingual-e5-base','Shimin/LLaMA-embeeding','Forbu14/openai_clip_embeddings','shibing624/text2vec-base-multilingual','consciousAI/cai-lunaris-text-embeddings','consciousAI/cai-stellaris-text-embeddings','intfloat/multilingual-e5-small','intfloat/multilingual-e5-large','jinaai/jina-embedding-s-en-v1','jinaai/jina-embedding-b-en-v1','jinaai/jina-embedding-l-en-v1','deepfile/embedder-100p','lixsh6/XLM-3B5-embedding','lixsh6/XLM-0B6-embedding','thenlper/gte-base','thenlper/gte-large','thenlper/gte-small','lixsh6/MegatronBert-1B3-embedding','facebook/SONAR','Hum-Works/lodestone-base-4096-v1','sensenova/piccolo-base-zh','sensenova/piccolo-large-zh','infgrad/stella-base-zh','infgrad/stella-large-zh','BAAI/bge-reranker-base','BAAI/bge-base-en-v1.5','BAAI/bge-large-en-v1.5','BAAI/bge-small-en-v1.5','BAAI/bge-reranker-large','mgoin/all-MiniLM-L6-v2-ds','neuralmagic/bge-small-en-v1.5-sparse','jinaai/jina-embeddings-v2-base-en','jinaai/jina-embeddings-v2-small-en','neuralmagic/bge-small-en-v1.5-quant','nickprock/stsbm-sentence-flare-it','nickprock/mmarco-sentence-flare-it','neuralmagic/bge-base-en-v1.5-sparse','neuralmagic/bge-base-en-v1.5-quant','neuralmagic/bge-large-en-v1.5-sparse','neuralmagic/bge-large-en-v1.5-quant','TaylorAI/gte-tiny','TaylorAI/bge-micro','llmrails/ember-v1','TaylorAI/bge-micro-v2','zeroshot/gte-small-quant','infgrad/stella-large-zh-v2','infgrad/stella-base-zh-v2','zeroshot/gte-large-quant','zeroshot/gte-large-sparse','EdwardBurgin/paraphrase-multilingual-mpnet-base-v2','Amu/tao','infgrad/stella-base-en-v2','djovak/multi-qa-MiniLM-L6-cos-v1','izhx/udever-bloom-560m','izhx/udever-bloom-1b1','izhx/udever-bloom-3b','izhx/udever-bloom-7b1','thtang/ALL_862873','andersonbcdefg/bge-small-4096','Cohere/Cohere-embed-multilingual-light-v3.0','Cohere/Cohere-embed-multilingual-v3.0','Cohere/Cohere-embed-english-light-v3.0','Cohere/Cohere-embed-english-v3.0','Amu/tao-8k','thenlper/gte-large-zh','thenlper/gte-base-zh','thenlper/gte-small-zh','jamesgpt1/sf_model_e5','OrlikB/st-polish-kartonberta-base-alpha-v1','TownsWu/PEG','sdadas/mmlw-e5-small','sdadas/mmlw-e5-base','sdadas/mmlw-e5-large','sdadas/mmlw-roberta-base','sdadas/mmlw-roberta-large','jinaai/jina-embeddings-v2-base-code','aws-neuron/bge-base-en-v1-5-seqlen-384-bs-1','Erin/mist-zh','ClayAtlas/winberta-base','Pristinenlp/alime-reranker-large-zh','WhereIsAI/UAE-Large-V1','OrdalieTech/Solon-embeddings-large-0.1','ClayAtlas/winberta-large','intfloat/e5-mistral-7b-instruct','liujiarik/lim_base_zh','RookieHX/bge_m3e_stella','akarum/cloudy-large-zh','zhou-xl/bi-cse','lier007/xiaobu-embedding','jinaai/jina-embeddings-v2-base-zh','jinaai/jina-embeddings-v2-base-de','nomic-ai/nomic-embed-text-v1-ablated','nomic-ai/nomic-embed-text-v1-unsupervised','mukaj/fin-mpnet-base','Pristinenlp/alime-embedding-large-zh','pascalhuerten/instructor-skillfit','jinaai/jina-embeddings-v2-base-es','Salesforce/SFR-Embedding-Mistral','DMetaSoul/Dmeta-embedding-zh','Xenova/jina-embeddings-v2-base-zh','Xenova/jina-embeddings-v2-base-de','avsolatorio/GIST-Embedding-v0','nomic-ai/nomic-embed-text-v1','avsolatorio/GIST-all-MiniLM-L6-v2','avsolatorio/GIST-small-Embedding-v0','biswa921/bge-m3','Jechto/e5-dansk-test-0.1','intfloat/multilingual-e5-large-instruct','tanmaylaud/ret-phi2-v0','nomic-ai/nomic-embed-text-v1.5','GritLM/GritLM-7B','GritLM/GritLM-8x7B','avsolatorio/GIST-large-Embedding-v0','ClayAtlas/windberta-large','infgrad/stella-base-zh-v3-1792d','dunzhang/stella-large-zh-v3-1792d','jspringer/echo-mistral-7b-instruct-lasttoken','dunzhang/stella-mrl-large-zh-v3.5-1792d','sentosa/ZNV-Embedding','Nehc/e5-large-ru','neofung/m3e-ernie-xbase-zh','mixedbread-ai/mxbai-embed-2d-large-v1','mixedbread-ai/mxbai-embed-large-v1','aspire/acge_text_embedding','manu/sentence_croissant_alpha_v0.1','wongctroman/hktv-fine-tuned-cloudy-large-zh-metaphor14','manu/sentence_croissant_alpha_v0.2','mradermacher/GritLM-8x7B-GGUF','jhu-clsp/FollowIR-7B','DMetaSoul/Dmeta-embedding-zh-small','dwzhu/e5-base-4k','McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp','McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp','McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp','ChristianAzinn/uae-large-v1-gguf','ChristianAzinn/gist-large-embedding-v0-gguf','ChristianAzinn/bge-base-en-v1.5-gguf','ChristianAzinn/bge-small-en-v1.5-gguf','ChristianAzinn/bge-large-en-v1.5-gguf','ChristianAzinn/gte-base-gguf','ChristianAzinn/gte-large-gguf','ChristianAzinn/gte-small-gguf','ChristianAzinn/mxbai-embed-large-v1-gguf','ChristianAzinn/gist-small-embedding-v0-gguf','ChristianAzinn/e5-base-v2-gguf','ChristianAzinn/e5-large-v2-gguf','ChristianAzinn/e5-small-v2-gguf','ChristianAzinn/labse-gguf','srikanthmalla/BAAI-bge-reranker-large','Snowflake/snowflake-arctic-embed-m','manu/bge-m3-custom-fr','Snowflake/snowflake-arctic-embed-m-long','Snowflake/snowflake-arctic-embed-s','Snowflake/snowflake-arctic-embed-xs','Snowflake/snowflake-arctic-embed-l','ChristianAzinn/snowflake-arctic-embed-l-gguf','ChristianAzinn/snowflake-arctic-embed-m-long-GGUF','ChristianAzinn/snowflake-arctic-embed-m-gguf','ChristianAzinn/snowflake-arctic-embed-s-gguf','ChristianAzinn/snowflake-arctic-embed-xs-gguf','dwzhu/e5rope-base','pengql/checkpoint-9000','Alibaba-NLP/gte-base-en-v1.5','Alibaba-NLP/gte-large-en-v1.5','Alibaba-NLP/gte-Qwen1.5-7B-instruct','sensenova/piccolo-large-zh-v2','Mihaiii/gte-micro','NLPArtisan/qwen-1.8b-retrieval-test','Mihaiii/gte-micro-v2','Mihaiii/gte-micro-v3','Mihaiii/gte-micro-v4','Mihaiii/Taximetristi-2023','manu/sentence_croissant_alpha_v0.3','Mihaiii/Bulbasaur','Mihaiii/Ivysaur','manu/sentence_croissant_alpha_v0.4','Mihaiii/Venusaur','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','amazon/Titan-text-embeddings-v2','Mihaiii/Squirtle','Mihaiii/Wartortle','avsolatorio/NoInstruct-small-Embedding-v0','Mihaiii/test24','Mihaiii/test25','yessilver/new_model','fine-tuned/jina-embeddings-v2-base-en-03052024-r5ez-webapp','fine-tuned/jina-embeddings-v2-base-en-03052024-c20v-webapp','fine-tuned/jina-embeddings-v2-base-en-03052024-x8ew-webapp','fine-tuned/jina-embeddings-v2-base-en-03052024-73xx-webapp','fine-tuned/jina-embeddings-v2-base-en-03052024-21on-webapp','fine-tuned/jina-embeddings-v2-base-en-03052024-0swb-webapp','corto-ai/nomic-embed-text-v1','fine-tuned/jina-embeddings-v2-base-en-06052024-lmgf-webapp','fine-tuned/jina-embeddings-v2-base-en-06052024-6bdu-webapp','fine-tuned/jina-embeddings-v2-base-en-06052024-5pdj-webapp','fine-tuned/jina-embeddings-v2-base-en-06052024-yl1z-webapp','fine-tuned/jina-embeddings-v2-base-en-652024-vsmg-webapp','fine-tuned/jina-embeddings-v2-base-en-06052024-ruwi-webapp','fine-tuned/test','fine-tuned/jina-embeddings-v2-base-code-06052024-mhal-webapp','fine-tuned/jina-embeddings-v2-base-en-562024-j9xx-webapp','fine-tuned/jina-embeddings-v2-base-en-572024-xg53-webapp','fine-tuned/jina-embeddings-v2-base-en-202457-oc31-webapp','fine-tuned/scientific_papers_from_arxiv','fine-tuned/coding','fine-tuned/very_specific_technical_questions_about_Ubuntu','fine-tuned/CMedQAv2-reranking-improved','Labib11/MUG-B-1.6','shhy1995/AGE_Hybrid','fine-tuned/jina-embeddings-v2-base-en-10052024-lns6-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-scientific-papers-from-arxiv','fine-tuned/jinaai_jina-embeddings-v2-base-code-askubuntu','fine-tuned/jinaai_jina-embeddings-v2-base-en-scidocs','fine-tuned/jinaai_jina-embeddings-v2-base-code-stackoverflow','fine-tuned/jina-embeddings-v2-base-en-5102024-kvgq-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-zh-CMedQAv2','fine-tuned/jina-embeddings-v2-base-code-11_05_2024-hbxc-webapp','fine-tuned/jina-embeddings-v2-base-en-5102024-h7o7-webapp','fine-tuned/CMedQAv2-3','michaelfeil/jina-embeddings-v2-base-code','fine-tuned/jina-embeddings-v2-base-en-2024512-wvj9-webapp','fine-tuned/jina-embeddings-v2-base-en-5122024-3toh-webapp','MoMonir/SFR-Embedding-Mistral-GGUF','technicolor/Angle_BERT','fine-tuned/jina-embeddings-v2-base-en-2024513-kkxa-webapp','fine-tuned/jina-embeddings-v2-base-en-13052024-35bv-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-code-jinaai_jina-embeddings-v2-base-cod','fine-tuned/jinaai_jina-embeddings-v2-base-en-jinaai_jina-embeddings-v2-base-en-sc','fine-tuned/jinaai_jina-embeddings-v2-base-zh-jinaai_jina-embeddings-v2-base-zh-CM','fine-tuned/jinaai_jina-embeddings-v2-base-zh-CMedQAv2-3','fine-tuned/scidocs','fine-tuned/askubuntu','fine-tuned/stackoverflow','fine-tuned/cmedqav2','fine-tuned/jina-embeddings-v2-base-en-13052024-ch9n-webapp','fine-tuned/askubuntu-c','fine-tuned/askubuntu-l','fine-tuned/scidocs-c','fine-tuned/stackoverflow-c','fine-tuned/cmedqav2-c','fine-tuned/norwegian-nli-triplets-c','AdrienB134/llm2vec-croissant-mntp','Erin/IYun-large-zh','fine-tuned/jina-embeddings-v2-base-en-14052024-5b5o-webapp','fine-tuned/jina-embeddings-v2-base-en-14052024-9xxb-webapp','fine-tuned/jina-embeddings-v2-base-en-14052024-afuz-webapp','fine-tuned/dutch-legal-c','AdrienB134/llm2vec-occiglot-mntp','fine-tuned/dutch-legal-c-64-24','w601sxs/b1ade-embed','fine-tuned/dutch-legal-c-1280-24','neofung/bge-reranker-large-1k','fine-tuned/askubuntu-c-128-24','fine-tuned/askubuntu-c-256-24','fine-tuned/stackoverflow-c-128-24','fine-tuned/cmedqav2-c-128-24','fine-tuned/scidocs-c-128-24','fine-tuned/dutch-legal-c-128-24','fine-tuned/scidocs-c-256-24','fine-tuned/stackoverflow-c-256-24','qihoo360/360Zhinao-search','fine-tuned/stackoverflow-c-64-24','fine-tuned/askubuntu-c-64-24','fine-tuned/scidocs-c-64-24','fine-tuned/cmedqav2-c-64-24','fine-tuned/jina-embeddings-v2-base-en-15052024-stsl-webapp','fine-tuned/jina-embeddings-v2-base-en-5152024-tsbl-webapp','fine-tuned/jina-embeddings-v2-base-en-5162024-o9um-webapp','fine-tuned/test-run','fine-tuned/stackoverflow-c-64-24-gpt-4o-2024-05-13','MoMonir/gte-Qwen1.5-7B-instruct-GGUF','fine-tuned/scidocs-c-64-24-gpt-4o-2024-05-133652','fine-tuned/askubuntu-c-64-24-gpt-4o-2024-05-135760','fine-tuned/stackoverflow-c-64-24-gpt-4o-2024-05-137765','fine-tuned/scidocs-c-64-24-gpt-4o-2024-05-13-46337','fine-tuned/askubuntu-c-64-24-gpt-4o-2024-05-131171','fine-tuned/scidocs-c-64-24-gpt-4o-2024-05-135334','fine-tuned/askubuntu-c-64-24-gpt-4o-2024-05-13-61285','fine-tuned/cmedqav2-c-64-24-gpt-4o-2024-05-13-50353','fine-tuned/jina-embeddings-v2-base-en-1752024-13s3-webapp','fine-tuned/jina-embeddings-v2-base-en-1752024-zdtc-webapp','fine-tuned/jina-embeddings-v2-base-en-17052024-uhub-webapp','neofung/bge-reranker-base-1k','fine-tuned/jina-embeddings-v2-base-en-17052024-dumr-webapp','fine-tuned/arguana-c-64-24-gpt-4o-2024-05-136897','fine-tuned/arguana-c-64-24-gpt-4o-2024-05-136538','fine-tuned/arguana-c-128-24-gpt-4o-2024-05-13-68212','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-51550','fine-tuned/jina-embeddings-v2-base-en-19052024-oiu8-webapp','fine-tuned/jina-embeddings-v2-base-en-5192024-xqq9-webapp','fine-tuned/jina-embeddings-v2-base-en-5192024-qeye-webapp','fine-tuned/jina-embeddings-v2-base-en-5192024-seuc-webapp','qihoo360/360Zhinao-1.8B-Reranking','fine-tuned/jina-embeddings-v2-base-en-5202024-55bm-webapp','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-693632','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-819563','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-413214','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-129048','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-550302','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-799305','fine-tuned/jina-embeddings-v2-base-en-5202024-6tkj-webapp','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-264015','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-994439','fine-tuned/jina-embeddings-v2-base-en-5202024-rxyq-webapp','jinaai/jina-clip-v1','fine-tuned/jina-embeddings-v2-base-en-21052024-5qm5-webapp','dayyass/universal-sentence-encoder-multilingual-large-3-pytorch','fine-tuned/jina-embeddings-v2-base-en-21052024-5smg-webapp','fine-tuned/jina-embeddings-v2-base-en-22052024-vuno-webapp','fine-tuned/arguana-c-256-24-gpt-4o-2024-05-13-387094','fine-tuned/LegalBenchConsumerContractsQA-256-24-gpt-4o-2024-05-13-292605','fine-tuned/LegalBenchCorporateLobbying-256-24-gpt-4o-2024-05-13-296144','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-454852','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-79875','fine-tuned/TRECCOVID-256-24-gpt-4o-2024-05-13-190413','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-727361','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-410031','fine-tuned/jina-embeddings-v2-base-code-5222024-i8af-webapp','fine-tuned/jina-embeddings-v2-base-en-5222024-hkde-webapp','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-14719','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-526066','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-10630','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-825318','nvidia/NV-Embed-v1','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-203779','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-497939','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-417900','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-994884','fine-tuned/jina-embeddings-v2-base-en-23052024-hbdj-webapp','fine-tuned/jina-embeddings-v2-base-en-23052024-6kfw-webapp','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-214114','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-587313','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-36954','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-814821','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-256742','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-317735','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-378237','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-992459','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-552473','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-816730','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-875153','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-630221','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-214478','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-645586','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-786584','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-785172','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-687872','fine-tuned/BAAI_bge-small-en-v1_5-23052024-upq5-webapp','fine-tuned/NFCorpus-8-8-gpt-4o-2024-05-13-855191','fine-tuned/NFCorpus-8-8-gpt-4o-2024-05-13-978964','fine-tuned/NFCorpus-8-8-gpt-4o-2024-05-13-847943','fine-tuned/NFCorpus-8-8-gpt-4o-2024-05-13-449863','fine-tuned/NFCorpus-8-8-gpt-4o-2024-05-13-610535','fine-tuned/NFCorpus-8-8-gpt-4o-2024-05-13-322852','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-898550','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-546049','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-499715','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-598568','fine-tuned/BAAI_bge-large-en-v1_5-5242024-5uvy-webapp','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-304829','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-138515','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-269096','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-778232','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-111876','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-292803','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-96776','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-67198','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-310581','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-449834','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-737659','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-976783','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-27685','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-54716','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-166315','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-812157','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-133486','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-423936','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-772252','w601sxs/b1ade-embed-kd','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-141246','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-478897','fine-tuned/ArguAna-256-24-gpt-4o-2024-05-13-952023','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-157892','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-780826','fine-tuned/TRECCOVID-256-24-gpt-4o-2024-05-13-475598','fine-tuned/QuoraRetrieval-256-24-gpt-4o-2024-05-13-635320','fine-tuned/Touche2020-256-24-gpt-4o-2024-05-13-27907','fine-tuned/BAAI_bge-small-en-v1_5-5252024-jzfp-webapp','fine-tuned/TRECCOVID-256-24-gpt-4o-2024-05-13-953989','fine-tuned/ArguAna-256-24-gpt-4o-2024-05-13-413991','fine-tuned/QuoraRetrieval-256-24-gpt-4o-2024-05-13-80208','fine-tuned/SciFact-256-24-gpt-4o-2024-05-13-484582','fine-tuned/FiQA2018-256-24-gpt-4o-2024-05-13-919917','fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-988957','fine-tuned/SCIDOCS-256-24-gpt-4o-2024-05-13-597314','fine-tuned/TRECCOVID-256-24-gpt-4o-2024-05-13-896673','fine-tuned/ArguAna-256-24-gpt-4o-2024-05-13-689823','fine-tuned/BAAI_bge-small-en-v1_5-5272024-2fs4-webapp','fine-tuned/BAAI_bge-small-en-v1_5-27052024-4e8w-webapp','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-890333','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-140539','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-2499','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-733782','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-221689','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-465198','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-698531','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-548936','fine-tuned/BAAI_bge-small-en-v1_5-5272024-ou25-webapp','agier9/UAE-Large-V1-Q5_K_S-GGUF','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-69882','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-822545','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-268697','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-43315','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-866232','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-580978','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-115380','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-985263','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-439294','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-607244','fine-tuned/TRECCOVID-512-192-gpt-4o-2024-05-13-347397','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-650620','fine-tuned/QuoraRetrieval-512-192-gpt-4o-2024-05-13-777321','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-73934','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-14571','fine-tuned/TRECCOVID-512-192-gpt-4o-2024-05-13-653452','fine-tuned/QuoraRetrieval-512-192-gpt-4o-2024-05-13-768442','fine-tuned/BAAI_bge-small-en-v1_5-5282024-hkt5-webapp','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-100928','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-906438','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-266507','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-93805','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-424608','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-710799','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-357185','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-873132','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-452456','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-143735','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-625238','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-186741','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-935443','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-418918','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-110174','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-859511','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-437825','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-986812','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-37395','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-591725','fine-tuned/BAAI_bge-small-en-v1_5-2852024-6p16-webapp','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-93651135','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-89953157','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-23636059','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-83930416','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-27692546','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-76823162','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-89836585','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-28032241','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-34914559','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-10552781','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-44219785','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-60453771','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-34917964','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-24541174','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-20151707','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-26543668','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-3292683','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-14028623','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-378068','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-27258064','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-79168271','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-80780135','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-42468142','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-47583376','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-80745457','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-34699555','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-35912','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-6089388','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-43473113','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-31581583','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-79659206','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-51211577','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-53785794','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-37851926','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-93507731','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-24464680','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-1134151','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-87401391','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-76679499','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-58211433','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-56351634','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-87403910','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-67485775','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-8421720','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-50444055','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-67948597','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-63275487','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-90390391','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-16241583','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-86331274','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-53403987','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-3465370','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-19100452','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-83904142','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-37125303','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-94762694','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-20768519','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-35609715','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-14003539','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-16083606','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-3973638','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-76839538','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-90164285','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-52015789','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-93248154','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-74504128','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-65608189','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-92012085','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-34898812','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-59792256','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-26737110','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-41821758','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-29425597','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-12907987','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-34642434','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-65268203','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-85722278','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-7975202','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-62563104','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-22039677','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-80948573','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-48400660','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-10086588','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-11626257','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-5953538','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-68485784','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-51991531','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-81928581','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-6825910','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-52686172','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-63983441','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-76979764','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-25305323','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-89774081','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-99342737','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-97839788','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-52238558','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-486134','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-46607440','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-80802988','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-67820659','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-37230491','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-64924747','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-17390035','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-66909812','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-67941497','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-95714065','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-65992666','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-89826544','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-74939490','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-16883408','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-40695234','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-68577224','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-47339454','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-36338558','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-17911388','fine-tuned/FiQA2018-512-192-gpt-4o-2024-05-13-97777963','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-51883844','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-24419258','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-34427772','Linq-AI-Research/Linq-Embed-Mistral','fine-tuned/ArguAna-512-192-gpt-4o-2024-05-13-14562627','fine-tuned/SCIDOCS-512-192-gpt-4o-2024-05-13-37833293','fine-tuned/before-finetuning-512-192-gpt-4o-2024-05-13-65274313','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-99421248','fine-tuned/NFCorpus-512-192-gpt-4o-2024-05-13-67596481','fine-tuned/SciFact-512-192-gpt-4o-2024-05-13-3038586','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-18360524','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-73143156','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-20584918','fine-tuned/FiQA2018-32000-384-gpt-4o-2024-05-13-52831585','fine-tuned/SciFact-32000-384-gpt-4o-2024-05-13-45622553','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-45587246','fine-tuned/ArguAna-32000-384-gpt-4o-2024-05-13-39088299','fine-tuned/SCIDOCS-32000-384-gpt-4o-2024-05-13-5483216','fine-tuned/FiQA2018-32000-384-gpt-4o-2024-05-13-23538198','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-94858978','fine-tuned/SciFact-32000-384-gpt-4o-2024-05-13-25926506','fine-tuned/ArguAna-32000-384-gpt-4o-2024-05-13-60385830','fine-tuned/SCIDOCS-32000-384-gpt-4o-2024-05-13-19472313','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-1216656','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-39265981','fine-tuned/SciFact-32000-384-gpt-4o-2024-05-13-76083984','fine-tuned/SCIDOCS-32000-384-gpt-4o-2024-05-13-97946708','fine-tuned/FiQA2018-32000-384-gpt-4o-2024-05-13-66633416','fine-tuned/ArguAna-32000-384-gpt-4o-2024-05-13-13220755','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-62034393','Classical/Yinka','fine-tuned/BAAI_bge-small-en-v1_5-30052024-rc2l-webapp','fine-tuned/ArguAna-32000-384-gpt-4o-2024-05-13-55034819','twadada/tst','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-2553188','fine-tuned/FiQA2018-32000-384-gpt-4o-2024-05-13-28832324','fine-tuned/ArguAna-32000-384-gpt-4o-2024-05-13-50573159','fine-tuned/SCIDOCS-32000-384-gpt-4o-2024-05-13-38097330','fine-tuned/SciFact-32000-384-gpt-4o-2024-05-13-66747460','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-48618256','fine-tuned/BAAI_bge-small-en-v1_5-612024-vf79-webapp','fine-tuned/BAAI_bge-small-en-v1_5-632024-34lw-webapp','corto-ai/bge-reranker-large-onnx','fine-tuned/BAAI_bge-small-en-v1_5-04062024-hsmq-webapp','iampanda/zpoint_large_embedding_zh','silverjam/jina-embeddings-v2-base-zh','fine-tuned/jinaai_jina-embeddings-v2-base-en-05062024-16gq-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-05062024-445b-webapp','neofung/LdIR-reranker-large','fine-tuned/jinaai_jina-embeddings-v2-base-en-05062024-zvoa-webapp','fine-tuned/BAAI_bge-small-en-v1_5-05062024-x987-webapp','fine-tuned/deepspeed-from-new-new-docker','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-86786922','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-59074949','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-55567015','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-67199932','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-24297328','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-81211802','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-7385160','fine-tuned/FiQA2018-32000-384-gpt-4o-2024-05-13-74794049','fine-tuned/SciFact-32000-384-gpt-4o-2024-05-13-42885533','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-27359624','fine-tuned/ArguAna-32000-384-gpt-4o-2024-05-13-35162543','fine-tuned/SCIDOCS-32000-384-gpt-4o-2024-05-13-33133286','fine-tuned/FiQA2018-32000-384-gpt-4o-2024-05-13-83115388','fine-tuned/SciFact-32000-384-gpt-4o-2024-05-13-41822019','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-66131574','fine-tuned/ArguAna-32000-384-gpt-4o-2024-05-13-68388407','fine-tuned/SCIDOCS-32000-384-gpt-4o-2024-05-13-71434542','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-6875032','fine-tuned/before-finetuning-32000-384-gpt-4o-2024-05-13-91940173','fine-tuned/NFCorpus-32000-384-gpt-4o-2024-05-13-70846146','fine-tuned/BAAI_bge-large-en-v1_5-672024-v51y-webapp','Gameselo/STS-multilingual-mpnet-base-v2','itod/UAE-Large-V1-Q8_0-GGUF','fine-tuned/jinaai_jina-embeddings-v2-base-en-08062024-z8ik-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-202469-tgjk-webapp','liddlefish/privacy_embedding_rag_10k_base_checkpoint_2','liddlefish/privacy_embedding_rag_10k_base_final','w601sxs/b1ade-embed-kd_3','fine-tuned/jinaai_jina-embeddings-v2-base-en-6112024-fmxr-webapp','liddlefish/privacy_embedding_rag_10k_base_15_final','liddlefish/privacy_embedding_rag_10k_base_12_final','fine-tuned/BAAI_bge-m3-6122024-ibs3-webapp','fine-tuned/BAAI_bge-m3-2024__6__12_-1217-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-es-6122024-fv1x-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-6122024-bhm2-webapp','fine-tuned/BAAI_bge-large-en-v1_5-1362024-2wos-webapp','raghavlight/TDTE','fine-tuned/jinaai_jina-embeddings-v2-base-en-6132024-wvrg-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-6132024-bez1-webapp','fine-tuned/BAAI_bge-large-en-1362024-gcw6-webapp','fine-tuned/BAAI_bge-base-en-1362024-n19c-webapp','fine-tuned/BAAI_bge-m3-1362024-m82b-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-6142024-huet-webapp','fine-tuned/BAAI_bge-m3-6142024-0ndt-webapp','fine-tuned/BAAI_bge-large-en-v1_5-14062024-fimj-webapp','Omartificial-Intelligence-Space/Arabic-all-nli-triplet-Matryoshka','CAiRE/UniVaR-lambda-80','CAiRE/UniVaR-lambda-20','CAiRE/UniVaR-lambda-5','CAiRE/UniVaR-lambda-1','fine-tuned/BAAI_bge-large-en-v1_5-14062024-xdwa-webapp','Salesforce/SFR-Embedding-2_R','fine-tuned/BAAI_bge-large-en-15062024-atex-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-2024615-ioyu-webapp','ILKT/2024-06-15_10-09-42','Alibaba-NLP/gte-Qwen2-7B-instruct','fine-tuned/BAAI_bge-large-en-v1_5-1562024-to89-webapp','Omartificial-Intelligence-Space/Arabic-mpnet-base-all-nli-triplet','fine-tuned/jinaai_jina-embeddings-v2-base-en-6162024-xxse-webapp','Omartificial-Intelligence-Space/Arabic-labse-Matryoshka','Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka','Omartificial-Intelligence-Space/Marbert-all-nli-triplet-Matryoshka','ILKT/2024-06-17_21-37-12','fine-tuned/BAAI_bge-small-en-v1_5-18062024-56t5-webapp','ILKT/2024-06-19_08-22-22','ILKT/2024-06-19_10-03-38','fine-tuned/jinaai_jina-embeddings-v2-base-en-6192024-56os-webapp','ILKT/2024-06-19_21-12-17','ILKT/2024-06-19_22-27-15','ILKT/2024-06-19_22-23-38','fine-tuned/jinaai_jina-embeddings-v2-base-en-20062024-djhb-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-20062024-t2n9-webapp','ILKT/2024-06-20_12-31-59','fine-tuned/BAAI_bge-large-en-2062024-u43q-webapp','ILKT/2024-06-20_12-31-55','tomaarsen/jina-clip-v1-st','tomaarsen/jina-clip-v1-st-remote','fine-tuned/jinaai_jina-embeddings-v2-base-en-6212024-p8j6-webapp','ILKT/2024-06-22_12-37-29_epoch_1','ILKT/2024-06-22_12-37-29_epoch_2','ILKT/2024-06-22_12-37-29_epoch_3','ILKT/2024-06-22_12-37-29_epoch_4','ILKT/2024-06-22_12-37-29_epoch_5','fine-tuned/jinaai_jina-embeddings-v2-base-es-22062024-taeu-webapp','ILKT/2024-06-22_12-37-29_epoch_6','ILKT/2024-06-22_12-37-29_epoch_7','ILKT/2024-06-22_12-37-29_epoch_8','ILKT/2024-06-22_12-37-29_epoch_9','ILKT/2024-06-22_12-37-29_epoch_10','ILKT/2024-06-22_12-37-29_epoch_11','ILKT/2024-06-22_12-37-29_epoch_12','fine-tuned/jinaai_jina-embeddings-v2-base-en-6232024-zldx-webapp','ILKT/2024-06-22_12-37-29_epoch_13','ILKT/2024-06-22_12-37-29_epoch_14','ILKT/2024-06-23_09-09-07_epoch_1','ILKT/2024-06-22_12-37-29_epoch_15','ILKT/2024-06-23_09-09-07_epoch_2','ILKT/2024-06-23_09-09-07_epoch_3','ILKT/2024-06-23_09-09-07_epoch_4','ILKT/2024-06-23_09-09-07_epoch_5','ILKT/2024-06-23_09-09-07_epoch_6','ILKT/2024-06-23_09-09-07_epoch_7','ILKT/2024-06-23_09-09-07_epoch_8','fine-tuned/BAAI_bge-m3-6232024-4vtf-webapp','ILKT/2024-06-23_09-09-07_epoch_9','ILKT/2024-06-24_00-11-56_epoch_1','ILKT/2024-06-23_09-09-07_epoch_10','ILKT/2024-06-24_00-11-56_epoch_2','ILKT/2024-06-23_09-09-07_epoch_11','ILKT/2024-06-24_00-11-56_epoch_3','ILKT/2024-06-24_00-11-56_epoch_4','ILKT/2024-06-23_09-09-07_epoch_12','ILKT/2024-06-24_00-11-56_epoch_5','ILKT/2024-06-23_09-09-07_epoch_13','ILKT/2024-06-24_00-11-56_epoch_6','ILKT/2024-06-24_00-11-56_epoch_7','Lajavaness/bilingual-embedding-large','fine-tuned/jinaai_jina-embeddings-v2-base-en-24_06_2024-lrip-webapp','ILKT/2024-06-24_22-31-18_epoch_1','ILKT/2024-06-24_22-31-28_epoch_1','ILKT/2024-06-24_22-31-18_epoch_2','ILKT/2024-06-24_22-31-28_epoch_2','ILKT/2024-06-24_22-31-18_epoch_3','ILKT/2024-06-24_22-31-28_epoch_3','ILKT/2024-06-24_22-31-18_epoch_4','ILKT/2024-06-24_22-31-28_epoch_4','ILKT/2024-06-24_22-31-18_epoch_5','ILKT/2024-06-24_22-31-28_epoch_5','ILKT/2024-06-24_22-31-18_epoch_6','ILKT/2024-06-24_22-31-28_epoch_6','ILKT/2024-06-24_22-31-18_epoch_7','ILKT/2024-06-24_22-31-28_epoch_7','ILKT/2024-06-24_22-31-18_epoch_8','ILKT/2024-06-24_22-31-28_epoch_8','ILKT/2024-06-24_22-31-18_epoch_9','ILKT/2024-06-24_22-31-28_epoch_9','ILKT/2024-06-24_22-31-18_epoch_10','ILKT/2024-06-24_22-31-28_epoch_10','ILKT/2024-06-24_22-31-18_epoch_11','ILKT/2024-06-24_22-31-28_epoch_11','ILKT/2024-06-24_22-31-18_epoch_12','ILKT/2024-06-24_22-31-28_epoch_12','ILKT/2024-06-24_22-31-18_epoch_13','ILKT/2024-06-24_22-31-28_epoch_13','ILKT/2024-06-24_22-31-18_epoch_14','ILKT/2024-06-24_22-31-28_epoch_14','ILKT/2024-06-24_22-31-18_epoch_15','ILKT/2024-06-24_22-31-28_epoch_15','ILKT/2024-06-24_22-31-18_epoch_16','ILKT/2024-06-24_22-31-28_epoch_16','ILKT/2024-06-24_22-31-18_epoch_17','ILKT/2024-06-24_22-31-28_epoch_17','ILKT/2024-06-24_22-31-18_epoch_18','ILKT/2024-06-24_22-31-28_epoch_18','ILKT/2024-06-24_22-31-18_epoch_19','ILKT/2024-06-24_22-31-28_epoch_19','ILKT/2024-06-24_22-31-18_epoch_20','ILKT/2024-06-24_22-31-28_epoch_20','ILKT/2024-06-24_22-31-18_epoch_21','ILKT/2024-06-24_22-31-28_epoch_21','ILKT/2024-06-24_22-31-18_epoch_22','ILKT/2024-06-24_22-31-28_epoch_22','ILKT/2024-06-24_22-31-18_epoch_23','ILKT/2024-06-24_22-31-28_epoch_23','ILKT/2024-06-24_22-31-18_epoch_24','ILKT/2024-06-24_22-31-28_epoch_24','ILKT/2024-06-24_22-31-18_epoch_25','ILKT/2024-06-24_22-31-28_epoch_25','ILKT/2024-06-24_22-31-18_epoch_26','ILKT/2024-06-24_22-31-28_epoch_26','ILKT/2024-06-24_22-31-18_epoch_27','ILKT/2024-06-24_22-31-28_epoch_27','ILKT/2024-06-24_22-31-18_epoch_28','ILKT/2024-06-24_22-31-28_epoch_28','ILKT/2024-06-24_22-31-18_epoch_29','ILKT/2024-06-24_22-31-28_epoch_29','ILKT/2024-06-24_22-31-18_epoch_30','Lenovo-Zhihui/Zhihui_LLM_Embedding','ILKT/2024-06-24_22-31-28_epoch_30','ILKT/2024-06-24_22-31-18_epoch_31','ILKT/2024-06-24_22-31-28_epoch_31','ILKT/2024-06-24_22-31-18_epoch_32','ILKT/2024-06-24_22-31-28_epoch_32','ILKT/2024-06-24_22-31-18_epoch_33','ILKT/2024-06-24_22-31-28_epoch_33','ILKT/2024-06-24_22-31-18_epoch_34','ILKT/2024-06-24_22-31-28_epoch_34','ILKT/2024-06-24_22-31-18_epoch_35','ILKT/2024-06-24_22-31-28_epoch_35','ILKT/2024-06-24_22-31-18_epoch_36','ILKT/2024-06-24_22-31-28_epoch_36','ILKT/2024-06-24_22-31-18_epoch_37','ILKT/2024-06-24_22-31-28_epoch_37','ILKT/2024-06-24_22-31-18_epoch_38','ILKT/2024-06-24_22-31-28_epoch_38','ILKT/2024-06-24_22-31-18_epoch_39','ILKT/2024-06-24_22-31-28_epoch_39','ILKT/2024-06-24_22-31-18_epoch_40','ILKT/2024-06-24_22-31-28_epoch_40','ILKT/2024-06-24_22-31-18_epoch_41','ILKT/2024-06-24_22-31-28_epoch_41','ILKT/2024-06-24_22-31-18_epoch_42','ILKT/2024-06-24_22-31-28_epoch_42','ILKT/2024-06-24_22-31-18_epoch_43','ILKT/2024-06-24_22-31-28_epoch_43','ILKT/2024-06-24_22-31-18_epoch_44','ILKT/2024-06-24_22-31-28_epoch_44','ILKT/2024-06-24_22-31-18_epoch_45','ILKT/2024-06-24_22-31-28_epoch_45','ILKT/2024-06-24_22-31-18_epoch_46','ILKT/2024-06-24_22-31-28_epoch_46','ILKT/2024-06-24_22-31-18_epoch_47','ILKT/2024-06-24_22-31-28_epoch_47','ILKT/2024-06-24_22-31-18_epoch_48','ILKT/2024-06-24_22-31-28_epoch_48','ILKT/2024-06-24_22-31-18_epoch_49','ILKT/2024-06-24_22-31-28_epoch_49','ILKT/2024-06-24_22-31-18_epoch_50','ILKT/2024-06-24_22-31-28_epoch_50','ILKT/2024-06-24_22-31-18_epoch_51','ILKT/2024-06-24_22-31-28_epoch_51','ILKT/2024-06-24_22-31-18_epoch_52','ILKT/2024-06-24_22-31-28_epoch_52','ILKT/2024-06-24_22-31-18_epoch_53','ILKT/2024-06-24_22-31-28_epoch_53','ILKT/2024-06-24_22-31-18_epoch_54','ILKT/2024-06-24_22-31-28_epoch_54','ILKT/2024-06-24_22-31-18_epoch_55','ILKT/2024-06-24_22-31-28_epoch_55','ILKT/2024-06-24_22-31-18_epoch_56','ILKT/2024-06-24_22-31-28_epoch_56','ILKT/2024-06-24_22-31-18_epoch_57','ILKT/2024-06-24_22-31-28_epoch_57','ILKT/2024-06-24_22-31-18_epoch_58','ILKT/2024-06-24_22-31-28_epoch_58','ILKT/2024-06-24_22-31-18_epoch_59','ILKT/2024-06-24_22-31-28_epoch_59','ILKT/2024-06-24_22-31-18_epoch_60','ILKT/2024-06-24_22-31-28_epoch_60','ILKT/2024-06-24_22-31-18_epoch_61','ILKT/2024-06-24_22-31-28_epoch_61','ILKT/2024-06-24_22-31-18_epoch_62','ILKT/2024-06-24_22-31-28_epoch_62','ILKT/2024-06-24_22-31-18_epoch_63','ILKT/2024-06-24_22-31-28_epoch_63','ILKT/2024-06-24_22-31-18_epoch_64','ILKT/2024-06-24_22-31-28_epoch_64','ILKT/2024-06-24_22-31-18_epoch_65','ILKT/2024-06-24_22-31-28_epoch_65','ILKT/2024-06-24_22-31-18_epoch_66','ILKT/2024-06-24_22-31-28_epoch_66','ILKT/2024-06-24_22-31-18_epoch_67','Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet','ILKT/2024-06-24_22-31-28_epoch_67','ILKT/2024-06-24_22-31-18_epoch_68','ILKT/2024-06-24_22-31-28_epoch_68','ILKT/2024-06-24_22-31-18_epoch_69','ILKT/2024-06-24_22-31-28_epoch_69','ILKT/2024-06-24_22-31-18_epoch_70','ILKT/2024-06-24_22-31-28_epoch_70','ILKT/2024-06-24_22-31-18_epoch_71','ILKT/2024-06-24_22-31-28_epoch_71','ILKT/2024-06-24_22-31-18_epoch_72','ILKT/2024-06-24_22-31-28_epoch_72','ILKT/2024-06-24_22-31-18_epoch_73','ILKT/2024-06-24_22-31-28_epoch_73','ILKT/2024-06-24_22-31-18_epoch_74','ILKT/2024-06-24_22-31-28_epoch_74','ILKT/2024-06-24_22-31-18_epoch_75','Intel/neural-embedding-v1','ILKT/2024-06-24_22-31-28_epoch_75','fine-tuned/BAAI_bge-m3-26062024-gdon-webapp','Lajavaness/bilingual-embedding-base','fine-tuned/jinaai_jina-embeddings-v2-base-es-6262024-yjwm-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-6262024-wtkc-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-6272024-qn9b-webapp','BeastyZ/e5-R-mistral-7b','ILKT/2024-06-23_09-09-07_epoch_14','ILKT/2024-06-23_09-09-07_epoch_15','ILKT/2024-06-23_09-09-07_epoch_16','ILKT/2024-06-23_09-09-07_epoch_17','ILKT/2024-06-23_09-09-07_epoch_18','ILKT/2024-06-23_09-09-07_epoch_19','ILKT/2024-06-23_09-09-07_epoch_20','ILKT/2024-06-23_09-09-07_epoch_21','ILKT/2024-06-23_09-09-07_epoch_22','ILKT/2024-06-23_09-09-07_epoch_23','ILKT/2024-06-23_09-09-07_epoch_24','ILKT/2024-06-23_09-09-07_epoch_25','ILKT/2024-06-23_09-09-07_epoch_26','ILKT/2024-06-23_09-09-07_epoch_27','ILKT/2024-06-23_09-09-07_epoch_28','ILKT/2024-06-23_09-09-07_epoch_29','ILKT/2024-06-23_09-09-07_epoch_30','ILKT/2024-06-23_09-09-07_epoch_31','ILKT/2024-06-23_09-09-07_epoch_32','ILKT/2024-06-23_09-09-07_epoch_33','ILKT/2024-06-23_09-09-07_epoch_34','ILKT/2024-06-23_09-09-07_epoch_35','ILKT/2024-06-23_09-09-07_epoch_36','ILKT/2024-06-23_09-09-07_epoch_37','ILKT/2024-06-23_09-09-07_epoch_38','ILKT/2024-06-23_09-09-07_epoch_39','ILKT/2024-06-23_09-09-07_epoch_40','ILKT/2024-06-23_09-09-07_epoch_41','ILKT/2024-06-23_09-09-07_epoch_42','ILKT/2024-06-23_09-09-07_epoch_43','ILKT/2024-06-23_09-09-07_epoch_44','ILKT/2024-06-23_09-09-07_epoch_45','ILKT/2024-06-23_09-09-07_epoch_46','ILKT/2024-06-23_09-09-07_epoch_47','ILKT/2024-06-23_09-09-07_epoch_48','ILKT/2024-06-23_09-09-07_epoch_49','ILKT/2024-06-23_09-09-07_epoch_50','ILKT/2024-06-23_09-09-07_epoch_51','ILKT/2024-06-23_09-09-07_epoch_52','ILKT/2024-06-23_09-09-07_epoch_53','ILKT/2024-06-23_09-09-07_epoch_54','ILKT/2024-06-23_09-09-07_epoch_55','ILKT/2024-06-23_09-09-07_epoch_56','ILKT/2024-06-23_09-09-07_epoch_57','ILKT/2024-06-23_09-09-07_epoch_58','ILKT/2024-06-23_09-09-07_epoch_59','ILKT/2024-06-23_09-09-07_epoch_60','ILKT/2024-06-23_09-09-07_epoch_61','ILKT/2024-06-23_09-09-07_epoch_62','ILKT/2024-06-23_09-09-07_epoch_63','ILKT/2024-06-23_09-09-07_epoch_64','ILKT/2024-06-23_09-09-07_epoch_65','ILKT/2024-06-23_09-09-07_epoch_66','ILKT/2024-06-23_09-09-07_epoch_67','ILKT/2024-06-23_09-09-07_epoch_68','ILKT/2024-06-23_09-09-07_epoch_69','ILKT/2024-06-23_09-09-07_epoch_70','ILKT/2024-06-23_09-09-07_epoch_71','ILKT/2024-06-23_09-09-07_epoch_72','ILKT/2024-06-23_09-09-07_epoch_73','ILKT/2024-06-23_09-09-07_epoch_74','ILKT/2024-06-23_09-09-07_epoch_75','Pekarnick/e5-large-v2-Q4_K_M-GGUF','ILKT/2024-06-24_00-11-56_epoch_8','ILKT/2024-06-24_00-11-56_epoch_9','ILKT/2024-06-24_00-11-56_epoch_10','ILKT/2024-06-24_00-11-56_epoch_11','ILKT/2024-06-24_00-11-56_epoch_12','ILKT/2024-06-24_00-11-56_epoch_13','ILKT/2024-06-24_00-11-56_epoch_14','ILKT/2024-06-24_00-11-56_epoch_15','ILKT/2024-06-24_00-11-56_epoch_16','ILKT/2024-06-24_00-11-56_epoch_17','ILKT/2024-06-24_00-11-56_epoch_18','ILKT/2024-06-24_00-11-56_epoch_19','ILKT/2024-06-24_00-11-56_epoch_20','ILKT/2024-06-24_00-11-56_epoch_21','ILKT/2024-06-24_00-11-56_epoch_22','ILKT/2024-06-24_00-11-56_epoch_23','ILKT/2024-06-24_00-11-56_epoch_24','ILKT/2024-06-24_00-11-56_epoch_25','ILKT/2024-06-24_00-11-56_epoch_26','ILKT/2024-06-24_00-11-56_epoch_27','ILKT/2024-06-24_00-11-56_epoch_28','ILKT/2024-06-24_00-11-56_epoch_29','ILKT/2024-06-24_00-11-56_epoch_30','ILKT/2024-06-24_00-11-56_epoch_31','ILKT/2024-06-24_00-11-56_epoch_32','ILKT/2024-06-24_00-11-56_epoch_33','ILKT/2024-06-24_00-11-56_epoch_34','ILKT/2024-06-24_00-11-56_epoch_35','ILKT/2024-06-24_00-11-56_epoch_36','ILKT/2024-06-24_00-11-56_epoch_37','ILKT/2024-06-24_00-11-56_epoch_38','ILKT/2024-06-24_00-11-56_epoch_39','ILKT/2024-06-24_00-11-56_epoch_40','ILKT/2024-06-24_00-11-56_epoch_41','ILKT/2024-06-24_00-11-56_epoch_42','ILKT/2024-06-24_00-11-56_epoch_43','ILKT/2024-06-24_00-11-56_epoch_44','ILKT/2024-06-24_00-11-56_epoch_45','ILKT/2024-06-24_00-11-56_epoch_46','ILKT/2024-06-24_00-11-56_epoch_47','ILKT/2024-06-24_00-11-56_epoch_48','ILKT/2024-06-24_00-11-56_epoch_49','ILKT/2024-06-24_00-11-56_epoch_50','ILKT/2024-06-24_00-11-56_epoch_51','ILKT/2024-06-24_00-11-56_epoch_52','ILKT/2024-06-24_00-11-56_epoch_53','ILKT/2024-06-24_00-11-56_epoch_54','ILKT/2024-06-24_00-11-56_epoch_55','ILKT/2024-06-24_00-11-56_epoch_56','ILKT/2024-06-24_00-11-56_epoch_57','ILKT/2024-06-24_00-11-56_epoch_58','ILKT/2024-06-24_00-11-56_epoch_59','ILKT/2024-06-24_00-11-56_epoch_60','ILKT/2024-06-24_00-11-56_epoch_61','ILKT/2024-06-24_00-11-56_epoch_62','ILKT/2024-06-24_00-11-56_epoch_63','ILKT/2024-06-24_00-11-56_epoch_64','ILKT/2024-06-24_00-11-56_epoch_65','ILKT/2024-06-24_00-11-56_epoch_66','ILKT/2024-06-24_00-11-56_epoch_67','ILKT/2024-06-24_00-11-56_epoch_68','ILKT/2024-06-24_00-11-56_epoch_69','ILKT/2024-06-24_00-11-56_epoch_70','ILKT/2024-06-24_00-11-56_epoch_71','ILKT/2024-06-24_00-11-56_epoch_72','ILKT/2024-06-24_00-11-56_epoch_73','ILKT/2024-06-24_00-11-56_epoch_74','ILKT/2024-06-24_00-11-56_epoch_75','ILKT/2024-06-22_12-37-29_epoch_16','ILKT/2024-06-22_12-37-29_epoch_17','ILKT/2024-06-22_12-37-29_epoch_18','ILKT/2024-06-22_12-37-29_epoch_19','ILKT/2024-06-22_12-37-29_epoch_20','ILKT/2024-06-22_12-37-29_epoch_21','ILKT/2024-06-22_12-37-29_epoch_22','ILKT/2024-06-22_12-37-29_epoch_23','ILKT/2024-06-22_12-37-29_epoch_24','ILKT/2024-06-22_12-37-29_epoch_25','ILKT/2024-06-22_12-37-29_epoch_26','ILKT/2024-06-22_12-37-29_epoch_27','ILKT/2024-06-22_12-37-29_epoch_28','ILKT/2024-06-22_12-37-29_epoch_29','ILKT/2024-06-22_12-37-29_epoch_30','ILKT/2024-06-22_12-37-29_epoch_31','ILKT/2024-06-22_12-37-29_epoch_32','ILKT/2024-06-22_12-37-29_epoch_33','ILKT/2024-06-22_12-37-29_epoch_34','ILKT/2024-06-22_12-37-29_epoch_35','ILKT/2024-06-22_12-37-29_epoch_36','ILKT/2024-06-22_12-37-29_epoch_37','ILKT/2024-06-22_12-37-29_epoch_38','ILKT/2024-06-22_12-37-29_epoch_39','ILKT/2024-06-22_12-37-29_epoch_40','ILKT/2024-06-22_12-37-29_epoch_41','ILKT/2024-06-22_12-37-29_epoch_42','ILKT/2024-06-22_12-37-29_epoch_43','ILKT/2024-06-22_12-37-29_epoch_44','ILKT/2024-06-22_12-37-29_epoch_45','ILKT/2024-06-22_12-37-29_epoch_46','ILKT/2024-06-22_12-37-29_epoch_47','ILKT/2024-06-22_12-37-29_epoch_48','ILKT/2024-06-22_12-37-29_epoch_49','ILKT/2024-06-22_12-37-29_epoch_50','ILKT/2024-06-22_12-37-29_epoch_51','ILKT/2024-06-22_12-37-29_epoch_52','ILKT/2024-06-22_12-37-29_epoch_53','ILKT/2024-06-22_12-37-29_epoch_54','ILKT/2024-06-22_12-37-29_epoch_55','ILKT/2024-06-22_12-37-29_epoch_56','ILKT/2024-06-22_12-37-29_epoch_57','ILKT/2024-06-22_12-37-29_epoch_58','ILKT/2024-06-22_12-37-29_epoch_59','ILKT/2024-06-22_12-37-29_epoch_60','ILKT/2024-06-22_12-37-29_epoch_61','ILKT/2024-06-22_12-37-29_epoch_62','ILKT/2024-06-22_12-37-29_epoch_63','ILKT/2024-06-22_12-37-29_epoch_64','ILKT/2024-06-22_12-37-29_epoch_65','ILKT/2024-06-22_12-37-29_epoch_66','ILKT/2024-06-22_12-37-29_epoch_67','ILKT/2024-06-22_12-37-29_epoch_68','ILKT/2024-06-22_12-37-29_epoch_69','ILKT/2024-06-22_12-37-29_epoch_70','ILKT/2024-06-22_12-37-29_epoch_71','ILKT/2024-06-22_12-37-29_epoch_72','ILKT/2024-06-22_12-37-29_epoch_73','ILKT/2024-06-22_12-37-29_epoch_74','ILKT/2024-06-22_12-37-29_epoch_75','Lajavaness/bilingual-embedding-large-8k','Alibaba-NLP/gte-Qwen2-1.5B-instruct','Jaume/gemma-2b-embeddings','lier007/xiaobu-embedding-v2','chihlunLee/NoInstruct-small-Embedding-v0-Q4_0-GGUF','fine-tuned/jinaai_jina-embeddings-v2-base-es-472024-aqk1-webapp','second-state/gte-Qwen2-1.5B-instruct-GGUF','gaianet/gte-Qwen2-1.5B-instruct-GGUF','yco/bilingual-embedding-base','fine-tuned/jinaai_jina-embeddings-v2-base-en-05072024-aj6g-webapp','AbderrahmanSkiredj1/arabic_text_embedding_sts_arabertv02_arabicnlitriplet','AbderrahmanSkiredj1/Arabic_text_embedding_for_sts','dimcha/mxbai-embed-large-v1-Q4_K_M-GGUF','fine-tuned/BAAI_bge-m3-782024-wl54-webapp','nvidia/NV-Retriever-v1','fine-tuned/jinaai_jina-embeddings-v2-base-en-792024-tyen-webapp','fine-tuned/jinaai_jina-embeddings-v2-base-en-11072024-bh6v-webapp','archit28/bge-large-en-v1.5-Q4_K_S-GGUF','dunzhang/stella_en_1.5B_v5','dunzhang/stella_en_400M_v5','niancheng/gte-Qwen2-1.5B-instruct-Q4_K_M-GGUF','niancheng/gte-Qwen2-7B-instruct-Q4_K_M-GGUF','fine-tuned/jinaai_jina-embeddings-v2-base-en-15072024-5xy1-webapp','fine-tuned/BAAI_bge-small-en-v1_5-7152024-w1z0-webapp', 'Cohere/Cohere-embed-english-v3.0','Cohere/Cohere-embed-english-v3.0','Cohere/Cohere-embed-multilingual-light-v3.0','Cohere/Cohere-embed-multilingual-v3.0','vesteinn/DanskBERT','jhu-clsp/FollowIR-7B','GritLM/GritLM-7B','GritLM/GritLM-7B','McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised','McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse','McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised','McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse','McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised','McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse','sentence-transformers/LaBSE','Linq-AI-Research/Linq-Embed-Mistral','nvidia/NV-Embed-v1','nvidia/NV-Retriever-v1','Salesforce/SFR-Embedding-Mistral','sentence-transformers/all-MiniLM-L12-v2','sentence-transformers/all-MiniLM-L12-v2','sentence-transformers/all-MiniLM-L6-v2','sentence-transformers/all-MiniLM-L6-v2','sentence-transformers/all-mpnet-base-v2','sentence-transformers/all-mpnet-base-v2','sentence-transformers/allenai-specter','Geotrend/bert-base-10lang-cased','Geotrend/bert-base-15lang-cased','Geotrend/bert-base-25lang-cased','google-bert/bert-base-multilingual-cased','google-bert/bert-base-multilingual-uncased','KB/bert-base-swedish-cased','bert-base-uncased','BAAI/bge-base-en-v1.5','BAAI/bge-base-en-v1.5','BAAI/bge-base-zh-v1.5','BAAI/bge-large-en-v1.5','BAAI/bge-large-en-v1.5','BAAI/bge-large-zh-noinstruct','BAAI/bge-large-zh-v1.5','BAAI/bge-m3','BAAI/bge-m3','BAAI/bge-small-en-v1.5','BAAI/bge-small-en-v1.5','BAAI/bge-small-zh-v1.5','almanach/camembert-base','almanach/camembert-large','nthakur/contriever-base-msmarco','facebook/contriever','facebook/contriever','T-Systems-onsite/cross-en-de-roberta-sentence-transformer','chcaa/dfm-encoder-large-v1','chcaa/dfm-encoder-large-v1','Geotrend/distilbert-base-25lang-cased','Geotrend/distilbert-base-en-fr-cased','Geotrend/distilbert-base-en-fr-es-pt-it-cased','Geotrend/distilbert-base-fr-cased','distilbert-base-uncased','sentence-transformers/distiluse-base-multilingual-cased-v2','dwzhu/e5-base-4k','intfloat/e5-base-v2','intfloat/e5-base','intfloat/e5-large-v2','intfloat/e5-large','intfloat/e5-mistral-7b-instruct','intfloat/e5-mistral-7b-instruct-noinstruct','intfloat/e5-small','jonfd/electra-small-nordic','KBLab/electra-small-swedish-cased-discriminator','google/flan-t5-base','google/flan-t5-large','flaubert/flaubert_base_cased','flaubert/flaubert_base_uncased','flaubert/flaubert_large_cased','deepset/gbert-base','deepset/gbert-large','deepset/gelectra-base','deepset/gelectra-large','sentence-transformers/average_word_embeddings_glove.6B.300d','uklfr/gottbert-base','Alibaba-NLP/gte-Qwen1.5-7B-instruct','Alibaba-NLP/gte-Qwen2-7B-instruct','sentence-transformers/gtr-t5-base','sentence-transformers/gtr-t5-large','sentence-transformers/gtr-t5-xl','sentence-transformers/gtr-t5-xxl','ipipan/herbert-base-retrieval-v2','hkunlp/instructor-base','hkunlp/instructor-large','hkunlp/instructor-xl','jinaai/jina-embeddings-v2-base-en','sentence-transformers/average_word_embeddings_komninos','meta-llama/Llama-2-7b-chat-hf','silk-road/luotuo-bert-medium','moka-ai/m3e-base','moka-ai/m3e-large','mistralai/Mistral-7B-Instruct-v0.2','castorini/monobert-large-msmarco','castorini/monot5-3b-msmarco-10k','castorini/monot5-base-msmarco-10k','sentence-transformers/msmarco-bert-co-condensor','sentence-transformers/multi-qa-MiniLM-L6-cos-v1','intfloat/multilingual-e5-base','intfloat/multilingual-e5-large','intfloat/multilingual-e5-small','NbAiLab/nb-bert-base','NbAiLab/nb-bert-large','nomic-ai/nomic-embed-text-v1','nomic-ai/nomic-embed-text-v1.5','nomic-ai/nomic-embed-text-v1.5','nomic-ai/nomic-embed-text-v1.5','nomic-ai/nomic-embed-text-v1.5','ltg/norbert3-base','ltg/norbert3-large','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2','sentence-transformers/paraphrase-multilingual-mpnet-base-v2','KBLab/sentence-bert-swedish-cased','dangvantuan/sentence-camembert-base','dangvantuan/sentence-camembert-large','Wissam42/sentence-croissant-llm-base','sentence-transformers/sentence-t5-base','sentence-transformers/sentence-t5-large','sentence-transformers/sentence-t5-xl','sentence-transformers/sentence-t5-xxl','ipipan/silver-retriever-base-v1','sdadas/st-polish-paraphrase-from-distilroberta','sdadas/st-polish-paraphrase-from-mpnet','princeton-nlp/sup-simcse-bert-base-uncased','orionweller/tart-dual-contriever-msmarco','facebook/tart-full-flan-t5-xl','shibing624/text2vec-base-chinese','GanymedeNil/text2vec-large-chinese','izhx/udever-bloom-1b1','izhx/udever-bloom-560m','vprelovac/universal-sentence-encoder-multilingual-3','vprelovac/universal-sentence-encoder-multilingual-large-3','princeton-nlp/unsup-simcse-bert-base-uncased','sentence-transformers/use-cmlm-multilingual','xlm-roberta-base','xlm-roberta-large']

# Possible changes:
# Could add graphs / other visual content
# Could add verification marks

# Sources:
# https://huggingface.co/spaces/gradio/leaderboard
# https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard
# https://getemoji.com/
