from functools import reduce
import re

import gradio as gr
import pandas as pd

from envs import REPO_ID
from refresh import BOARDS_CONFIG, TASKS, TASKS_CONFIG, TASK_DESCRIPTIONS, PRETTY_NAMES, load_results, make_clickable_model
from refresh import PROPRIETARY_MODELS, SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS, CROSS_ENCODERS, BI_ENCODERS, EXTERNAL_MODEL_TO_LINK


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
# print("','".join(MODEL_META["models_to_skip"]))
# print("','".join(list(MODEL_META['model_meta'].keys())))
UNUSED = ['michaelfeil/ct2fast-e5-large-v2','McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse','newsrx/instructor-xl','sionic-ai/sionic-ai-v1','lsf1000/bge-evaluation','Intel/bge-small-en-v1.5-sst2','newsrx/instructor-xl-newsrx','McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse','davidpeer/gte-small','goldenrooster/multilingual-e5-large','kozistr/fused-large-en','mixamrepijey/instructor-small','McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised','DecisionOptimizationSystem/DeepFeatEmbeddingLargeContext','Intel/bge-base-en-v1.5-sst2-int8-dynamic','morgendigital/multilingual-e5-large-quantized','BAAI/bge-small-en','ggrn/e5-small-v2','vectoriseai/gte-small','giulio98/placeholder','odunola/UAE-Large-VI','vectoriseai/e5-large-v2','gruber/e5-small-v2-ggml','Severian/nomic','arcdev/e5-mistral-7b-instruct','mlx-community/multilingual-e5-base-mlx','michaelfeil/ct2fast-bge-base-en-v1.5','Intel/bge-small-en-v1.5-sst2-int8-static','jncraton/stella-base-en-v2-ct2-int8','vectoriseai/multilingual-e5-large','rlsChapters/Chapters-SFR-Embedding-Mistral','arcdev/SFR-Embedding-Mistral','McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised','vectoriseai/gte-base','mixamrepijey/instructor-models','GovCompete/e5-large-v2','ef-zulla/e5-multi-sml-torch','khoa-klaytn/bge-small-en-v1.5-angle','krilecy/e5-mistral-7b-instruct','vectoriseai/bge-base-en-v1.5','vectoriseai/instructor-base','jingyeom/korean_embedding_model','rizki/bgr-tf','barisaydin/bge-base-en','jamesgpt1/zzz','Malmuk1/e5-large-v2_Sharded','vectoriseai/ember-v1','Consensus/instructor-base','barisaydin/bge-small-en','barisaydin/gte-base','woody72/multilingual-e5-base','Einas/einas_ashkar','michaelfeil/ct2fast-bge-large-en-v1.5','vectoriseai/bge-small-en-v1.5','iampanda/Test','cherubhao/yogamodel','ieasybooks/multilingual-e5-large-onnx','jncraton/e5-small-v2-ct2-int8','radames/e5-large','khoa-klaytn/bge-base-en-v1.5-angle','Intel/bge-base-en-v1.5-sst2-int8-static','vectoriseai/e5-large','TitanML/jina-v2-base-en-embed','Koat/gte-tiny','binqiangliu/EmbeddingModlebgelargeENv1.5','beademiguelperez/sentence-transformers-multilingual-e5-small','sionic-ai/sionic-ai-v2','jamesdborin/jina-v2-base-en-embed','maiyad/multilingual-e5-small','dmlls/all-mpnet-base-v2','odunola/e5-base-v2','vectoriseai/bge-large-en-v1.5','vectoriseai/bge-small-en','karrar-alwaili/UAE-Large-V1','t12e/instructor-base','Frazic/udever-bloom-3b-sentence','Geolumina/instructor-xl','hsikchi/dump','recipe/embeddings','michaelfeil/ct2fast-bge-small-en-v1.5','ildodeltaRule/multilingual-e5-large','shubham-bgi/UAE-Large','BAAI/bge-large-en','michaelfeil/ct2fast-e5-small-v2','cgldo/semanticClone','barisaydin/gte-small','aident-ai/bge-base-en-onnx','jamesgpt1/english-large-v1','michaelfeil/ct2fast-e5-small','baseplate/instructor-large-1','newsrx/instructor-large','Narsil/bge-base-en','michaelfeil/ct2fast-e5-large','mlx-community/multilingual-e5-small-mlx','lightbird-ai/nomic','MaziyarPanahi/GritLM-8x7B-GGUF','newsrx/instructor-large-newsrx','dhairya0907/thenlper-get-large','barisaydin/bge-large-en','jncraton/bge-small-en-ct2-int8','retrainai/instructor-xl','BAAI/bge-base-en','gentlebowl/instructor-large-safetensors','d0rj/e5-large-en-ru','atian-chapters/Chapters-SFR-Embedding-Mistral','Intel/bge-base-en-v1.5-sts-int8-static','Intel/bge-base-en-v1.5-sts-int8-dynamic','jncraton/GIST-small-Embedding-v0-ct2-int8','jncraton/gte-tiny-ct2-int8','d0rj/e5-small-en-ru','vectoriseai/e5-small-v2','SmartComponents/bge-micro-v2','michaelfeil/ct2fast-gte-base','vectoriseai/e5-base-v2','Intel/bge-base-en-v1.5-sst2','McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised','Research2NLP/electrical_stella','weakit-v/bge-base-en-v1.5-onnx','GovCompete/instructor-xl','barisaydin/text2vec-base-multilingual','Intel/bge-small-en-v1.5-sst2-int8-dynamic','jncraton/gte-small-ct2-int8','d0rj/e5-base-en-ru','barisaydin/gte-large','fresha/e5-large-v2-endpoint','vectoriseai/instructor-large','Severian/embed','vectoriseai/e5-base','mlx-community/multilingual-e5-large-mlx','vectoriseai/gte-large','anttip/ct2fast-e5-small-v2-hfie','michaelfeil/ct2fast-gte-large','gizmo-ai/Cohere-embed-multilingual-v3.0','McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse','Kenknight1999/tungdd7_ft_e5','joteqwork/new_gsev0','vantagediscovery/jina-embeddings-v2-base-en','vantagediscovery/nomic-embed-text-v1','vantagediscovery/nomic-embed-text-v1.5','srikanthmalla/hkunlp-instructor-xl','afrideva/GIST-all-MiniLM-L6-v2-GGUF','nadeem1362/mxbai-embed-large-v1-Q4_K_M-GGUF','agier9/gte-Qwen1.5-7B-instruct-Q5_K_M-GGUF','ekorman-strive/bge-large-en-v1.5','raghavlight/SE_v1','liddlefish/privacyembeddingv2_bge_small','ahmet1338/finetuned_embedder','radia/snowflake-arctic-embed-l-Q4_K_M-GGUF','GregorBiswanger/GritLM-7B-Q4_K_M-GGUF','powermove72/GritLM-7B-Q4_K_M-GGUF','sunzx0810/gte-Qwen2-7B-instruct-Q5_K_M-GGUF','nazimali/gte-Qwen2-7B-instruct-Q6_K-GGUF','nazimali/gte-Qwen2-7B-instruct-Q6_K-GGUF','fishbone64/gte-Qwen2-7B-instruct-Q8_0-GGUF','tobchef/gte-Qwen2-1.5B-instruct-Q4_K_M-GGUF','liddlefish/privacy_embedding_rag','liddlefish/privacy_embedding_rag_10k_tmp','liddlefish/privacy_embedding_bge_small_synthetic','mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF','leonn71/gte-Qwen2-1.5B-instruct-Q6_K-GGUF', 'Baichuan-text-embedding','Cohere-embed-english-v3.0','Cohere-embed-multilingual-light-v3.0','Cohere-embed-multilingual-v3.0','DanskBERT','FollowIR-7B','GritLM-7B','LASER2','LLM2Vec-Llama-2-supervised','LLM2Vec-Llama-2-unsupervised','LLM2Vec-Meta-Llama-3-supervised','LLM2Vec-Meta-Llama-3-unsupervised','LLM2Vec-Mistral-supervised','LLM2Vec-Mistral-unsupervised','LLM2Vec-Sheared-Llama-supervised','LLM2Vec-Sheared-Llama-unsupervised','LaBSE','OpenSearch-text-hybrid','SFR-Embedding-Mistral','all-MiniLM-L12-v2','all-MiniLM-L6-v2','all-mpnet-base-v2','allenai-specter','bert-base-10lang-cased','bert-base-15lang-cased','bert-base-25lang-cased','bert-base-multilingual-cased','bert-base-multilingual-uncased','bert-base-swedish-cased','bert-base-uncased','bge-base-zh-v1.5','bge-large-en-v1.5','bge-large-zh-noinstruct','bge-large-zh-v1.5','bge-m3','bge-small-zh-v1.5','bm25','camembert-base','camembert-large','contriever-base-msmarco','cross-en-de-roberta-sentence-transformer','dfm-encoder-large-v1','dfm-sentence-encoder-large-1','distilbert-base-25lang-cased','distilbert-base-en-fr-cased','distilbert-base-en-fr-es-pt-it-cased','distilbert-base-fr-cased','distilbert-base-uncased','distiluse-base-multilingual-cased-v2','e5-base-4k','e5-base-v2','e5-base','e5-large-v2','e5-large','e5-mistral-7b-instruct','e5-small','electra-small-nordic','electra-small-swedish-cased-discriminator','elser-v2','flan-t5-base','flan-t5-large','flaubert_base_cased','flaubert_base_uncased','flaubert_large_cased','gbert-base','gbert-large','gelectra-base','gelectra-large','glove.6B.300d','google-gecko-256.text-embedding-preview-0409','google-gecko.text-embedding-preview-0409','gottbert-base','gte-Qwen1.5-7B-instruct','gte-Qwen2-7B-instruct','gtr-t5-base','gtr-t5-large','gtr-t5-xl','gtr-t5-xxl','herbert-base-retrieval-v2','instructor-base','instructor-large','instructor-xl','jina-embeddings-v2-base-en','komninos','llama-2-7b-chat','luotuo-bert-medium','m3e-base','m3e-large','mistral-7b-instruct-v0.2','mistral-embed','monobert-large-msmarco','monot5-3b-msmarco-10k','monot5-base-msmarco-10k','msmarco-bert-co-condensor','multi-qa-MiniLM-L6-cos-v1','multilingual-e5-base','multilingual-e5-large','multilingual-e5-small','nb-bert-base','nb-bert-large','nomic-embed-text-v1','nomic-embed-text-v1.5-128','nomic-embed-text-v1.5-256','nomic-embed-text-v1.5-512','nomic-embed-text-v1.5-64','norbert3-base','norbert3-large','paraphrase-multilingual-MiniLM-L12-v2','paraphrase-multilingual-mpnet-base-v2','sentence-bert-swedish-cased','sentence-camembert-base','sentence-camembert-large','sentence-croissant-llm-base','sentence-t5-base','sentence-t5-large','sentence-t5-xl','sentence-t5-xxl','silver-retriever-base-v1','st-polish-paraphrase-from-distilroberta','st-polish-paraphrase-from-mpnet','sup-simcse-bert-base-uncased','text-embedding-3-large','text-embedding-3-large-256','text-embedding-3-small','text-embedding-ada-002','text-search-ada-001','text-search-ada-doc-001','text-search-ada-query-001','text-search-babbage-001','text-search-curie-001','text-search-davinci-001','text-similarity-ada-001','text-similarity-babbage-001','text-similarity-curie-001','text-similarity-davinci-001','tart-dual-contriever-msmarco','tart-full-flan-t5-xl','text2vec-base-chinese','text2vec-base-multilingual','text2vec-large-chinese','titan-embed-text-v1','udever-bloom-1b1','udever-bloom-560m','universal-sentence-encoder-multilingual-3','universal-sentence-encoder-multilingual-large-3','unsup-simcse-bert-base-uncased','use-cmlm-multilingual','voyage-2','voyage-code-2','voyage-large-2-instruct','voyage-law-2','voyage-lite-01-instruct','voyage-lite-02-instruct','voyage-multilingual-2','xlm-roberta-base','xlm-roberta-large','NV-Retriever-v1','NV-Embed-v1','Linq-Embed-Mistral']

# Possible changes:
# Could add graphs / other visual content
# Could add verification marks

# Sources:
# https://huggingface.co/spaces/gradio/leaderboard
# https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard
# https://getemoji.com/
