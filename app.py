from datasets import load_dataset
import gradio as gr
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import metadata_load
import pandas as pd

TASKS = [
    "BitextMining",
    "Classification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
]

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification (en)",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification (en)",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification (en)",
    "MassiveScenarioClassification (en)",
    "MTOPDomainClassification (en)",
    "MTOPIntentClassification (en)",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLASSIFICATION_NORM = [x.replace(" (en)", "") for x in TASK_LIST_CLASSIFICATION]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_RETRIEVAL_NORM = TASK_LIST_RETRIEVAL + ["CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval"
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17 (en-en)",
    "STS22 (en)",
    "STSBenchmark",
]

TASK_LIST_STS_NORM = [x.replace(" (en)", "").replace(" (en-en)", "") for x in TASK_LIST_STS]

TASK_LIST_SUMMARIZATION = [
    "SummEval",
]

TASK_LIST_EN = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS + TASK_LIST_SUMMARIZATION

TASK_TO_METRIC = {
    "BitextMining": "f1",
    "Clustering": "v_measure",
    "Classification": "accuracy",
    "PairClassification": "cos_sim_ap",
    "Reranking": "map",
    "Retrieval": "ndcg_at_10",
    "STS": "cos_sim_spearman",
    "Summarization": "cos_sim_spearman",
}

def make_clickable_model(model_name, link=None):
    # Remove user from model name
    model_name = model_name.split("/")[-1]
    if link is None:
        link = "https://huggingface.co/" + model_name
    return (
        f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name}</a>'
    )

# Models without metadata, thus we cannot fetch their results naturally
EXTERNAL_MODELS = [
    "LASER2",
    "LaBSE",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "allenai-specter",
    "bert-base-uncased",
    "contriever-base-msmarco",
    "glove.6B.300d",
    "gtr-t5-base",
    "gtr-t5-large",
    "gtr-t5-xl",
    "gtr-t5-xxl",
    "komninos",
    "msmarco-bert-co-condensor",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "sentence-t5-base",
    "sentence-t5-large",
    "sentence-t5-xl",
    "sentence-t5-xxl",
    "sgpt-bloom-1b3-nli",
    "sgpt-bloom-7b1-msmarco",
    "sgpt-nli-bloom-1b3",
    "sup-simcse-bert-base-uncased",
#    "text-similarity-ada-001",
    "unsup-simcse-bert-base-uncased",
]
EXTERNAL_MODEL_TO_LINK = {
    "LASER2": "https://github.com/facebookresearch/LASER",
    "text-similarity-ada-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
}
EXTERNAL_MODEL_RESULTS = {model: {k: {v: []} for k, v in TASK_TO_METRIC.items()} for model in EXTERNAL_MODELS}

def add_lang(examples):
    if not(examples["eval_language"]):
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"]
    else:
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"] + f' ({examples["eval_language"]})'
    return examples

def add_task(examples):
    # Could be added to the dataset loading script instead
    if examples["mteb_dataset_name"] in TASK_LIST_CLASSIFICATION_NORM:
        examples["mteb_task"] = "Classification"
    elif examples["mteb_dataset_name"] in TASK_LIST_CLUSTERING:
        examples["mteb_task"] = "Clustering"
    elif examples["mteb_dataset_name"] in TASK_LIST_PAIR_CLASSIFICATION:
        examples["mteb_task"] = "PairClassification"
    elif examples["mteb_dataset_name"] in TASK_LIST_RERANKING:
        examples["mteb_task"] = "Reranking"
    elif examples["mteb_dataset_name"] in TASK_LIST_RETRIEVAL_NORM:
        examples["mteb_task"] = "Retrieval"
    elif examples["mteb_dataset_name"] in TASK_LIST_STS_NORM:
        examples["mteb_task"] = "STS"
    elif examples["mteb_dataset_name"] in TASK_LIST_SUMMARIZATION:
        examples["mteb_task"] = "Summarization"
    else:
        examples["mteb_task"] = "BitextMining"
    return examples

for model in EXTERNAL_MODELS:
    ds = load_dataset("mteb/results", model)
    ds = ds.map(add_lang)
    ds = ds.map(add_task)
    base_dict = {"Model": make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, "https://huggingface.co/mteb/leaderboard"))}
    # For now only one metric per task - Could add more metrics lateron
    for task, metric in TASK_TO_METRIC.items():
        ds_dict = ds.filter(lambda x: (x["mteb_task"] == task) and (x["metric"] == metric))["test"].to_dict()
        ds_dict = {k: round(v, 2) for k, v in zip(ds_dict["mteb_dataset_name_with_lang"], ds_dict["score"])}
        EXTERNAL_MODEL_RESULTS[model][task][metric].append({**base_dict, **ds_dict})


def get_mteb_data(tasks=["Clustering"], langs=[], cast_to_str=True, task_to_metric=TASK_TO_METRIC):
    api = HfApi()
    models = api.list_models(filter="mteb")
    # Initialize list to models that we cannot fetch metadata from
    df_list = []
    for model in EXTERNAL_MODEL_RESULTS:
        results_list = [res for task in tasks for res in EXTERNAL_MODEL_RESULTS[model][task][task_to_metric[task]]]
        if langs:
            # Would be cleaner to rely on an extra language column instead
            langs_format = [f"({lang})" for lang in langs]
            res = {k: v for d in results_list for k, v in d.items() if any([k.split(" ")[-1] in (k, x) for x in langs_format])}
        else:
            res = {k: v for d in results_list for k, v in d.items()}
        # Model & at least one result
        if len(res) > 1:
            df_list.append(res)
    
    for model in models:
        readme_path = hf_hub_download(model.modelId, filename="README.md")
        meta = metadata_load(readme_path)
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
        if langs:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks) and (sub_res.get("dataset", {}).get("config", "default") in ("default", *langs))]
        else:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks)]
        out = [{res["dataset"]["name"].replace("MTEB ", ""): [round(score["value"], 2) for score in res["metrics"] if score["type"] == task_to_metric.get(res["task"]["type"])][0]} for res in task_results]
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = make_clickable_model(model.modelId)
        df_list.append(out)
    df = pd.DataFrame(df_list)
    # Put 'Model' column first
    cols = sorted(list(df.columns))
    cols.insert(0, cols.pop(cols.index("Model")))
    df = df[cols]
    df.fillna("", inplace=True)
    if cast_to_str:
        return df.astype(str) # Cast to str as Gradio does not accept floats
    return df

def get_mteb_average():
    global DATA_OVERALL, DATA_CLASSIFICATION_EN, DATA_CLUSTERING, DATA_PAIR_CLASSIFICATION, DATA_RERANKING, DATA_RETRIEVAL, DATA_STS_EN, DATA_SUMMARIZATION, NUM_SCORES
    DATA_OVERALL = get_mteb_data(
        tasks=[
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
            "STS",
            "Summarization",
        ],
        langs=["en", "en-en"],
        cast_to_str=False
    )
    # Approximation (Missing Bitext Mining & including some nans)
    NUM_SCORES = DATA_OVERALL.shape[0] * DATA_OVERALL.shape[1]

    # Debugging:
    # DATA_OVERALL.to_csv("overall.csv")
    
    DATA_OVERALL.insert(1, f"Average ({len(TASK_LIST_EN)} datasets)", DATA_OVERALL[TASK_LIST_EN].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(2, f"Classification Average ({len(TASK_LIST_CLASSIFICATION)} datasets)", DATA_OVERALL[TASK_LIST_CLASSIFICATION].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(3, f"Clustering Average ({len(TASK_LIST_CLUSTERING)} datasets)", DATA_OVERALL[TASK_LIST_CLUSTERING].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(4, f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION)} datasets)", DATA_OVERALL[TASK_LIST_PAIR_CLASSIFICATION].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(5, f"Reranking Average ({len(TASK_LIST_RERANKING)} datasets)", DATA_OVERALL[TASK_LIST_RERANKING].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(6, f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)", DATA_OVERALL[TASK_LIST_RETRIEVAL].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(7, f"STS Average ({len(TASK_LIST_STS)} datasets)", DATA_OVERALL[TASK_LIST_STS].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(8, f"Summarization Average ({len(TASK_LIST_SUMMARIZATION)} dataset)", DATA_OVERALL[TASK_LIST_SUMMARIZATION].mean(axis=1, skipna=False))
    DATA_OVERALL.sort_values(f"Average ({len(TASK_LIST_EN)} datasets)", ascending=False, inplace=True)
    # Start ranking from 1
    DATA_OVERALL.insert(0, "Rank", list(range(1, len(DATA_OVERALL) + 1)))

    DATA_OVERALL = DATA_OVERALL.round(2).astype(str)

    DATA_CLASSIFICATION_EN = DATA_OVERALL[["Model"] + TASK_LIST_CLASSIFICATION]
    DATA_CLUSTERING = DATA_OVERALL[["Model"] + TASK_LIST_CLUSTERING]
    DATA_PAIR_CLASSIFICATION = DATA_OVERALL[["Model"] + TASK_LIST_PAIR_CLASSIFICATION]
    DATA_RERANKING = DATA_OVERALL[["Model"] + TASK_LIST_RERANKING]
    DATA_RETRIEVAL = DATA_OVERALL[["Model"] + TASK_LIST_RETRIEVAL]
    DATA_STS_EN = DATA_OVERALL[["Model"] + TASK_LIST_STS]
    DATA_SUMMARIZATION = DATA_OVERALL[["Model"] + TASK_LIST_SUMMARIZATION]

    DATA_OVERALL = DATA_OVERALL[["Rank", "Model", f"Average ({len(TASK_LIST_EN)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION)} datasets)", f"Reranking Average ({len(TASK_LIST_RERANKING)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)", f"STS Average ({len(TASK_LIST_STS)} datasets)", f"Summarization Average ({len(TASK_LIST_SUMMARIZATION)} dataset)"]]

    return DATA_OVERALL

get_mteb_average()
block = gr.Blocks()


with block:
    gr.Markdown(f"""
    Massive Text Embedding Benchmark (MTEB) Leaderboard. To submit, refer to the <a href="https://github.com/embeddings-benchmark/mteb#leaderboard" target="_blank" style="text-decoration: underline">MTEB GitHub repository</a> 🤗

    - **Total Datasets**: 56
    - **Total Languages**: 117
    - **Total Scores**: >{NUM_SCORES}
    - **Total Models**: {len(DATA_OVERALL)}
    """)
    with gr.Tabs():
        with gr.TabItem("Overall"):
            with gr.Row():
                gr.Markdown("""
                **Overall MTEB English leaderboard 🔮**
                
                - **Metric:** Various, refer to task tabs
                - **Languages:** English, refer to task tabs for others
                """)
            with gr.Row():
                data_overall = gr.components.Dataframe(
                    DATA_OVERALL,
                    datatype=["markdown"] * len(DATA_OVERALL.columns) * 2,
                    type="pandas",
                    wrap=True,
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                data_run.click(get_mteb_average, inputs=None, outputs=data_overall)                
        with gr.TabItem("BitextMining"):
            with gr.Row():
                    gr.Markdown("""
                    **Bitext Mining Leaderboard 🎌**
                    
                    - **Metric:** F1 (f1)
                    - **Languages:** 117
                    """)
            with gr.Row():
                data_bitext_mining = gr.components.Dataframe(
                    datatype=["markdown"] * 500, # hack when we don't know how many columns
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_bitext_mining = gr.Variable(value=["BitextMining"])
                data_run.click(
                    get_mteb_data,
                    inputs=[task_bitext_mining],
                    outputs=data_bitext_mining,
                )
        with gr.TabItem("Classification"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Leaderboard ❤️**
                    
                    - **Metric:** Accuracy (accuracy)
                    - **Languages:** English
                    """)
                with gr.Row():
                    data_classification_en = gr.components.Dataframe(
                        DATA_CLASSIFICATION_EN,
                        datatype=["markdown"] * len(DATA_CLASSIFICATION_EN.columns) * 20,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_classification_en = gr.Button("Refresh")
                    task_classification_en = gr.Variable(value=["Classification"])
                    lang_classification_en = gr.Variable(value=["en"])
                    data_run_classification_en.click(
                        get_mteb_data,
                        inputs=[
                            task_classification_en,
                            lang_classification_en,
                        ],
                        outputs=data_classification_en,
                    )
            with gr.TabItem("Multilingual"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Multilingual Leaderboard 💜💚💙**
                    
                    - **Metric:** Accuracy (accuracy)
                    - **Languages:** 51
                    """)
                with gr.Row():
                    data_classification = gr.components.Dataframe(
                        datatype=["markdown"] * 200, # hack when we don't know how many columns
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_classification = gr.Variable(value=["Classification"])
                    data_run.click(
                        get_mteb_data,
                        inputs=[task_classification],
                        outputs=data_classification,
                    )
        with gr.TabItem("Clustering"):
            with gr.Row():
                gr.Markdown("""
                **Clustering Leaderboard ✨**
                
                - **Metric:** Validity Measure (v_measure)
                - **Languages:** English
                """)
            with gr.Row():
                data_clustering = gr.components.Dataframe(
                    DATA_CLUSTERING,
                    datatype=["markdown"] * len(DATA_CLUSTERING.columns) * 2,
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_clustering = gr.Variable(value=["Clustering"])
                data_run.click(
                    get_mteb_data,
                    inputs=[task_clustering],
                    outputs=data_clustering,
                )
        with gr.TabItem("Pair Classification"):
            with gr.Row():
                gr.Markdown("""
                **Pair Classification Leaderboard 🎭**
                
                - **Metric:** Average Precision based on Cosine Similarities (cos_sim_ap)
                - **Languages:** English
                """)
            with gr.Row():
                data_pair_classification = gr.components.Dataframe(
                    DATA_PAIR_CLASSIFICATION,
                    datatype=["markdown"] * len(DATA_PAIR_CLASSIFICATION.columns) * 2,
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_pair_classification = gr.Variable(value=["PairClassification"])
                data_run.click(
                    get_mteb_data,
                    inputs=[task_pair_classification],
                    outputs=data_pair_classification,
                )
        with gr.TabItem("Retrieval"):
            with gr.Row():
                gr.Markdown("""
                **Retrieval Leaderboard  🔎**
                
                - **Metric:** Normalized Discounted Cumulative Gain @ k (ndcg_at_10)
                - **Languages:** English
                """)
            with gr.Row():
                data_retrieval = gr.components.Dataframe(
                    DATA_RETRIEVAL,
                    datatype=["markdown"] * len(DATA_RETRIEVAL.columns) * 2,
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_retrieval = gr.Variable(value=["Retrieval"])
                data_run.click(
                    get_mteb_data, inputs=[task_retrieval], outputs=data_retrieval
                )
        with gr.TabItem("Reranking"):
            with gr.Row():
                gr.Markdown("""
                **Reranking Leaderboard 🥈**
                
                - **Metric:** Mean Average Precision (MAP)
                - **Languages:** English
                """)
            with gr.Row():
                data_reranking = gr.components.Dataframe(
                    DATA_RERANKING,
                    datatype=["markdown"] * len(DATA_RERANKING.columns) * 2,
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_reranking = gr.Variable(value=["Reranking"])
                metric_reranking = gr.Variable(value="map")
                data_run.click(
                    get_mteb_data, inputs=[task_reranking], outputs=data_reranking
                )
        with gr.TabItem("STS"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **STS Leaderboard 🤖**
                    
                    - **Metric:** Spearman correlation based on cosine similarity
                    - **Languages:** English
                    """)
                with gr.Row():
                    data_sts_en = gr.components.Dataframe(
                        DATA_STS_EN,
                        datatype=["markdown"] * len(DATA_STS_EN.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_sts_en = gr.Button("Refresh")
                    task_sts_en = gr.Variable(value=["STS"])
                    lang_sts_en = gr.Variable(value=["en", "en-en"])
                    data_run_sts_en.click(
                        get_mteb_data,
                        inputs=[task_sts_en, lang_sts_en],
                        outputs=data_sts_en,
                    )
            with gr.TabItem("Multilingual"):
                with gr.Row():
                    gr.Markdown("""
                    **STS Multilingual Leaderboard 👽**
                    
                    - **Metric:** Spearman correlation based on cosine similarity
                    - **Languages:** Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Russian, Spanish
                    """)
                with gr.Row():
                    data_sts = gr.components.Dataframe(
                        datatype=["markdown"] * 50, # hack when we don't know how many columns
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_sts = gr.Variable(value=["STS"])
                    data_run.click(get_mteb_data, inputs=[task_sts], outputs=data_sts)
        with gr.TabItem("Summarization"):
            with gr.Row():
                gr.Markdown("""
                **Summarization Leaderboard 📜**
                
                - **Metric:** Spearman correlation based on cosine similarity
                - **Languages:** English
                """)
            with gr.Row():
                data_summarization = gr.components.Dataframe(
                    DATA_SUMMARIZATION * len(DATA_SUMMARIZATION.columns) * 2,
                    datatype="markdown",
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_summarization = gr.Variable(value=["Summarization"])
                data_run.click(
                    get_mteb_data,
                    inputs=[task_summarization],
                    outputs=data_summarization,
                )
    gr.Markdown(f"""
    <p style="text-align: center;">Made with ❤️ for NLP by <a href=https://huggingface.co/Muennighoff>Niklas Muennighoff</a>.</p>
    """)
    # Running the function on page load in addition to when the button is clicked
    # This is optional - If deactivated the data created loaded at "Build time" is shown like for Overall tab
    block.load(get_mteb_data, inputs=[task_bitext_mining], outputs=data_bitext_mining)
    block.load(get_mteb_data, inputs=[task_classification_en, lang_classification_en], outputs=data_classification_en)
    block.load(get_mteb_data, inputs=[task_classification], outputs=data_classification)
    block.load(get_mteb_data, inputs=[task_clustering], outputs=data_clustering)
    block.load(get_mteb_data, inputs=[task_retrieval], outputs=data_retrieval)
    block.load(get_mteb_data, inputs=[task_reranking], outputs=data_reranking)
    block.load(get_mteb_data, inputs=[task_sts_en, lang_sts_en], outputs=data_sts_en)
    block.load(get_mteb_data, inputs=[task_sts], outputs=data_sts)
    block.load(get_mteb_data, inputs=[task_summarization], outputs=data_summarization)

block.launch()


# Possible changes:
# Could check if tasks are valid (Currently users could just invent new tasks - similar for languages)
# Could make it load in the background without the Gradio logo closer to the Deep RL space
# Could add graphs / other visual content

# Sources:
# https://huggingface.co/spaces/gradio/leaderboard
# https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard
# https://getemoji.com/
