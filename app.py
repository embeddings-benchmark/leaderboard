import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import metadata_load

path = f"https://huggingface.co/api/spaces"

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


TASK_LIST_SUMMARIZATION = [
    "SummEval",
]

TASK_LIST_EN = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS + TASK_LIST_SUMMARIZATION

TASK_TO_TASK_LIST = {}



def make_clickable_model(model_name):
    # Remove user from model name
    model_name_show = " ".join(model_name.split("/")[1:])
    link = "https://huggingface.co/" + model_name
    return (
        f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name_show}</a>'
    )


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

def get_mteb_data(tasks=["Clustering"], metric="v_measure", langs=[], cast_to_str=True, task_to_metric=TASK_TO_METRIC):
    api = HfApi()
    models = api.list_models(filter="mteb")
    df_list = []
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
        #if langs is None:
        task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks) and (sub_res.get("dataset", {}).get("config", "default") in ("default", *langs))]
        out = [{res["dataset"]["name"].replace("MTEB ", ""): [round(score["value"], 2) for score in res["metrics"] if score["type"] == task_to_metric.get(res["task"]["type"])][0]} for res in task_results]
        #else:
            # Multilingual
        #    out = list(
        #        map(
        #            lambda x: {
        #                x["dataset"]["name"].replace("MTEB ", ""): round(
        #                    list(filter(lambda x: x["type"] == metric, x["metrics"]))[0]["value"], 2
        #                )
        #            },
        #            filter(
        #                lambda x: (x.get("task", {}).get("type", "") in tasks)
        #                and (x.get("dataset", {}).get("config", "") in ("default", *langs)),
        #                meta["model-index"][0]["results"],
        #            ),
        #        )
        #    )
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = make_clickable_model(model.modelId)
        df_list.append(out)
    df = pd.DataFrame(df_list)
    # Put 'Model' column first
    cols = sorted(list(df.columns))
    cols.insert(0, cols.pop(cols.index("Model")))
    df = df[cols]
    # df.insert(1, "Average", df.mean(axis=1, skipna=False))
    df.fillna("", inplace=True)
    if cast_to_str:
        return df.astype(str) # Cast to str as Gradio does not accept floats
    return df


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

DATA_OVERALL.insert(1, "Average", DATA_OVERALL[TASK_LIST_EN].mean(axis=1, skipna=False))
DATA_OVERALL.insert(2, "Classification Average", DATA_OVERALL[TASK_LIST_CLASSIFICATION].mean(axis=1, skipna=False))
DATA_OVERALL.insert(3, "Clustering Average", DATA_OVERALL[TASK_LIST_CLUSTERING].mean(axis=1, skipna=False))
DATA_OVERALL.insert(4, "Pair Classification Average", DATA_OVERALL[TASK_LIST_PAIR_CLASSIFICATION].mean(axis=1, skipna=False))
DATA_OVERALL.insert(5, "Reranking Average", DATA_OVERALL[TASK_LIST_RERANKING].mean(axis=1, skipna=False))
DATA_OVERALL.insert(6, "Retrieval Average", DATA_OVERALL[TASK_LIST_RETRIEVAL].mean(axis=1, skipna=False))
DATA_OVERALL.insert(7, "STS Average", DATA_OVERALL[TASK_LIST_STS].mean(axis=1, skipna=False))
DATA_OVERALL.insert(8, "Summarization Average", DATA_OVERALL[TASK_LIST_SUMMARIZATION].mean(axis=1, skipna=False))
DATA_OVERALL = DATA_OVERALL.round(2).astype(str)

DATA_CLASSIFICATION_EN = DATA_OVERALL[["Model"] + TASK_LIST_CLASSIFICATION]
DATA_CLUSTERING = DATA_OVERALL[["Model"] + TASK_LIST_CLUSTERING]
DATA_PAIR_CLASSIFICATION = DATA_OVERALL[["Model"] + TASK_LIST_PAIR_CLASSIFICATION]
DATA_RERANKING = DATA_OVERALL[["Model"] + TASK_LIST_RERANKING]
DATA_RETRIEVAL = DATA_OVERALL[["Model"] + TASK_LIST_RETRIEVAL]
DATA_STS_EN = DATA_OVERALL[["Model"] + TASK_LIST_STS]
DATA_SUMMARIZATION = DATA_OVERALL[["Model"] + TASK_LIST_SUMMARIZATION]

DATA_OVERALL = DATA_OVERALL[["Model", "Average", "Classification Average", "Clustering Average", "Pair Classification Average", "Reranking Average", "Retrieval Average", "STS Average", "Summarization Average"]]


block = gr.Blocks()

with block:
    gr.Markdown(
        """Leaderboard for XX most popular Blocks Event Spaces. To learn more and join, see <a href="https://huggingface.co/Gradio-Blocks" target="_blank" style="text-decoration: underline">Blocks Party Event</a>"""
    )
    with gr.Tabs():
        with gr.TabItem("Overall"):
            with gr.Row():
                gr.Markdown("""Average Scores""")
            with gr.Row():
                data_overall = gr.components.Dataframe(
                    DATA_OVERALL,
                    datatype="markdown",
                    type="pandas",
                    col_count=(len(DATA_OVERALL.columns), "fixed"),
                    wrap=True,
                )
        with gr.TabItem("Classification"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""Leaderboard for Classification""")
                with gr.Row():
                    data_classification_en = gr.components.Dataframe(
                        DATA_CLASSIFICATION_EN,
                        datatype="markdown",
                        type="pandas",
                        col_count=(len(DATA_CLASSIFICATION_EN.columns), "fixed"),
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_classification_en = gr.Variable(value="Classification")
                    metric_classification_en = gr.Variable(value="accuracy")
                    lang_classification_en = gr.Variable(value=["en"])
                    data_run.click(
                        get_mteb_data,
                        inputs=[
                            task_classification_en,
                            metric_classification_en,
                            lang_classification_en,
                        ],
                        outputs=data_classification_en,
                    )
            with gr.TabItem("Multilingual"):
                with gr.Row():
                    gr.Markdown("""Multilingual Classification""")
                with gr.Row():
                    data_classification = gr.components.Dataframe(
                        datatype=["markdown"] * 500,
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_classification = gr.Variable(value="Classification")
                    metric_classification = gr.Variable(value="accuracy")
                    data_run.click(
                        get_mteb_data,
                        inputs=[task_classification, metric_classification],
                        outputs=data_classification,
                    )
        with gr.TabItem("Clustering"):
            with gr.Row():
                gr.Markdown("""Leaderboard for Clustering""")
            with gr.Row():
                data_clustering = gr.components.Dataframe(
                    datatype=["markdown"] * 500,
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_clustering = gr.Variable(value="Clustering")
                metric_clustering = gr.Variable(value="v_measure")
                data_run.click(
                    get_mteb_data,
                    inputs=[task_clustering, metric_clustering],
                    outputs=data_clustering,
                )
        with gr.TabItem("Retrieval"):
            with gr.Row():
                gr.Markdown("""Leaderboard for Retrieval""")
            with gr.Row():
                data_retrieval = gr.components.Dataframe(
                    datatype=["markdown"] * 500,
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_retrieval = gr.Variable(value="Retrieval")
                metric_retrieval = gr.Variable(value="ndcg_at_10")
                data_run.click(
                    get_mteb_data, inputs=[task_retrieval, metric_retrieval], outputs=data_retrieval
                )
        with gr.TabItem("Reranking"):
            with gr.Row():
                gr.Markdown("""Leaderboard for Reranking""")
            with gr.Row():
                data_reranking = gr.components.Dataframe(
                    datatype=["markdown"] * 500,
                    type="pandas",
                    # col_count=(12, "fixed"),
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_reranking = gr.Variable(value="Reranking")
                metric_reranking = gr.Variable(value="map")
                data_run.click(
                    get_mteb_data, inputs=[task_reranking, metric_reranking], outputs=data_reranking
                )
        with gr.TabItem("STS"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""Leaderboard for STS""")
                with gr.Row():
                    data_sts_en = gr.components.Dataframe(
                        datatype=["markdown"] * 500,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_en = gr.Button("Refresh")
                    task_sts_en = gr.Variable(value="STS")
                    metric_sts_en = gr.Variable(value="cos_sim_spearman")
                    lang_sts_en = gr.Variable(value=["en", "en-en"])
                    data_run.click(
                        get_mteb_data,
                        inputs=[task_sts_en, metric_sts_en, lang_sts_en],
                        outputs=data_sts_en,
                    )
            with gr.TabItem("Multilingual"):
                with gr.Row():
                    gr.Markdown("""Leaderboard for STS""")
                with gr.Row():
                    data_sts = gr.components.Dataframe(
                        datatype=["markdown"] * 500,
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_sts = gr.Variable(value="STS")
                    metric_sts = gr.Variable(value="cos_sim_spearman")
                    data_run.click(get_mteb_data, inputs=[task_sts, metric_sts], outputs=data_sts)
        with gr.TabItem("Summarization"):
            with gr.Row():
                gr.Markdown("""Leaderboard for Summarization""")
            with gr.Row():
                data_summarization = gr.components.Dataframe(
                    datatype=["markdown"] * 500,
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_summarization = gr.Variable(value="Summarization")
                metric_summarization = gr.Variable(value="cos_sim_spearman")
                data_run.click(
                    get_mteb_data,
                    inputs=[task_summarization, metric_summarization],
                    outputs=data_summarization,
                )
    # running the function on page load in addition to when the button is clicked
    #block.load(
    #    get_mteb_data,
    #    inputs=[task_classification_en, metric_classification_en],
    #    outputs=data_classification_en,
    #    show_progress=False,
    #)
    block.load(
        get_mteb_data,
        inputs=[task_classification, metric_classification],
        outputs=data_classification,
    )
    block.load(get_mteb_data, inputs=[task_clustering, metric_clustering], outputs=data_clustering)
    block.load(get_mteb_data, inputs=[task_retrieval, metric_retrieval], outputs=data_retrieval)
    block.load(get_mteb_data, inputs=[task_reranking, metric_reranking], outputs=data_reranking)
    block.load(get_mteb_data, inputs=[task_sts, metric_sts], outputs=data_sts)
    block.load(
        get_mteb_data, inputs=[task_summarization, metric_summarization], outputs=data_summarization
    )

block.launch()
