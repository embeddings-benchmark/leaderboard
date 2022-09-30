import gradio as gr
import requests
import pandas as pd
from huggingface_hub.hf_api import SpaceInfo
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import metadata_load

path = f"https://huggingface.co/api/spaces"

def get_blocks_party_spaces():
    r = requests.get(path)
    d = r.json()
    spaces = [SpaceInfo(**x) for x in d]
    blocks_spaces = {}
    for i in range(0,len(spaces)):
        if spaces[i].id.split('/')[0] == 'Gradio-Blocks' and hasattr(spaces[i], 'likes') and spaces[i].id != 'Gradio-Blocks/Leaderboard' and spaces[i].id != 'Gradio-Blocks/README':
            blocks_spaces[spaces[i].id]=spaces[i].likes
    df = pd.DataFrame(
    [{"Spaces_Name": Spaces, "likes": likes} for Spaces,likes in blocks_spaces.items()])
    df = df.sort_values(by=['likes'],ascending=False)
    return df

def make_clickable_model(model_name):
    # remove user from model name
    model_name_show = ' '.join(model_name.split('/')[1:])
    link = "https://huggingface.co/" + model_name
    return f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name_show}</a>'


def get_mteb_data(task="Clustering", metric="v_measure", lang=None):
    api = HfApi()
    models = api.list_models(filter="mteb")
    df_list = []
    for model in models:
        readme_path = hf_hub_download(model.modelId, filename="README.md")
        meta = metadata_load(readme_path)
        # Use "get" instead of dict indexing to ignore incompat metadata instead of erroring out
        if lang is None:
            out = list(
                map(
                    lambda x: {x["dataset"]["name"].replace("MTEB ", ""): round(list(filter(lambda x: x["type"] == metric, x["metrics"]))[0]["value"], 2)}, 
                    filter(lambda x: x.get("task", {}).get("type", "") == task, meta["model-index"][0]["results"])
                )
            )
        else:
            # Multilingual
            out = list(
                map(
                    lambda x: {x["dataset"]["name"].replace("MTEB ", ""): round(list(filter(lambda x: x["type"] == metric, x["metrics"]))[0]["value"], 2)}, 
                    filter(lambda x: (x.get("task", {}).get("type", "") == task) and (x.get("dataset", {}).get("config", "") in ("default", *lang)), meta["model-index"][0]["results"])
                )
            )
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = make_clickable_model(model.modelId)
        df_list.append(out)
    df = pd.DataFrame(df_list)
    # Put 'Model' column first
    cols = sorted(list(df.columns))
    cols.insert(0, cols.pop(cols.index("Model")))
    df = df[cols]

    df.fillna('', inplace=True)
    return df.astype(str) # Cast to str as Gradio does not accept floats

block = gr.Blocks()

with block:    
    gr.Markdown("""Leaderboard for XX most popular Blocks Event Spaces. To learn more and join, see <a href="https://huggingface.co/Gradio-Blocks" target="_blank" style="text-decoration: underline">Blocks Party Event</a>""")
    with gr.Tabs():
        with gr.TabItem("Classification"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""Leaderboard for Classification""")
                with gr.Row():
                    data_classification_en = gr.components.Dataframe(
                        datatype=["markdown"] * 500,
                        type="pandas",
                        col_count=(13, "fixed"),
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_classification_en = gr.Variable(value="Classification")
                    metric_classification_en = gr.Variable(value="accuracy")
                    lang_classification_en = gr.Variable(value=["en"])
                    data_run.click(get_mteb_data, inputs=[task_classification_en, metric_classification_en, lang_classification_en], outputs=data_classification_en)
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
                    data_run.click(get_mteb_data, inputs=[task_classification, metric_classification], outputs=data_classification)
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
                data_run.click(get_mteb_data, inputs=[task_clustering, metric_clustering], outputs=data_clustering)
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
                data_run.click(get_mteb_data, inputs=[task_retrieval, metric_retrieval], outputs=data_retrieval)
        with gr.TabItem("Reranking"):
            with gr.Row():
                gr.Markdown("""Leaderboard for Reranking""")
            with gr.Row():
                data_reranking = gr.components.Dataframe(
                    datatype=["markdown"] * 500,
                    type="pandas",
                    #col_count=(12, "fixed"),
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_reranking = gr.Variable(value="Reranking")
                metric_reranking = gr.Variable(value="map")
                data_run.click(get_mteb_data, inputs=[task_reranking, metric_reranking], outputs=data_reranking)                
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
                    data_run.click(get_mteb_data, inputs=[task_sts_en, metric_sts_en, lang_sts_en], outputs=data_sts_en)
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
                data_run.click(get_mteb_data, inputs=[task_summarization, metric_summarization], outputs=data_summarization)                   
        with gr.TabItem("Blocks Party Leaderboard2"):
            with gr.Row():
                data = gr.components.Dataframe(type="pandas")
            with gr.Row():
                data_run = gr.Button("Refresh")
                data_run.click(get_blocks_party_spaces, inputs=None, outputs=data)
    # running the function on page load in addition to when the button is clicked
    block.load(get_mteb_data, inputs=[task_classification_en, metric_classification_en], outputs=data_classification_en)
    block.load(get_mteb_data, inputs=[task_classification, metric_classification], outputs=data_classification)
    block.load(get_mteb_data, inputs=[task_clustering, metric_clustering], outputs=data_clustering)
    block.load(get_mteb_data, inputs=[task_retrieval, metric_retrieval], outputs=data_retrieval)
    block.load(get_mteb_data, inputs=[task_reranking, metric_reranking], outputs=data_reranking)
    block.load(get_mteb_data, inputs=[task_sts, metric_sts], outputs=data_sts)    
    block.load(get_mteb_data, inputs=[task_summarization, metric_summarization], outputs=data_summarization)        

    block.load(get_blocks_party_spaces, inputs=None, outputs=data)  

block.launch()

