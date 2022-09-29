import gradio as gr
import requests
import pandas as pd
from huggingface_hub.hf_api import SpaceInfo
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import metadata_load

path = f"https://huggingface.co/api/spaces"


#api = HfApi()
#models = api.list_models(filter="mteb")
#readme_path = hf_hub_download(models[0].modelId, filename="README.md")
#meta = metadata_load(readme_path)
#list(filter(lambda x: x["task"]["type"] == "Retrieval", meta["model-index"][0]["results"]))


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

def get_clustering(task="Clustering", metric="v_measure"):
    api = HfApi()
    models = api.list_models(filter="mteb")
    df_list = []
    for model in models:
        readme_path = hf_hub_download(model.modelId, filename="README.md")
        meta = metadata_load(readme_path)
        out = list(
            map(
                lambda x: {x["dataset"]["name"]: list(filter(lambda x: x["type"] == metric, x["metrics"]))[0]["value"]}, 
                filter(lambda x: x["task"]["type"] == task, meta["model-index"][0]["results"])
            )
        )
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = model.modelId
        df_list.append(out)
    df = pd.DataFrame(df_list)
    # Put Model in the beginning & sort the others
    df = df[[df.columns[-1]] + sorted(df.columns[:-1])]
    return df

block = gr.Blocks()

with block:    
    gr.Markdown("""Leaderboard for XX most popular Blocks Event Spaces. To learn more and join, see <a href="https://huggingface.co/Gradio-Blocks" target="_blank" style="text-decoration: underline">Blocks Party Event</a>""")
    with gr.Tabs():
        with gr.TabItem("Blocks Party Leaderboard"):
            with gr.Row():
                data = gr.components.Dataframe(type="pandas")
            with gr.Row():
                data_run = gr.Button("Refresh")
                data_run.click(get_blocks_party_spaces, inputs=None, outputs=data)
        with gr.TabItem("Clustering"):
            with gr.Row():
                gr.Markdown("""Leaderboard for Clustering""")
            with gr.Row():
                data = gr.components.Dataframe(type="pandas")
            with gr.Row():
                data_run = gr.Button("Refresh")
                data_run.click(get_clustering, inputs=None, outputs=data)
        with gr.TabItem("Blocks Party Leaderboard2"):
            with gr.Row():
                data = gr.components.Dataframe(type="pandas")
            with gr.Row():
                data_run = gr.Button("Refresh")
                data_run.click(get_blocks_party_spaces, inputs=None, outputs=data)
    # running the function on page load in addition to when the button is clicked
    block.load(get_blocks_party_spaces, inputs=None, outputs=data)               

block.launch()

