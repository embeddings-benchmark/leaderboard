---
title: MTEB Leaderboard 
emoji: 🥇
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
tags:
  - leaderboard
startup_duration_timeout: 1h
fullWidth: true
---

## The MTEB Leaderboard repository

This repository contains the code for pushing and updating the MTEB leaderboard daily. 

| Relevant Links                                                | Decription                                                                                                                                                                                                |
|------------------------------------------|------------------------------|
| [mteb](https://github.com/embeddings-benchmark/mteb)          | The implementation of the benchmark. Here you e.g. find the code to run your model on the benchmark.                                                                                                      |
| [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | The leaderboard itself, here you can view results of model run on MTEB.                                                                                                                                   |
| [results](https://github.com/embeddings-benchmark/results)    | The results of MTEB is stored here. Though you can publish them to the leaderboard [adding](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md) the result to your model card. |

## Developer setup

To setup the repository:

```         
git clone {repo_url}
# potentially create virtual environment using python 3.9
pip install -r requirements.txt
```