import os
from yaml import safe_load

from huggingface_hub import HfApi

LEADERBOARD_CONFIG_PATH = "config.yaml"
with open(LEADERBOARD_CONFIG_PATH, 'r', encoding='utf-8') as f:
    LEADERBOARD_CONFIG = safe_load(f)
MODEL_META_PATH = "model_meta.yaml"
with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
    MODEL_META = safe_load(f)

# Try first to get the config from the environment variables, then from the config.yaml file
def get_config(name, default):
    res = None

    if name in os.environ:
        res = os.environ[name]
    elif 'config' in LEADERBOARD_CONFIG:
        res = LEADERBOARD_CONFIG['config'].get(name, None)

    if res is None:
        return default
    return res


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


# clone / pull the lmeh eval data
HF_TOKEN = get_config("HF_TOKEN", None)

LEADERBOARD_NAME = get_config("LEADERBOARD_NAME", "MTEB Leaderboard")

REPO_ID = get_config("REPO_ID", "mteb/leaderboard")
RESULTS_REPO = get_config("RESULTS_REPO", "mteb/results")

CACHE_PATH = get_config("HF_HOME", ".")
os.environ["HF_HOME"] = CACHE_PATH

# Check if it is using persistent storage
if not os.access(CACHE_PATH, os.W_OK):
    print(f"No write access to HF_HOME: {CACHE_PATH}. Resetting to current directory.")
    CACHE_PATH = "."
    os.environ["HF_HOME"] = CACHE_PATH
else:
    print(f"Write access confirmed for HF_HOME")

API = HfApi(token=HF_TOKEN)
