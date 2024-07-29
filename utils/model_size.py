import json
import re
import traceback
from huggingface_hub.hf_api import ModelInfo, get_safetensors_metadata, model_info as get_model_info, get_hf_file_metadata, hf_hub_url
from huggingface_hub import hf_hub_download

# Map model IDs to the number of bytes used for one parameter. So, 4 bytes for fp32, 2 bytes for fp16, etc.
# By default, we assume that the model is stored in fp32.
KNOWN_BYTES_PER_PARAM = {
    "dwzhu/e5-base-4k": 2,
}

def get_model_parameters_memory(model_info: ModelInfo):
    '''Get the size of the model in million of parameters.'''
    try:
        safetensors = get_safetensors_metadata(model_info.id)
    except Exception as e:
        pass
    else:
        num_parameters = sum(safetensors.parameter_count.values())
        return round(num_parameters / 1e6), round(num_parameters * 4 / 1024**3, 2)

    filenames = [sib.rfilename for sib in model_info.siblings]
    if "pytorch_model.bin" in filenames:
        url = hf_hub_url(model_info.id, filename="pytorch_model.bin")
        meta = get_hf_file_metadata(url)
        bytes_per_param = KNOWN_BYTES_PER_PARAM.get(model_info.id, 4)
        num_params = round(meta.size / bytes_per_param / 1e6)
        size_gb = round(meta.size * (4 / bytes_per_param) / 1024**3, 2)
        return num_params, size_gb
    
    if "pytorch_model.bin.index.json" in filenames:
        index_path = hf_hub_download(model_info.id, filename="pytorch_model.bin.index.json")
        """
        {
        "metadata": {
            "total_size": 28272820224
        },....
        """
        size = json.load(open(index_path))
        bytes_per_param = KNOWN_BYTES_PER_PARAM.get(model_info.id, 4)
        if ("metadata" in size) and ("total_size" in size["metadata"]):
            return round(size["metadata"]["total_size"] / bytes_per_param / 1e6), round(size["metadata"]["total_size"] / 1024**3, 2)

    raise Exception(f"Could not find the model parameters for {model_info.id}")
