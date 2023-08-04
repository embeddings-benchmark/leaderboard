import json

from datasets import load_dataset
import gradio as gr
from huggingface_hub import get_hf_file_metadata, HfApi, hf_hub_download, hf_hub_url
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

TASK_LIST_BITEXT_MINING = ['BUCC (de-en)', 'BUCC (fr-en)', 'BUCC (ru-en)', 'BUCC (zh-en)', 'Tatoeba (afr-eng)', 'Tatoeba (amh-eng)', 'Tatoeba (ang-eng)', 'Tatoeba (ara-eng)', 'Tatoeba (arq-eng)', 'Tatoeba (arz-eng)', 'Tatoeba (ast-eng)', 'Tatoeba (awa-eng)', 'Tatoeba (aze-eng)', 'Tatoeba (bel-eng)', 'Tatoeba (ben-eng)', 'Tatoeba (ber-eng)', 'Tatoeba (bos-eng)', 'Tatoeba (bre-eng)', 'Tatoeba (bul-eng)', 'Tatoeba (cat-eng)', 'Tatoeba (cbk-eng)', 'Tatoeba (ceb-eng)', 'Tatoeba (ces-eng)', 'Tatoeba (cha-eng)', 'Tatoeba (cmn-eng)', 'Tatoeba (cor-eng)', 'Tatoeba (csb-eng)', 'Tatoeba (cym-eng)', 'Tatoeba (dan-eng)', 'Tatoeba (deu-eng)', 'Tatoeba (dsb-eng)', 'Tatoeba (dtp-eng)', 'Tatoeba (ell-eng)', 'Tatoeba (epo-eng)', 'Tatoeba (est-eng)', 'Tatoeba (eus-eng)', 'Tatoeba (fao-eng)', 'Tatoeba (fin-eng)', 'Tatoeba (fra-eng)', 'Tatoeba (fry-eng)', 'Tatoeba (gla-eng)', 'Tatoeba (gle-eng)', 'Tatoeba (glg-eng)', 'Tatoeba (gsw-eng)', 'Tatoeba (heb-eng)', 'Tatoeba (hin-eng)', 'Tatoeba (hrv-eng)', 'Tatoeba (hsb-eng)', 'Tatoeba (hun-eng)', 'Tatoeba (hye-eng)', 'Tatoeba (ido-eng)', 'Tatoeba (ile-eng)', 'Tatoeba (ina-eng)', 'Tatoeba (ind-eng)', 'Tatoeba (isl-eng)', 'Tatoeba (ita-eng)', 'Tatoeba (jav-eng)', 'Tatoeba (jpn-eng)', 'Tatoeba (kab-eng)', 'Tatoeba (kat-eng)', 'Tatoeba (kaz-eng)', 'Tatoeba (khm-eng)', 'Tatoeba (kor-eng)', 'Tatoeba (kur-eng)', 'Tatoeba (kzj-eng)', 'Tatoeba (lat-eng)', 'Tatoeba (lfn-eng)', 'Tatoeba (lit-eng)', 'Tatoeba (lvs-eng)', 'Tatoeba (mal-eng)', 'Tatoeba (mar-eng)', 'Tatoeba (max-eng)', 'Tatoeba (mhr-eng)', 'Tatoeba (mkd-eng)', 'Tatoeba (mon-eng)', 'Tatoeba (nds-eng)', 'Tatoeba (nld-eng)', 'Tatoeba (nno-eng)', 'Tatoeba (nob-eng)', 'Tatoeba (nov-eng)', 'Tatoeba (oci-eng)', 'Tatoeba (orv-eng)', 'Tatoeba (pam-eng)', 'Tatoeba (pes-eng)', 'Tatoeba (pms-eng)', 'Tatoeba (pol-eng)', 'Tatoeba (por-eng)', 'Tatoeba (ron-eng)', 'Tatoeba (rus-eng)', 'Tatoeba (slk-eng)', 'Tatoeba (slv-eng)', 'Tatoeba (spa-eng)', 'Tatoeba (sqi-eng)', 'Tatoeba (srp-eng)', 'Tatoeba (swe-eng)', 'Tatoeba (swg-eng)', 'Tatoeba (swh-eng)', 'Tatoeba (tam-eng)', 'Tatoeba (tat-eng)', 'Tatoeba (tel-eng)', 'Tatoeba (tgl-eng)', 'Tatoeba (tha-eng)', 'Tatoeba (tuk-eng)', 'Tatoeba (tur-eng)', 'Tatoeba (tzl-eng)', 'Tatoeba (uig-eng)', 'Tatoeba (ukr-eng)', 'Tatoeba (urd-eng)', 'Tatoeba (uzb-eng)', 'Tatoeba (vie-eng)', 'Tatoeba (war-eng)', 'Tatoeba (wuu-eng)', 'Tatoeba (xho-eng)', 'Tatoeba (yid-eng)', 'Tatoeba (yue-eng)', 'Tatoeba (zsm-eng)']
TASK_LIST_BITEXT_MINING_OTHER = ["BornholmBitextMining"]

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

TASK_LIST_CLASSIFICATION_DA = [
    "AngryTweetsClassification",
    "DanishPoliticalCommentsClassification",
    "DKHateClassification",
    "LccSentimentClassification",
    "MassiveIntentClassification (da)",
    "MassiveScenarioClassification (da)",
    "NordicLangClassification",
    "ScalaDaClassification",
]

TASK_LIST_CLASSIFICATION_NB = [
    "NoRecClassification",
    "NordicLangClassification",
    "NorwegianParliament",
    "MassiveIntentClassification (nb)",
    "MassiveScenarioClassification (nb)",
    "ScalaNbClassification",
]

TASK_LIST_CLASSIFICATION_SV = [
    "DalajClassification",
    "MassiveIntentClassification (sv)",
    "MassiveScenarioClassification (sv)",
    "NordicLangClassification",
    "ScalaSvClassification",
    "SweRecClassification",
]

TASK_LIST_CLASSIFICATION_OTHER = ['AmazonCounterfactualClassification (de)', 'AmazonCounterfactualClassification (ja)', 'AmazonReviewsClassification (de)', 'AmazonReviewsClassification (es)', 'AmazonReviewsClassification (fr)', 'AmazonReviewsClassification (ja)', 'AmazonReviewsClassification (zh)', 'MTOPDomainClassification (de)', 'MTOPDomainClassification (es)', 'MTOPDomainClassification (fr)', 'MTOPDomainClassification (hi)', 'MTOPDomainClassification (th)', 'MTOPIntentClassification (de)', 'MTOPIntentClassification (es)', 'MTOPIntentClassification (fr)', 'MTOPIntentClassification (hi)', 'MTOPIntentClassification (th)', 'MassiveIntentClassification (af)', 'MassiveIntentClassification (am)', 'MassiveIntentClassification (ar)', 'MassiveIntentClassification (az)', 'MassiveIntentClassification (bn)', 'MassiveIntentClassification (cy)', 'MassiveIntentClassification (de)', 'MassiveIntentClassification (el)', 'MassiveIntentClassification (es)', 'MassiveIntentClassification (fa)', 'MassiveIntentClassification (fi)', 'MassiveIntentClassification (fr)', 'MassiveIntentClassification (he)', 'MassiveIntentClassification (hi)', 'MassiveIntentClassification (hu)', 'MassiveIntentClassification (hy)', 'MassiveIntentClassification (id)', 'MassiveIntentClassification (is)', 'MassiveIntentClassification (it)', 'MassiveIntentClassification (ja)', 'MassiveIntentClassification (jv)', 'MassiveIntentClassification (ka)', 'MassiveIntentClassification (km)', 'MassiveIntentClassification (kn)', 'MassiveIntentClassification (ko)', 'MassiveIntentClassification (lv)', 'MassiveIntentClassification (ml)', 'MassiveIntentClassification (mn)', 'MassiveIntentClassification (ms)', 'MassiveIntentClassification (my)', 'MassiveIntentClassification (nl)', 'MassiveIntentClassification (pl)', 'MassiveIntentClassification (pt)', 'MassiveIntentClassification (ro)', 'MassiveIntentClassification (ru)', 'MassiveIntentClassification (sl)', 'MassiveIntentClassification (sq)', 'MassiveIntentClassification (sw)', 'MassiveIntentClassification (ta)', 'MassiveIntentClassification (te)', 'MassiveIntentClassification (th)', 'MassiveIntentClassification (tl)', 'MassiveIntentClassification (tr)', 'MassiveIntentClassification (ur)', 'MassiveIntentClassification (vi)', 'MassiveIntentClassification (zh-CN)', 'MassiveIntentClassification (zh-TW)', 'MassiveScenarioClassification (af)', 'MassiveScenarioClassification (am)', 'MassiveScenarioClassification (ar)', 'MassiveScenarioClassification (az)', 'MassiveScenarioClassification (bn)', 'MassiveScenarioClassification (cy)', 'MassiveScenarioClassification (de)', 'MassiveScenarioClassification (el)', 'MassiveScenarioClassification (es)', 'MassiveScenarioClassification (fa)', 'MassiveScenarioClassification (fi)', 'MassiveScenarioClassification (fr)', 'MassiveScenarioClassification (he)', 'MassiveScenarioClassification (hi)', 'MassiveScenarioClassification (hu)', 'MassiveScenarioClassification (hy)', 'MassiveScenarioClassification (id)', 'MassiveScenarioClassification (is)', 'MassiveScenarioClassification (it)', 'MassiveScenarioClassification (ja)', 'MassiveScenarioClassification (jv)', 'MassiveScenarioClassification (ka)', 'MassiveScenarioClassification (km)', 'MassiveScenarioClassification (kn)', 'MassiveScenarioClassification (ko)', 'MassiveScenarioClassification (lv)', 'MassiveScenarioClassification (ml)', 'MassiveScenarioClassification (mn)', 'MassiveScenarioClassification (ms)', 'MassiveScenarioClassification (my)', 'MassiveScenarioClassification (nl)', 'MassiveScenarioClassification (pl)', 'MassiveScenarioClassification (pt)', 'MassiveScenarioClassification (ro)', 'MassiveScenarioClassification (ru)', 'MassiveScenarioClassification (sl)', 'MassiveScenarioClassification (sq)', 'MassiveScenarioClassification (sw)', 'MassiveScenarioClassification (ta)', 'MassiveScenarioClassification (te)', 'MassiveScenarioClassification (th)', 'MassiveScenarioClassification (tl)', 'MassiveScenarioClassification (tr)', 'MassiveScenarioClassification (ur)', 'MassiveScenarioClassification (vi)', 'MassiveScenarioClassification (zh-CN)', 'MassiveScenarioClassification (zh-TW)']

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


TASK_LIST_CLUSTERING_DE = [
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "TenKGnadClusteringP2P",
    "TenKGnadClusteringS2S",
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

TASK_LIST_RETRIEVAL_PL = [
    "ArguAna-PL",
    "DBPedia-PL",
    "FiQA2018-PL",
    "HotpotQA-PL",
    "MSMARCO-PL",
    "NFCorpus-PL",
    "NQ-PL",
    "Quora-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "TRECCOVID-PL",
]

TASK_LIST_RETRIEVAL_NORM = TASK_LIST_RETRIEVAL + [
    "CQADupstackAndroidRetrieval",
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
    if link is None:
        link = "https://huggingface.co/" + model_name
    # Remove user from model name
    return (
        f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name.split("/")[-1]}</a>'
    )

# Models without metadata, thus we cannot fetch their results naturally
EXTERNAL_MODELS = [
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "allenai-specter",
    "bert-base-swedish-cased",
    "bert-base-uncased",
    "contriever-base-msmarco",
    "cross-en-de-roberta-sentence-transformer",
    "dfm-encoder-large-v1",
    "dfm-sentence-encoder-large-1",
    "distiluse-base-multilingual-cased-v2",
    "DanskBERT",
    "e5-base",
    "e5-large",
    "e5-small",    
    "electra-small-nordic",
    "electra-small-swedish-cased-discriminator",
    "gbert-base",
    "gbert-large",
    "gelectra-base",
    "gelectra-large",
    "gottbert-base",
    "glove.6B.300d",
    "gtr-t5-base",
    "gtr-t5-large",
    "gtr-t5-xl",
    "gtr-t5-xxl",
    "komninos",
    "LASER2",
    "LaBSE",
    "msmarco-bert-co-condensor",
    "multilingual-e5-base",
    "multilingual-e5-large",
    "multilingual-e5-small",
    "nb-bert-base",
    "nb-bert-large",
    "norbert3-base",
    "norbert3-large",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "sentence-bert-swedish-cased",
    "sentence-t5-base",
    "sentence-t5-large",
    "sentence-t5-xl",
    "sentence-t5-xxl",
    "sup-simcse-bert-base-uncased",
    "text-embedding-ada-002",
    "text-similarity-ada-001",
    "text-similarity-babbage-001",
    "text-similarity-curie-001",
    "text-similarity-davinci-001",
    "text-search-ada-doc-001",
    "text-search-ada-001",
    "text-search-babbage-001",
    "text-search-curie-001",
    "text-search-davinci-001",
    "unsup-simcse-bert-base-uncased",
    "use-cmlm-multilingual",
    "xlm-roberta-base",
    "xlm-roberta-large",  
]

EXTERNAL_MODEL_TO_LINK = {
    "allenai-specter": "https://huggingface.co/sentence-transformers/allenai-specter",
    "allenai-specter": "https://huggingface.co/sentence-transformers/allenai-specter",
    "all-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    "bert-base-swedish-cased": "https://huggingface.co/KB/bert-base-swedish-cased",
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased",
    "contriever-base-msmarco": "https://huggingface.co/nthakur/contriever-base-msmarco",
    "cross-en-de-roberta-sentence-transformer": "https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "DanskBERT": "https://huggingface.co/vesteinn/DanskBERT",
    "distiluse-base-multilingual-cased-v2": "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2",
    "dfm-encoder-large-v1": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
    "dfm-sentence-encoder-large-1": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
    "e5-base": "https://huggingface.co/intfloat/e5-base",
    "e5-large": "https://huggingface.co/intfloat/e5-large",
    "e5-small": "https://huggingface.co/intfloat/e5-small",
    "electra-small-nordic": "https://huggingface.co/jonfd/electra-small-nordic",
    "electra-small-swedish-cased-discriminator": "https://huggingface.co/KBLab/electra-small-swedish-cased-discriminator",
    "gbert-base": "https://huggingface.co/deepset/gbert-base",
    "gbert-large": "https://huggingface.co/deepset/gbert-large",
    "gelectra-base": "https://huggingface.co/deepset/gelectra-base",
    "gelectra-large": "https://huggingface.co/deepset/gelectra-large",
    "glove.6B.300d": "https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d",
    "gottbert-base": "https://huggingface.co/uklfr/gottbert-base",
    "gtr-t5-base": "https://huggingface.co/sentence-transformers/gtr-t5-base",
    "gtr-t5-large": "https://huggingface.co/sentence-transformers/gtr-t5-large",
    "gtr-t5-xl": "https://huggingface.co/sentence-transformers/gtr-t5-xl",
    "gtr-t5-xxl": "https://huggingface.co/sentence-transformers/gtr-t5-xxl",
    "komninos": "https://huggingface.co/sentence-transformers/average_word_embeddings_komninos",
    "LASER2": "https://github.com/facebookresearch/LASER",
    "LaBSE": "https://huggingface.co/sentence-transformers/LaBSE",
    "msmarco-bert-co-condensor": "https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor",
    "multilingual-e5-base": "https://huggingface.co/intfloat/multilingual-e5-base",
    "multilingual-e5-large": "https://huggingface.co/intfloat/multilingual-e5-large",
    "multilingual-e5-small": "https://huggingface.co/intfloat/multilingual-e5-small",
    "nb-bert-base": "https://huggingface.co/NbAiLab/nb-bert-base",
    "nb-bert-large": "https://huggingface.co/NbAiLab/nb-bert-large",
    "norbert3-base": "https://huggingface.co/ltg/norbert3-base",
    "norbert3-large": "https://huggingface.co/ltg/norbert3-large",
    "paraphrase-multilingual-mpnet-base-v2": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2",    
    "paraphrase-multilingual-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-bert-swedish-cased": "https://huggingface.co/KBLab/sentence-bert-swedish-cased",
    "sentence-t5-base": "https://huggingface.co/sentence-transformers/sentence-t5-base",
    "sentence-t5-large": "https://huggingface.co/sentence-transformers/sentence-t5-large",
    "sentence-t5-xl": "https://huggingface.co/sentence-transformers/sentence-t5-xl",
    "sentence-t5-xxl": "https://huggingface.co/sentence-transformers/sentence-t5-xxl",
    "sup-simcse-bert-base-uncased": "https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased",
    "text-embedding-ada-002": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-similarity-ada-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-similarity-babbage-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-similarity-curie-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",        
    "text-similarity-davinci-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",    
    "text-search-ada-doc-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-ada-query-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-ada-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-curie-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-babbage-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-davinci-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "unsup-simcse-bert-base-uncased": "https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased",
    "use-cmlm-multilingual": "https://huggingface.co/sentence-transformers/use-cmlm-multilingual",
    "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base",
    "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large",
}

EXTERNAL_MODEL_TO_DIM = {
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "allenai-specter": 768,    
    "bert-base-swedish-cased": 768,
    "bert-base-uncased": 768,
    "contriever-base-msmarco": 768,
    "cross-en-de-roberta-sentence-transformer": 768,
    "DanskBERT": 768,
    "distiluse-base-multilingual-cased-v2": 512,
    "dfm-encoder-large-v1": 1024,
    "dfm-sentence-encoder-large-1": 1024,
    "e5-base": 768,
    "e5-small": 384,
    "e5-large": 1024,    
    "electra-small-nordic": 256,
    "electra-small-swedish-cased-discriminator": 256,
    "LASER2": 1024,
    "LaBSE": 768,
    "gbert-base": 768,
    "gbert-large": 1024,
    "gelectra-base": 768,
    "gelectra-large": 1024,
    "glove.6B.300d": 300,
    "gottbert-base": 768,    
    "gtr-t5-base": 768,
    "gtr-t5-large": 768,
    "gtr-t5-xl": 768,
    "gtr-t5-xxl": 768,
    "komninos": 300,
    "msmarco-bert-co-condensor": 768,
    "multilingual-e5-base": 768,
    "multilingual-e5-small": 384,
    "multilingual-e5-large": 1024,
    "nb-bert-base": 768,
    "nb-bert-large": 1024,
    "norbert3-base": 768,
    "norbert3-large": 1024,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-bert-swedish-cased": 768,
    "sentence-t5-base": 768,
    "sentence-t5-large": 768,
    "sentence-t5-xl": 768,
    "sentence-t5-xxl": 768,
    "sup-simcse-bert-base-uncased": 768,
    "use-cmlm-multilingual": 768,
    "unsup-simcse-bert-base-uncased": 768,
    "text-embedding-ada-002": 1536,
    "text-similarity-ada-001": 1024,
    "text-similarity-babbage-001": 2048,    
    "text-similarity-curie-001": 4096,
    "text-similarity-davinci-001": 12288,    
    "text-search-ada-doc-001": 1024,
    "text-search-ada-query-001": 1024,
    "text-search-ada-001": 1024,   
    "text-search-babbage-001": 2048,     
    "text-search-curie-001": 4096,
    "text-search-davinci-001": 12288,
    "xlm-roberta-base":  768,
    "xlm-roberta-large":  1024,
}


EXTERNAL_MODEL_TO_SEQLEN = {
    "all-MiniLM-L12-v2": 512,
    "all-MiniLM-L6-v2": 512,
    "all-mpnet-base-v2": 514,
    "allenai-specter": 512,
    "bert-base-swedish-cased": 512,    
    "bert-base-uncased": 512,
    "contriever-base-msmarco": 512,
    "cross-en-de-roberta-sentence-transformer": 514,
    "DanskBERT": 514,
    "dfm-encoder-large-v1": 512,
    "dfm-sentence-encoder-large-1": 512,
    "distiluse-base-multilingual-cased-v2": 512,
    "e5-base": 512,
    "e5-large": 512,    
    "e5-small": 512,    
    "electra-small-nordic": 512,
    "electra-small-swedish-cased-discriminator": 512,
    "gbert-base": 512,
    "gbert-large": 512,
    "gelectra-base": 512,
    "gelectra-large": 512,
    "gottbert-base": 512,
    "glove.6B.300d": "N/A",
    "gtr-t5-base": 512,
    "gtr-t5-large": 512,
    "gtr-t5-xl": 512,
    "gtr-t5-xxl": 512,
    "komninos": "N/A",
    "LASER2": "N/A",
    "LaBSE": 512,    
    "msmarco-bert-co-condensor": 512,
    "multilingual-e5-base": 514,
    "multilingual-e5-large": 514,    
    "multilingual-e5-small": 512,
    "nb-bert-base": 512,
    "nb-bert-large": 512,
    "norbert3-base": 512,
    "norbert3-large": 512,
    "paraphrase-multilingual-MiniLM-L12-v2": 512,
    "paraphrase-multilingual-mpnet-base-v2": 514,
    "sentence-bert-swedish-cased": 512,
    "sentence-t5-base": 512,
    "sentence-t5-large": 512,
    "sentence-t5-xl": 512,
    "sentence-t5-xxl": 512,
    "sup-simcse-bert-base-uncased": 512,
    "text-embedding-ada-002": 8191,
    "text-similarity-ada-001": 2046,
    "text-similarity-babbage-001": 2046,    
    "text-similarity-curie-001": 2046,
    "text-similarity-davinci-001": 2046,    
    "text-search-ada-doc-001": 2046,
    "text-search-ada-query-001": 2046,
    "text-search-ada-001": 2046,   
    "text-search-babbage-001": 2046,     
    "text-search-curie-001": 2046,
    "text-search-davinci-001": 2046,   
    "use-cmlm-multilingual": 512,
    "unsup-simcse-bert-base-uncased": 512,
    "xlm-roberta-base": 514,
    "xlm-roberta-large": 514,
}

EXTERNAL_MODEL_TO_SIZE = {
    "allenai-specter": 0.44,
    "all-MiniLM-L12-v2": 0.13,
    "all-MiniLM-L6-v2": 0.09,
    "all-mpnet-base-v2": 0.44,    
    "bert-base-uncased": 0.44,    
    "bert-base-swedish-cased": 0.50,
    "cross-en-de-roberta-sentence-transformer": 1.11,
    "contriever-base-msmarco": 0.44,    
    "DanskBERT": 0.50,
    "distiluse-base-multilingual-cased-v2": 0.54,
    "dfm-encoder-large-v1": 1.42,
    "dfm-sentence-encoder-large-1": 1.63,
    "e5-base": 0.44,
    "e5-small": 0.13,
    "e5-large": 1.34,      
    "electra-small-nordic": 0.09,
    "electra-small-swedish-cased-discriminator": 0.06,
    "gbert-base": 0.44,
    "gbert-large": 1.35,
    "gelectra-base": 0.44,
    "gelectra-large": 1.34,
    "glove.6B.300d": 0.48,
    "gottbert-base": 0.51,    
    "gtr-t5-base": 0.22,
    "gtr-t5-large": 0.67,
    "gtr-t5-xl": 2.48,
    "gtr-t5-xxl": 9.73,
    "komninos": 0.27,    
    "LASER2": 0.17,
    "LaBSE": 1.88,
    "msmarco-bert-co-condensor": 0.44,
    "multilingual-e5-base": 1.11,
    "multilingual-e5-small": 0.47,
    "multilingual-e5-large": 2.24,    
    "nb-bert-base": 0.71,
    "nb-bert-large": 1.42,
    "norbert3-base": 0.52,
    "norbert3-large": 1.47,
    "paraphrase-multilingual-mpnet-base-v2": 1.11,
    "paraphrase-multilingual-MiniLM-L12-v2": 0.47,
    "sentence-bert-swedish-cased": 0.50,
    "sentence-t5-base": 0.22,
    "sentence-t5-large": 0.67,
    "sentence-t5-xl": 2.48,
    "sentence-t5-xxl": 9.73,
    "sup-simcse-bert-base-uncased": 0.44,
    "unsup-simcse-bert-base-uncased": 0.44,
    "use-cmlm-multilingual": 1.89, 
    "xlm-roberta-base": 1.12,
    "xlm-roberta-large": 2.24,
}

MODELS_TO_SKIP = {
    "baseplate/instructor-large-1", # Duplicate
    "radames/e5-large", # Duplicate
    "gentlebowl/instructor-large-safetensors", # Duplicate
    "Consensus/instructor-base", # Duplicate
    "GovCompete/instructor-xl", # Duplicate
    "GovCompete/e5-large-v2", # Duplicate
    "t12e/instructor-base", # Duplicate
    "michaelfeil/ct2fast-e5-large-v2",
    "michaelfeil/ct2fast-e5-large",
    "michaelfeil/ct2fast-e5-small-v2",
    "newsrx/instructor-xl-newsrx",
    "newsrx/instructor-large-newsrx",
    "fresha/e5-large-v2-endpoint",
    "ggrn/e5-small-v2",
    "michaelfeil/ct2fast-e5-small",
    "jncraton/e5-small-v2-ct2-int8",
    "anttip/ct2fast-e5-small-v2-hfie",
    "newsrx/instructor-large",
    "newsrx/instructor-xl",
    "dmlls/all-mpnet-base-v2",
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
    if examples["mteb_dataset_name"] in TASK_LIST_CLASSIFICATION_NORM + TASK_LIST_CLASSIFICATION_DA + TASK_LIST_CLASSIFICATION_SV + TASK_LIST_CLASSIFICATION_NB:
        examples["mteb_task"] = "Classification"
    elif examples["mteb_dataset_name"] in TASK_LIST_CLUSTERING + TASK_LIST_CLUSTERING_DE:
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
    ds = load_dataset("mteb/results", model)#, download_mode='force_redownload', verification_mode="no_checks")
    # For local debugging:
    #, download_mode='force_redownload', verification_mode="no_checks")
    ds = ds.map(add_lang)
    ds = ds.map(add_task)
    base_dict = {"Model": make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, "https://huggingface.co/spaces/mteb/leaderboard"))}
    # For now only one metric per task - Could add more metrics lateron
    for task, metric in TASK_TO_METRIC.items():
        ds_dict = ds.filter(lambda x: (x["mteb_task"] == task) and (x["metric"] == metric))["test"].to_dict()
        ds_dict = {k: round(v, 2) for k, v in zip(ds_dict["mteb_dataset_name_with_lang"], ds_dict["score"])}
        EXTERNAL_MODEL_RESULTS[model][task][metric].append({**base_dict, **ds_dict})

def get_dim_seq_size(model):
    filenames = [sib.rfilename for sib in model.siblings]
    dim, seq, size = "", "", ""
    if "1_Pooling/config.json" in filenames:
        st_config_path = hf_hub_download(model.modelId, filename="1_Pooling/config.json")
        dim = json.load(open(st_config_path)).get("word_embedding_dimension", "")
    elif "2_Pooling/config.json" in filenames:
        st_config_path = hf_hub_download(model.modelId, filename="2_Pooling/config.json")
        dim = json.load(open(st_config_path)).get("word_embedding_dimension", "")
    if "config.json" in filenames:
        config_path = hf_hub_download(model.modelId, filename="config.json")
        config = json.load(open(config_path))
        if not dim:
            dim = config.get("hidden_dim", config.get("hidden_size", config.get("d_model", "")))
        seq = config.get("n_positions", config.get("max_position_embeddings", config.get("n_ctx", config.get("seq_length", ""))))
    # Get model file size without downloading
    if "pytorch_model.bin" in filenames:
        url = hf_hub_url(model.modelId, filename="pytorch_model.bin")
        meta = get_hf_file_metadata(url)
        size = round(meta.size / 1e9, 2)
    elif "pytorch_model.bin.index.json" in filenames:
        index_path = hf_hub_download(model.modelId, filename="pytorch_model.bin.index.json")
        """
        {
        "metadata": {
            "total_size": 28272820224
        },....
        """
        size = json.load(open(index_path))
        if ("metadata" in size) and ("total_size" in size["metadata"]):
            size = round(size["metadata"]["total_size"] / 1e9, 2)
    return dim, seq, size

def make_datasets_clickable(df):
    """Does not work"""
    if "BornholmBitextMining" in df.columns:
        link = "https://huggingface.co/datasets/strombergnlp/bornholmsk_parallel"
        df = df.rename(
            columns={f'BornholmBitextMining': '<a target="_blank" style="text-decoration: underline" href="{link}">BornholmBitextMining</a>',})
    return df


def add_rank(df):
    cols_to_rank = [col for col in df.columns if col not in ["Model", "Model Size (GB)", "Embedding Dimensions", "Sequence Length"]]
    if len(cols_to_rank) == 1:
        df.sort_values(cols_to_rank[0], ascending=False, inplace=True)
    else:
        df.insert(1, "Average", df[cols_to_rank].mean(axis=1, skipna=False))
        df.sort_values("Average", ascending=False, inplace=True)
    df.insert(0, "Rank", list(range(1, len(df) + 1)))
    df = df.round(2)
    # Fill NaN after averaging
    df.fillna("", inplace=True)
    return df

def get_mteb_data(tasks=["Clustering"], langs=[], datasets=[], fillna=True, add_emb_dim=False, task_to_metric=TASK_TO_METRIC, rank=True):
    api = HfApi()
    models = api.list_models(filter="mteb")
    # Initialize list to models that we cannot fetch metadata from
    df_list = []
    for model in EXTERNAL_MODEL_RESULTS:
        results_list = [res for task in tasks for res in EXTERNAL_MODEL_RESULTS[model][task][task_to_metric[task]]]
        if len(datasets) > 0:
            res = {k: v for d in results_list for k, v in d.items() if (k == "Model") or any([x in k for x in datasets])}
        elif langs:
            # Would be cleaner to rely on an extra language column instead
            langs_format = [f"({lang})" for lang in langs]
            res = {k: v for d in results_list for k, v in d.items() if any([k.split(" ")[-1] in (k, x) for x in langs_format])}
        else:
            res = {k: v for d in results_list for k, v in d.items()}
        # Model & at least one result
        if len(res) > 1:
            if add_emb_dim:
                res["Model Size (GB)"] = EXTERNAL_MODEL_TO_SIZE.get(model, "")
                res["Embedding Dimensions"] = EXTERNAL_MODEL_TO_DIM.get(model, "")
                res["Sequence Length"] = EXTERNAL_MODEL_TO_SEQLEN.get(model, "")
            df_list.append(res)
    
    for model in models:
        if model.modelId in MODELS_TO_SKIP: continue
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
        if len(datasets) > 0:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks) and any([x in sub_res.get("dataset", {}).get("name", "") for x in datasets])]
        elif langs:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks) and (sub_res.get("dataset", {}).get("config", "default") in ("default", *langs))]
        else:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if (sub_res.get("task", {}).get("type", "") in tasks)]
        out = [{res["dataset"]["name"].replace("MTEB ", ""): [round(score["value"], 2) for score in res["metrics"] if score["type"] == task_to_metric.get(res["task"]["type"])][0]} for res in task_results]
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = make_clickable_model(model.modelId)
        # Model & at least one result
        if len(out) > 1:
            if add_emb_dim:
                out["Embedding Dimensions"], out["Sequence Length"], out["Model Size (GB)"] = get_dim_seq_size(model)
            df_list.append(out)
    df = pd.DataFrame(df_list)
    # If there are any models that are the same, merge them
    # E.g. if out["Model"] has the same value in two places, merge & take whichever one is not NaN else just take the first one
    df = df.groupby("Model", as_index=False).first()
    # Put 'Model' column first
    cols = sorted(list(df.columns))
    cols.insert(0, cols.pop(cols.index("Model")))
    df = df[cols]
    if rank:
        df = add_rank(df)       
    if fillna:
        df.fillna("", inplace=True)
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
        fillna=False,
        add_emb_dim=True,
        rank=False,
    )
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

    DATA_OVERALL = DATA_OVERALL.round(2)

    DATA_CLASSIFICATION_EN = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_CLASSIFICATION])
    DATA_CLUSTERING = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_CLUSTERING])
    DATA_PAIR_CLASSIFICATION = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_PAIR_CLASSIFICATION])
    DATA_RERANKING = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_RERANKING])
    DATA_RETRIEVAL = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_RETRIEVAL])
    DATA_STS_EN = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_STS])
    DATA_SUMMARIZATION = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_SUMMARIZATION])

    # Fill NaN after averaging
    DATA_OVERALL.fillna("", inplace=True)

    DATA_OVERALL = DATA_OVERALL[["Rank", "Model", "Model Size (GB)", "Embedding Dimensions", "Sequence Length", f"Average ({len(TASK_LIST_EN)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION)} datasets)", f"Reranking Average ({len(TASK_LIST_RERANKING)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)", f"STS Average ({len(TASK_LIST_STS)} datasets)", f"Summarization Average ({len(TASK_LIST_SUMMARIZATION)} dataset)"]]

    return DATA_OVERALL

get_mteb_average()
DATA_BITEXT_MINING = get_mteb_data(["BitextMining"], [], TASK_LIST_BITEXT_MINING)
DATA_BITEXT_MINING_OTHER = get_mteb_data(["BitextMining"], [], TASK_LIST_BITEXT_MINING_OTHER)
DATA_CLASSIFICATION_DA = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_DA)
DATA_CLASSIFICATION_NB = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_NB)
DATA_CLASSIFICATION_SV = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_SV)
DATA_CLASSIFICATION_OTHER = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_OTHER)
DATA_CLUSTERING_GERMAN = get_mteb_data(["Clustering"], [], TASK_LIST_CLUSTERING_DE)
DATA_RETRIEVAL_PL = get_mteb_data(["Retrieval"], [], TASK_LIST_RETRIEVAL_PL)
DATA_STS = get_mteb_data(["STS"])

# Exact, add all non-nan integer values for every dataset
NUM_SCORES = 0
DATASETS = []
# LANGUAGES = []
for d in [DATA_BITEXT_MINING, DATA_BITEXT_MINING_OTHER, DATA_CLASSIFICATION_EN, DATA_CLASSIFICATION_DA, DATA_CLASSIFICATION_NB, DATA_CLASSIFICATION_SV, DATA_CLASSIFICATION_OTHER, DATA_CLUSTERING, DATA_CLUSTERING_GERMAN, DATA_PAIR_CLASSIFICATION, DATA_RERANKING, DATA_RETRIEVAL, DATA_STS_EN, DATA_STS, DATA_SUMMARIZATION]:
    # NUM_SCORES += d.iloc[:, 1:].apply(lambda x: sum([1 for y in x if isinstance(y, float) and not np.isnan(y)]), axis=1).sum()
    cols_to_ignore = 3 if "Average" in d.columns else 2
    # Count number of scores including only non-nan floats & excluding the rank column
    NUM_SCORES += d.iloc[:, cols_to_ignore:].notna().sum().sum()
    # Exclude rank & model name column (first two); Do not count different language versions as different datasets
    DATASETS += [i.split(" ")[0] for i in d.columns[cols_to_ignore:]]
    # LANGUAGES += [i.split(" ")[-1] for i in d.columns[cols_to_ignore:]]

NUM_DATASETS = len(set(DATASETS))
# NUM_LANGUAGES = len(set(LANGUAGES))

block = gr.Blocks()
with block:
    gr.Markdown(f"""
    Massive Text Embedding Benchmark (MTEB) Leaderboard. To submit, refer to the <a href="https://github.com/embeddings-benchmark/mteb#leaderboard" target="_blank" style="text-decoration: underline">MTEB GitHub repository</a> 🤗 Refer to the [MTEB paper](https://arxiv.org/abs/2210.07316) for details on metrics, tasks and models.

    - **Total Datasets**: {NUM_DATASETS}
    - **Total Languages**: 113
    - **Total Scores**: {NUM_SCORES}
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
                    datatype=["number", "markdown"] + ["number"] * len(DATA_OVERALL.columns),
                    type="pandas",
                    wrap=True,
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                data_run.click(get_mteb_average, inputs=None, outputs=data_overall)
        with gr.TabItem("Bitext Mining"):
            with gr.TabItem("English-X"):
                with gr.Row():
                        gr.Markdown("""
                        **Bitext Mining Leaderboard 🎌**
                        
                        - **Metric:** [F1](https://huggingface.co/spaces/evaluate-metric/f1)
                        - **Languages:** 117 (Pairs of: English & other language)
                        """)
                with gr.Row():
                    data_bitext_mining = gr.components.Dataframe(
                        DATA_BITEXT_MINING,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_BITEXT_MINING.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_bitext_mining = gr.Variable(value=["BitextMining"])
                    lang_bitext_mining_other = gr.Variable(value=[])
                    datasets_bitext_mining_other = gr.Variable(value=TASK_LIST_BITEXT_MINING)                    
                    data_run.click(
                        get_mteb_data,
                        inputs=[task_bitext_mining, lang_bitext_mining_other, datasets_bitext_mining_other],
                        outputs=data_bitext_mining,
                    )
            with gr.TabItem("Danish"):
                with gr.Row():
                        gr.Markdown("""
                        **Bitext Mining Danish Leaderboard 🇩🇰🎌**
                        
                        - **Metric:** [F1](https://huggingface.co/spaces/evaluate-metric/f1)
                        - **Languages:** Danish & Bornholmsk (Danish Dialect)
                        - **Credits:** [Kenneth Enevoldsen](https://github.com/KennethEnevoldsen), [scandinavian-embedding-benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)
                        """)
                with gr.Row():
                    data_bitext_mining_other = gr.components.Dataframe(
                        DATA_BITEXT_MINING_OTHER,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_BITEXT_MINING_OTHER.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_bitext_mining_other = gr.Variable(value=["BitextMining"])
                    lang_bitext_mining_other = gr.Variable(value=[])
                    datasets_bitext_mining_other = gr.Variable(value=TASK_LIST_BITEXT_MINING_OTHER)
                    data_run.click(
                        get_mteb_data,
                        inputs=[
                            task_bitext_mining_other,
                            lang_bitext_mining_other,
                            datasets_bitext_mining_other,
                        ],
                        outputs=data_bitext_mining_other,
                    )
        with gr.TabItem("Classification"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Leaderboard ❤️**
                    
                    - **Metric:** [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
                    - **Languages:** English
                    """)
                with gr.Row():
                    data_classification_en = gr.components.Dataframe(
                        DATA_CLASSIFICATION_EN,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLASSIFICATION_EN.columns),
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
            with gr.TabItem("Danish"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Danish Leaderboard 🤍🇩🇰**
                    
                    - **Metric:** [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
                    - **Languages:** Danish
                    - **Credits:** [Kenneth Enevoldsen](https://github.com/KennethEnevoldsen), [scandinavian-embedding-benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)
                    """)
                with gr.Row():
                    data_classification_da = gr.components.Dataframe(
                        DATA_CLASSIFICATION_DA,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLASSIFICATION_DA.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_classification_da = gr.Button("Refresh")
                    task_classification_da = gr.Variable(value=["Classification"])
                    lang_classification_da = gr.Variable(value=[])
                    datasets_classification_da = gr.Variable(value=TASK_LIST_CLASSIFICATION_DA)
                    data_run_classification_da.click(
                        get_mteb_data,
                        inputs=[
                            task_classification_da,
                            lang_classification_da,
                            datasets_classification_da,
                        ],
                        outputs=data_classification_da,
                    )
            with gr.TabItem("Norwegian"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Norwegian Leaderboard 💙🇳🇴**
                    
                    - **Metric:** [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
                    - **Languages:** Norwegian Bokmål
                    - **Credits:** [Kenneth Enevoldsen](https://github.com/KennethEnevoldsen), [scandinavian-embedding-benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)                                
                    """)
                with gr.Row():
                    data_classification_nb = gr.components.Dataframe(
                        DATA_CLASSIFICATION_NB,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLASSIFICATION_NB.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_classification_nb = gr.Button("Refresh")
                    task_classification_nb = gr.Variable(value=["Classification"])
                    lang_classification_nb = gr.Variable(value=[])
                    datasets_classification_nb = gr.Variable(value=TASK_LIST_CLASSIFICATION_NB)
                    data_run_classification_nb.click(
                        get_mteb_data,
                        inputs=[
                            task_classification_nb,
                            lang_classification_nb,
                            datasets_classification_nb,
                        ],
                        outputs=data_classification_nb,
                    )                    
            with gr.TabItem("Swedish"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Swedish Leaderboard 💛🇸🇪**
                    
                    - **Metric:** [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
                    - **Languages:** Swedish
                    - **Credits:** [Kenneth Enevoldsen](https://github.com/KennethEnevoldsen), [scandinavian-embedding-benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)
                    """)
                with gr.Row():
                    data_classification_sv = gr.components.Dataframe(
                        DATA_CLASSIFICATION_SV,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLASSIFICATION_SV.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_classification_sv = gr.Button("Refresh")
                    task_classification_sv = gr.Variable(value=["Classification"])
                    lang_classification_sv = gr.Variable(value=[])
                    datasets_classification_sv = gr.Variable(value=TASK_LIST_CLASSIFICATION_SV)
                    data_run_classification_sv.click(
                        get_mteb_data,
                        inputs=[
                            task_classification_sv,
                            lang_classification_sv,
                            datasets_classification_sv,
                        ],
                        outputs=data_classification_sv,
                    )
            with gr.TabItem("Other"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Other Languages Leaderboard 💜💚💙**
                    
                    - **Metric:** [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
                    - **Languages:** 47 (Only languages not included in the other tabs)
                    """)
                with gr.Row():
                    data_classification = gr.components.Dataframe(
                        DATA_CLASSIFICATION_OTHER,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLASSIFICATION_OTHER) * 10,
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_classification = gr.Variable(value=["Classification"])
                    lang_classification = gr.Variable(value=[])
                    datasets_classification = gr.Variable(value=TASK_LIST_CLASSIFICATION_OTHER)                 
                    data_run.click(
                        get_mteb_data,
                        inputs=[
                            task_classification,
                            lang_classification,
                            datasets_classification,
                        ],
                        outputs=data_classification,
                    )                                      
        with gr.TabItem("Clustering"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Clustering Leaderboard ✨**
                    
                    - **Metric:** Validity Measure (v_measure)
                    - **Languages:** English
                    """)
                with gr.Row():
                    data_clustering = gr.components.Dataframe(
                        DATA_CLUSTERING,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLUSTERING.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_clustering = gr.Variable(value=["Clustering"])
                    empty = gr.Variable(value=[])
                    datasets_clustering = gr.Variable(value=TASK_LIST_CLUSTERING)
                    data_run.click(
                        get_mteb_data,
                        inputs=[task_clustering, empty, datasets_clustering],
                        outputs=data_clustering,
                    )
            with gr.TabItem("German"):
                with gr.Row():
                    gr.Markdown("""
                    **Clustering German Leaderboard ✨🇩🇪**
                    
                    - **Metric:** Validity Measure (v_measure)
                    - **Languages:** German
                    - **Credits:** [Silvan](https://github.com/slvnwhrl)
                    """)
                with gr.Row():
                    data_clustering_de = gr.components.Dataframe(
                        DATA_CLUSTERING_GERMAN,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLUSTERING_GERMAN.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_clustering_de = gr.Variable(value=["Clustering"])
                    empty_de = gr.Variable(value=[])
                    datasets_clustering_de = gr.Variable(value=TASK_LIST_CLUSTERING_DE)
                    data_run.click(
                        get_mteb_data,
                        inputs=[task_clustering_de, empty_de, datasets_clustering_de],
                        outputs=data_clustering_de,
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
                    datatype=["number", "markdown"] + ["number"] * len(DATA_PAIR_CLASSIFICATION.columns),
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
                    datatype=["number", "markdown"] + ["number"] * len(DATA_RERANKING.columns),
                    type="pandas",
                )
            with gr.Row():
                data_run = gr.Button("Refresh")
                task_reranking = gr.Variable(value=["Reranking"])
                metric_reranking = gr.Variable(value="map")
                data_run.click(
                    get_mteb_data, inputs=[task_reranking], outputs=data_reranking
                )
        with gr.TabItem("Retrieval"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Retrieval Leaderboard  🔎**
                    
                    - **Metric:** Normalized Discounted Cumulative Gain @ k (ndcg_at_10)
                    - **Languages:** English
                    """)
                with gr.Row():
                    data_retrieval = gr.components.Dataframe(
                        DATA_RETRIEVAL,
                        # Add support for more columns than existing as a buffer for CQADupstack & other Retrieval tasks (e.g. MSMARCOv2)
                        datatype=["number", "markdown"] + ["number"] * len(DATA_RETRIEVAL.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_retrieval = gr.Variable(value=["Retrieval"])
                    data_run.click(
                        get_mteb_data, inputs=[task_retrieval], outputs=data_retrieval
                    )
            with gr.TabItem("Polish"):
                with gr.Row():
                    gr.Markdown("""
                    **Retrieval Leaderboard 🇵🇱**
                    
                    - **Metric:** Normalized Discounted Cumulative Gain @ k (ndcg_at_10)
                    - **Languages:** Polish
                    - **Credits:** [Konrad Wojtasik](https://github.com/kwojtasi) & [BEIR-PL](https://arxiv.org/abs/2305.19840)
                    """)
                with gr.Row():
                    data_retrieval_pl = gr.components.Dataframe(
                        DATA_RETRIEVAL_PL,
                        # Add support for more columns than existing as a buffer for CQADupstack & other Retrieval tasks (e.g. MSMARCOv2)
                        datatype=["number", "markdown"] + ["number"] * len(DATA_RETRIEVAL_PL.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run = gr.Button("Refresh")
                    task_retrieval_pl = gr.Variable(value=["Retrieval"])
                    data_run.click(
                        get_mteb_data, inputs=[task_retrieval_pl], outputs=data_retrieval_pl
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
                        datatype=["number", "markdown"] + ["number"] * len(DATA_STS_EN.columns),
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
                        DATA_STS,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_STS.columns) * 2,
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
                    DATA_SUMMARIZATION,
                    datatype=["number", "markdown"] + ["number"] * 2,
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
    gr.Markdown(r"""
    
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
    # Running the functions on page load in addition to when the button is clicked
    # This is optional - If deactivated the data loaded at "Build time" is shown like for Overall tab
    """
    block.load(get_mteb_data, inputs=[task_bitext_mining], outputs=data_bitext_mining)
    block.load(get_mteb_data, inputs=[task_classification_en, lang_classification_en], outputs=data_classification_en)
    block.load(get_mteb_data, inputs=[task_classification], outputs=data_classification)
    block.load(get_mteb_data, inputs=[task_clustering, empty, datasets_clustering], outputs=data_clustering)
    block.load(get_mteb_data, inputs=[task_clustering_de, empty_de, datasets_clustering_de], outputs=data_clustering_de)
    block.load(get_mteb_data, inputs=[task_pair_classification], outputs=data_pair_classification)
    block.load(get_mteb_data, inputs=[task_retrieval], outputs=data_retrieval)
    block.load(get_mteb_data, inputs=[task_reranking], outputs=data_reranking)
    block.load(get_mteb_data, inputs=[task_sts_en, lang_sts_en], outputs=data_sts_en)
    block.load(get_mteb_data, inputs=[task_sts], outputs=data_sts)
    block.load(get_mteb_data, inputs=[task_summarization], outputs=data_summarization)
    """

block.queue(concurrency_count=40, max_size=10)
block.launch()


# Possible changes:
# Could check if tasks are valid (Currently users could just invent new tasks - similar for languages)
# Could make it load in the background without the Gradio logo closer to the Deep RL space
# Could add graphs / other visual content
# Could add verification marks

# Sources:
# https://huggingface.co/spaces/gradio/leaderboard
# https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard
# https://getemoji.com/
