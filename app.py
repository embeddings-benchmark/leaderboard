from functools import partial, reduce
import json
import os
import re

from datasets import load_dataset
import gradio as gr
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import metadata_load
import pandas as pd
from tqdm.autonotebook import tqdm

from utils.model_size import get_model_parameters_memory

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
TASK_LIST_BITEXT_MINING_DA = ["BornholmBitextMining"]

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

TASK_LIST_CLASSIFICATION_FR = [
    "AmazonReviewsClassification (fr)",
    "MasakhaNEWSClassification (fra)",
    "MassiveIntentClassification (fr)",
    "MassiveScenarioClassification (fr)",
    "MTOPDomainClassification (fr)",
    "MTOPIntentClassification (fr)",
]

TASK_LIST_CLASSIFICATION_NB = [
    "NoRecClassification",
    "NordicLangClassification",
    "NorwegianParliament",
    "MassiveIntentClassification (nb)",
    "MassiveScenarioClassification (nb)",
    "ScalaNbClassification",
]

TASK_LIST_CLASSIFICATION_PL = [
    "AllegroReviews",
    "CBD",
    "MassiveIntentClassification (pl)",
    "MassiveScenarioClassification (pl)",
    "PAC",
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
]

TASK_LIST_CLASSIFICATION_SV = [
    "DalajClassification",
    "MassiveIntentClassification (sv)",
    "MassiveScenarioClassification (sv)",
    "NordicLangClassification",
    "ScalaSvClassification",
    "SweRecClassification",
]

TASK_LIST_CLASSIFICATION_ZH = [
    "AmazonReviewsClassification (zh)",
    "IFlyTek",
    "JDReview",
    "MassiveIntentClassification (zh-CN)",
    "MassiveScenarioClassification (zh-CN)",
    "MultilingualSentiment",
    "OnlineShopping",
    "TNews",
    "Waimai",
]

TASK_LIST_CLASSIFICATION_OTHER = ['AmazonCounterfactualClassification (de)', 'AmazonCounterfactualClassification (ja)', 'AmazonReviewsClassification (de)', 'AmazonReviewsClassification (es)', 'AmazonReviewsClassification (fr)', 'AmazonReviewsClassification (ja)', 'AmazonReviewsClassification (zh)', 'MTOPDomainClassification (de)', 'MTOPDomainClassification (es)', 'MTOPDomainClassification (fr)', 'MTOPDomainClassification (hi)', 'MTOPDomainClassification (th)', 'MTOPIntentClassification (de)', 'MTOPIntentClassification (es)', 'MTOPIntentClassification (fr)', 'MTOPIntentClassification (hi)', 'MTOPIntentClassification (th)', 'MassiveIntentClassification (af)', 'MassiveIntentClassification (am)', 'MassiveIntentClassification (ar)', 'MassiveIntentClassification (az)', 'MassiveIntentClassification (bn)', 'MassiveIntentClassification (cy)', 'MassiveIntentClassification (de)', 'MassiveIntentClassification (el)', 'MassiveIntentClassification (es)', 'MassiveIntentClassification (fa)', 'MassiveIntentClassification (fi)', 'MassiveIntentClassification (fr)', 'MassiveIntentClassification (he)', 'MassiveIntentClassification (hi)', 'MassiveIntentClassification (hu)', 'MassiveIntentClassification (hy)', 'MassiveIntentClassification (id)', 'MassiveIntentClassification (is)', 'MassiveIntentClassification (it)', 'MassiveIntentClassification (ja)', 'MassiveIntentClassification (jv)', 'MassiveIntentClassification (ka)', 'MassiveIntentClassification (km)', 'MassiveIntentClassification (kn)', 'MassiveIntentClassification (ko)', 'MassiveIntentClassification (lv)', 'MassiveIntentClassification (ml)', 'MassiveIntentClassification (mn)', 'MassiveIntentClassification (ms)', 'MassiveIntentClassification (my)', 'MassiveIntentClassification (nl)', 'MassiveIntentClassification (pt)', 'MassiveIntentClassification (ro)', 'MassiveIntentClassification (ru)', 'MassiveIntentClassification (sl)', 'MassiveIntentClassification (sq)', 'MassiveIntentClassification (sw)', 'MassiveIntentClassification (ta)', 'MassiveIntentClassification (te)', 'MassiveIntentClassification (th)', 'MassiveIntentClassification (tl)', 'MassiveIntentClassification (tr)', 'MassiveIntentClassification (ur)', 'MassiveIntentClassification (vi)', 'MassiveIntentClassification (zh-TW)', 'MassiveScenarioClassification (af)', 'MassiveScenarioClassification (am)', 'MassiveScenarioClassification (ar)', 'MassiveScenarioClassification (az)', 'MassiveScenarioClassification (bn)', 'MassiveScenarioClassification (cy)', 'MassiveScenarioClassification (de)', 'MassiveScenarioClassification (el)', 'MassiveScenarioClassification (es)', 'MassiveScenarioClassification (fa)', 'MassiveScenarioClassification (fi)', 'MassiveScenarioClassification (fr)', 'MassiveScenarioClassification (he)', 'MassiveScenarioClassification (hi)', 'MassiveScenarioClassification (hu)', 'MassiveScenarioClassification (hy)', 'MassiveScenarioClassification (id)', 'MassiveScenarioClassification (is)', 'MassiveScenarioClassification (it)', 'MassiveScenarioClassification (ja)', 'MassiveScenarioClassification (jv)', 'MassiveScenarioClassification (ka)', 'MassiveScenarioClassification (km)', 'MassiveScenarioClassification (kn)', 'MassiveScenarioClassification (ko)', 'MassiveScenarioClassification (lv)', 'MassiveScenarioClassification (ml)', 'MassiveScenarioClassification (mn)', 'MassiveScenarioClassification (ms)', 'MassiveScenarioClassification (my)', 'MassiveScenarioClassification (nl)', 'MassiveScenarioClassification (pt)', 'MassiveScenarioClassification (ro)', 'MassiveScenarioClassification (ru)', 'MassiveScenarioClassification (sl)', 'MassiveScenarioClassification (sq)', 'MassiveScenarioClassification (sw)', 'MassiveScenarioClassification (ta)', 'MassiveScenarioClassification (te)', 'MassiveScenarioClassification (th)', 'MassiveScenarioClassification (tl)', 'MassiveScenarioClassification (tr)', 'MassiveScenarioClassification (ur)', 'MassiveScenarioClassification (vi)', 'MassiveScenarioClassification (zh-TW)']

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

TASK_LIST_CLUSTERING_FR = [
    "AlloProfClusteringP2P",
    "AlloProfClusteringS2S",
    "HALClusteringS2S",
    "MLSUMClusteringP2P",
    "MLSUMClusteringS2S",
    "MasakhaNEWSClusteringP2P (fra)",
    "MasakhaNEWSClusteringS2S (fra)",
]

TASK_LIST_CLUSTERING_PL = [
    "8TagsClustering",
]

TASK_LIST_CLUSTERING_ZH = [
    "CLSClusteringP2P",
    "CLSClusteringS2S",
    "ThuNewsClusteringP2P",
    "ThuNewsClusteringS2S",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_PAIR_CLASSIFICATION_FR = [
    "OpusparcusPC (fr)",
    "PawsX (fr)",
]

TASK_LIST_PAIR_CLASSIFICATION_PL = [
    "CDSC-E",
    "PPC",
    "PSC",
    "SICK-E-PL",    
]    

TASK_LIST_PAIR_CLASSIFICATION_ZH = [
    "Cmnli",
    "Ocnli",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RERANKING_FR = [
    "AlloprofReranking",
    "SyntecReranking",
]

TASK_LIST_RERANKING_ZH = [
    "CMedQAv1",
    "CMedQAv2",
    "MMarcoReranking",
    "T2Reranking",
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

TASK_LIST_RETRIEVAL_FR = [
    "AlloprofRetrieval",
    "BSARDRetrieval",
    "MintakaRetrieval (fr)",
#    "MultiLongDocRetrieval",
    "SyntecRetrieval",
    "XPQARetrieval (fr)",
]

TASK_LIST_RETRIEVAL_LAW = [
    "AILACasedocs",
    "AILAStatutes",
    "GerDaLIRSmall",
    "LeCaRDv2",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "LegalQuAD",
    "LegalSummarization",
]

TASK_LIST_RETRIEVAL_PL = [
    "ArguAna-PL",
    "DBPedia-PL",
    "FiQA-PL",
    "HotpotQA-PL",
    "MSMARCO-PL",
    "NFCorpus-PL",
    "NQ-PL",
    "Quora-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "TRECCOVID-PL",
]

TASK_LIST_RETRIEVAL_ZH = [
    "CmedqaRetrieval",
    "CovidRetrieval",
    "DuRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "MMarcoRetrieval",
    "T2Retrieval",
    "VideoRetrieval",
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

TASK_LIST_STS_FR = [
    "STS22 (fr)",
    "STSBenchmarkMultilingualSTS (fr)",
    "SICKFr",
]

TASK_LIST_STS_PL = [
    "CDSC-R",
    "SICK-R-PL",
    "STS22 (pl)",
]

TASK_LIST_STS_ZH = [
    "AFQMC",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "QBQTC",
    "STS22 (zh)",
    "STSB",
]

TASK_LIST_STS_OTHER = ["STS17 (ar-ar)", "STS17 (en-ar)", "STS17 (en-de)", "STS17 (en-tr)", "STS17 (es-en)", "STS17 (es-es)", "STS17 (fr-en)", "STS17 (it-en)", "STS17 (ko-ko)", "STS17 (nl-en)", "STS22 (ar)", "STS22 (de)", "STS22 (de-en)", "STS22 (de-fr)", "STS22 (de-pl)", "STS22 (es)", "STS22 (es-en)", "STS22 (es-it)", "STS22 (fr)", "STS22 (fr-pl)", "STS22 (it)", "STS22 (pl)", "STS22 (pl-en)", "STS22 (ru)", "STS22 (tr)", "STS22 (zh-en)", "STSBenchmark",]

TASK_LIST_SUMMARIZATION = ["SummEval",]

TASK_LIST_SUMMARIZATION_FR = ["SummEvalFr"]

TASK_LIST_EN = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS + TASK_LIST_SUMMARIZATION
TASK_LIST_FR = TASK_LIST_CLASSIFICATION_FR + TASK_LIST_CLUSTERING_FR + TASK_LIST_PAIR_CLASSIFICATION_FR + TASK_LIST_RERANKING_FR + TASK_LIST_RETRIEVAL_FR + TASK_LIST_STS_FR + TASK_LIST_SUMMARIZATION_FR
TASK_LIST_PL = TASK_LIST_CLASSIFICATION_PL + TASK_LIST_CLUSTERING_PL + TASK_LIST_PAIR_CLASSIFICATION_PL + TASK_LIST_RETRIEVAL_PL + TASK_LIST_STS_PL
TASK_LIST_ZH = TASK_LIST_CLASSIFICATION_ZH + TASK_LIST_CLUSTERING_ZH + TASK_LIST_PAIR_CLASSIFICATION_ZH + TASK_LIST_RERANKING_ZH + TASK_LIST_RETRIEVAL_ZH + TASK_LIST_STS_ZH

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
    "Baichuan-text-embedding",
    "Cohere-embed-english-v3.0",
    "Cohere-embed-multilingual-v3.0",
    "Cohere-embed-multilingual-light-v3.0",
    "DanskBERT",
    "LASER2",
    "LLM2Vec-Llama-supervised",
    "LLM2Vec-Llama-unsupervised",
    "LLM2Vec-Mistral-supervised",
    "LLM2Vec-Mistral-unsupervised",
    "LLM2Vec-Sheared-Llama-supervised",
    "LLM2Vec-Sheared-Llama-unsupervised",
    "LaBSE",
    "OpenSearch-text-hybrid",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "allenai-specter",
    "bert-base-10lang-cased",
    "bert-base-15lang-cased",
    "bert-base-25lang-cased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",    
    "bert-base-swedish-cased",
    "bert-base-uncased",
    "bge-base-zh-v1.5",
    "bge-large-en-v1.5",
    "bge-large-zh-v1.5",
    "bge-large-zh-noinstruct",
    "bge-small-zh-v1.5",
    "contriever-base-msmarco",
    "cross-en-de-roberta-sentence-transformer",
    "dfm-encoder-large-v1",
    "dfm-sentence-encoder-large-1",
    "distiluse-base-multilingual-cased-v2",
    "e5-base",
    "e5-large",
    "e5-mistral-7b-instruct",
    "e5-small",
    "electra-small-nordic",
    "electra-small-swedish-cased-discriminator",
    "flaubert_base_cased",
    "flaubert_base_uncased",
    "flaubert_large_cased",
    "gbert-base",
    "gbert-large",
    "gelectra-base",
    "gelectra-large",
    "glove.6B.300d",
    "google-gecko.text-embedding-preview-0409",
    "google-gecko-256.text-embedding-preview-0409",
    "gottbert-base",
    "gtr-t5-base",
    "gtr-t5-large",
    "gtr-t5-xl",
    "gtr-t5-xxl",
    "herbert-base-retrieval-v2",
    "komninos",
    "luotuo-bert-medium",
    "m3e-base",
    "m3e-large",
    "mistral-embed",
    "msmarco-bert-co-condensor",
    "multi-qa-MiniLM-L6-cos-v1",    
    "multilingual-e5-base",
    "multilingual-e5-large",
    "multilingual-e5-small",
    "nb-bert-base",
    "nb-bert-large",
    "nomic-embed-text-v1.5-64",
    "nomic-embed-text-v1.5-128",
    "nomic-embed-text-v1.5-256",
    "nomic-embed-text-v1.5-512",
    "norbert3-base",
    "norbert3-large",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "sentence-bert-swedish-cased",
    "sentence-camembert-base",
    "sentence-camembert-large",
    "sentence-croissant-llm-base",    
    "sentence-t5-base",
    "sentence-t5-large",
    "sentence-t5-xl",
    "sentence-t5-xxl",
    "silver-retriever-base-v1",
    "sup-simcse-bert-base-uncased",
    "st-polish-paraphrase-from-distilroberta",
    "st-polish-paraphrase-from-mpnet",
    "text2vec-base-chinese",
    "text2vec-base-multilingual",
    "text2vec-large-chinese",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-3-large-256",
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
    "titan-embed-text-v1",
    "udever-bloom-1b1",
    "udever-bloom-560m",
    "universal-sentence-encoder-multilingual-3",
    "universal-sentence-encoder-multilingual-large-3",
    "unsup-simcse-bert-base-uncased",
    "use-cmlm-multilingual",
    "voyage-2",
    "voyage-code-2",
    "voyage-large-2-instruct",
    "voyage-law-2",
    "voyage-lite-01-instruct",
    "voyage-lite-02-instruct",
    "xlm-roberta-base",
    "xlm-roberta-large",
]

EXTERNAL_MODEL_TO_LINK = {
    "Baichuan-text-embedding": "https://platform.baichuan-ai.com/docs/text-Embedding",
    "Cohere-embed-english-v3.0": "https://huggingface.co/Cohere/Cohere-embed-english-v3.0",
    "Cohere-embed-multilingual-v3.0": "https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0",
    "Cohere-embed-multilingual-light-v3.0": "https://huggingface.co/Cohere/Cohere-embed-multilingual-light-v3.0",
    "DanskBERT": "https://huggingface.co/vesteinn/DanskBERT",
    "LASER2": "https://github.com/facebookresearch/LASER",
    "LLM2Vec-Llama-supervised": "https://huggingface.co/McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
    "LLM2Vec-Llama-unsupervised": "https://huggingface.co/McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
    "LLM2Vec-Mistral-supervised": "https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    "LLM2Vec-Mistral-unsupervised": "https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    "LLM2Vec-Sheared-Llama-supervised": "https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
    "LLM2Vec-Sheared-Llama-unsupervised": "https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
    "LaBSE": "https://huggingface.co/sentence-transformers/LaBSE",
    "OpenSearch-text-hybrid": "https://help.aliyun.com/zh/open-search/vector-search-edition/hybrid-retrieval",
    "allenai-specter": "https://huggingface.co/sentence-transformers/allenai-specter",
    "allenai-specter": "https://huggingface.co/sentence-transformers/allenai-specter",
    "all-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    "bert-base-10lang-cased": "https://huggingface.co/Geotrend/bert-base-10lang-cased",
    "bert-base-15lang-cased": "https://huggingface.co/Geotrend/bert-base-15lang-cased",
    "bert-base-25lang-cased": "https://huggingface.co/Geotrend/bert-base-25lang-cased",
    "bert-base-multilingual-cased": "https://huggingface.co/google-bert/bert-base-multilingual-cased",
    "bert-base-multilingual-uncased": "https://huggingface.co/google-bert/bert-base-multilingual-uncased",
    "bert-base-swedish-cased": "https://huggingface.co/KB/bert-base-swedish-cased",
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased",
    "bge-base-zh-v1.5": "https://huggingface.co/BAAI/bge-base-zh-v1.5",
    "bge-large-en-v1.5": "https://huggingface.co/BAAI/bge-large-en-v1.5",
    "bge-large-zh-v1.5": "https://huggingface.co/BAAI/bge-large-zh-v1.5",
    "bge-large-zh-noinstruct": "https://huggingface.co/BAAI/bge-large-zh-noinstruct",
    "bge-small-zh-v1.5": "https://huggingface.co/BAAI/bge-small-zh-v1.5",
    "camembert-base": "https://huggingface.co/almanach/camembert-base",
    "camembert-large": "https://huggingface.co/almanach/camembert-large",
    "contriever-base-msmarco": "https://huggingface.co/nthakur/contriever-base-msmarco",
    "cross-en-de-roberta-sentence-transformer": "https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "distilbert-base-25lang-cased": "https://huggingface.co/Geotrend/distilbert-base-25lang-cased",
    "distilbert-base-en-fr-cased": "https://huggingface.co/Geotrend/distilbert-base-en-fr-cased",
    "distilbert-base-en-fr-es-pt-it-cased": "https://huggingface.co/Geotrend/distilbert-base-en-fr-es-pt-it-cased",
    "distilbert-base-fr-cased": "https://huggingface.co/Geotrend/distilbert-base-fr-cased",
    "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased",
    "distiluse-base-multilingual-cased-v2": "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2",
    "dfm-encoder-large-v1": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
    "dfm-sentence-encoder-large-1": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
    "e5-base": "https://huggingface.co/intfloat/e5-base",
    "e5-large": "https://huggingface.co/intfloat/e5-large",
    "e5-mistral-7b-instruct": "https://huggingface.co/intfloat/e5-mistral-7b-instruct",
    "e5-small": "https://huggingface.co/intfloat/e5-small",
    "electra-small-nordic": "https://huggingface.co/jonfd/electra-small-nordic",
    "electra-small-swedish-cased-discriminator": "https://huggingface.co/KBLab/electra-small-swedish-cased-discriminator",
    "flaubert_base_cased": "https://huggingface.co/flaubert/flaubert_base_cased",
    "flaubert_base_uncased": "https://huggingface.co/flaubert/flaubert_base_uncased",
    "flaubert_large_cased": "https://huggingface.co/flaubert/flaubert_large_cased",
    "gbert-base": "https://huggingface.co/deepset/gbert-base",
    "gbert-large": "https://huggingface.co/deepset/gbert-large",
    "gelectra-base": "https://huggingface.co/deepset/gelectra-base",
    "gelectra-large": "https://huggingface.co/deepset/gelectra-large",
    "glove.6B.300d": "https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d",
    "google-gecko.text-embedding-preview-0409": "https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#latest_models",
    "google-gecko-256.text-embedding-preview-0409": "https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#latest_models",
    "gottbert-base": "https://huggingface.co/uklfr/gottbert-base",
    "gtr-t5-base": "https://huggingface.co/sentence-transformers/gtr-t5-base",
    "gtr-t5-large": "https://huggingface.co/sentence-transformers/gtr-t5-large",
    "gtr-t5-xl": "https://huggingface.co/sentence-transformers/gtr-t5-xl",
    "gtr-t5-xxl": "https://huggingface.co/sentence-transformers/gtr-t5-xxl",
    "herbert-base-retrieval-v2": "https://huggingface.co/ipipan/herbert-base-retrieval-v2",
    "komninos": "https://huggingface.co/sentence-transformers/average_word_embeddings_komninos",
    "luotuo-bert-medium": "https://huggingface.co/silk-road/luotuo-bert-medium",
    "m3e-base": "https://huggingface.co/moka-ai/m3e-base",
    "m3e-large": "https://huggingface.co/moka-ai/m3e-large",
    "mistral-embed": "https://docs.mistral.ai/guides/embeddings",
    "msmarco-bert-co-condensor": "https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor",
    "multi-qa-MiniLM-L6-cos-v1": "https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "multilingual-e5-base": "https://huggingface.co/intfloat/multilingual-e5-base",
    "multilingual-e5-large": "https://huggingface.co/intfloat/multilingual-e5-large",
    "multilingual-e5-small": "https://huggingface.co/intfloat/multilingual-e5-small",
    "nb-bert-base": "https://huggingface.co/NbAiLab/nb-bert-base",
    "nb-bert-large": "https://huggingface.co/NbAiLab/nb-bert-large",
    "nomic-embed-text-v1.5-64": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text-v1.5-128": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text-v1.5-256": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text-v1.5-512": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
    "norbert3-base": "https://huggingface.co/ltg/norbert3-base",
    "norbert3-large": "https://huggingface.co/ltg/norbert3-large",
    "paraphrase-multilingual-mpnet-base-v2": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2",    
    "paraphrase-multilingual-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-camembert-base": "https://huggingface.co/dangvantuan/sentence-camembert-base",
    "sentence-camembert-large": "https://huggingface.co/dangvantuan/sentence-camembert-large",
    "sentence-croissant-llm-base": "https://huggingface.co/Wissam42/sentence-croissant-llm-base",
    "sentence-bert-swedish-cased": "https://huggingface.co/KBLab/sentence-bert-swedish-cased",
    "sentence-t5-base": "https://huggingface.co/sentence-transformers/sentence-t5-base",
    "sentence-t5-large": "https://huggingface.co/sentence-transformers/sentence-t5-large",
    "sentence-t5-xl": "https://huggingface.co/sentence-transformers/sentence-t5-xl",
    "sentence-t5-xxl": "https://huggingface.co/sentence-transformers/sentence-t5-xxl",
    "silver-retriever-base-v1": "https://huggingface.co/ipipan/silver-retriever-base-v1",
    "sup-simcse-bert-base-uncased": "https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased",
    "st-polish-paraphrase-from-distilroberta": "https://huggingface.co/sdadas/st-polish-paraphrase-from-distilroberta",
    "st-polish-paraphrase-from-mpnet": "https://huggingface.co/sdadas/st-polish-paraphrase-from-mpnet",
    "text2vec-base-chinese": "https://huggingface.co/shibing624/text2vec-base-chinese",
    "text2vec-large-chinese": "https://huggingface.co/GanymedeNil/text2vec-large-chinese",
    "text-embedding-3-small": "https://openai.com/blog/new-embedding-models-and-api-updates",
    "text-embedding-3-large": "https://openai.com/blog/new-embedding-models-and-api-updates",
    "text-embedding-3-large-256": "https://openai.com/blog/new-embedding-models-and-api-updates",
    "text-embedding-ada-002": "https://openai.com/blog/new-and-improved-embedding-model",
    "text-similarity-ada-001": "https://openai.com/blog/introducing-text-and-code-embeddings",
    "text-similarity-babbage-001": "https://openai.com/blog/introducing-text-and-code-embeddings",
    "text-similarity-curie-001": "https://openai.com/blog/introducing-text-and-code-embeddings",        
    "text-similarity-davinci-001": "https://openai.com/blog/introducing-text-and-code-embeddings",    
    "text-search-ada-doc-001": "https://openai.com/blog/introducing-text-and-code-embeddings",
    "text-search-ada-query-001": "https://openai.com/blog/introducing-text-and-code-embeddings",
    "text-search-ada-001": "https://openai.com/blog/introducing-text-and-code-embeddings",
    "text-search-curie-001": "https://openai.com/blog/introducing-text-and-code-embeddings",
    "text-search-babbage-001": "https://openai.com/blog/introducing-text-and-code-embeddings",
    "text-search-davinci-001": "https://openai.com/blog/introducing-text-and-code-embeddings",
    "titan-embed-text-v1": "https://docs.aws.amazon.com/bedrock/latest/userguide/embeddings.html",
    "udever-bloom-1b1": "https://huggingface.co/izhx/udever-bloom-1b1",
    "udever-bloom-560m": "https://huggingface.co/izhx/udever-bloom-560m",
    "universal-sentence-encoder-multilingual-3": "https://huggingface.co/vprelovac/universal-sentence-encoder-multilingual-3",
    "universal-sentence-encoder-multilingual-large-3": "https://huggingface.co/vprelovac/universal-sentence-encoder-multilingual-large-3",
    "unsup-simcse-bert-base-uncased": "https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased",
    "use-cmlm-multilingual": "https://huggingface.co/sentence-transformers/use-cmlm-multilingual",
    "voyage-2": "https://docs.voyageai.com/embeddings/",
    "voyage-code-2": "https://docs.voyageai.com/embeddings/",
    "voyage-large-2-instruct": "https://docs.voyageai.com/embeddings/",
    "voyage-law-2": "https://docs.voyageai.com/embeddings/",
    "voyage-lite-01-instruct": "https://docs.voyageai.com/embeddings/",
    "voyage-lite-02-instruct": "https://docs.voyageai.com/embeddings/",
    "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base",
    "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large",
}

EXTERNAL_MODEL_TO_DIM = {
    "Baichuan-text-embedding": 1024,
    "Cohere-embed-english-v3.0": 1024,
    "Cohere-embed-multilingual-v3.0": 1024,
    "Cohere-embed-multilingual-light-v3.0": 384,
    "DanskBERT": 768,
    "LASER2": 1024,
    "LLM2Vec-Llama-supervised": 4096,
    "LLM2Vec-Llama-unsupervised": 4096,
    "LLM2Vec-Mistral-supervised": 4096,
    "LLM2Vec-Mistral-unsupervised": 4096,
    "LLM2Vec-Sheared-Llama-supervised": 2048,
    "LLM2Vec-Sheared-Llama-unsupervised": 2048,
    "LaBSE": 768,
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "allenai-specter": 768,
    "bert-base-10lang-cased": 768,
    "bert-base-15lang-cased": 768,
    "bert-base-25lang-cased": 768,
    "bert-base-multilingual-cased": 768,
    "bert-base-multilingual-uncased": 768,
    "bert-base-swedish-cased": 768,
    "bert-base-uncased": 768,
    "bge-base-zh-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "bge-large-zh-v1.5": 1024,
    "bge-large-zh-noinstruct": 1024,
    "bge-small-zh-v1.5": 512,
    "camembert-base": 512,
    "camembert-large": 768,
    "contriever-base-msmarco": 768,
    "cross-en-de-roberta-sentence-transformer": 768,
    "distilbert-base-25lang-cased": 768,
    "distilbert-base-en-fr-cased": 768,
    "distilbert-base-en-fr-es-pt-it-cased": 768,
    "distilbert-base-fr-cased": 768,
    "distilbert-base-uncased": 768,
    "distiluse-base-multilingual-cased-v2": 512,
    "dfm-encoder-large-v1": 1024,
    "dfm-sentence-encoder-large-1": 1024,
    "e5-base": 768,
    "e5-large": 1024,
    "e5-mistral-7b-instruct": 4096,
    "e5-small": 384,
    "electra-small-nordic": 256,
    "electra-small-swedish-cased-discriminator": 256,
    "flaubert_base_cased": 768,
    "flaubert_base_uncased": 768,
    "flaubert_large_cased": 1024,
    "luotuo-bert-medium": 768,
    "gbert-base": 768,
    "gbert-large": 1024,
    "gelectra-base": 768,
    "gelectra-large": 1024,
    "glove.6B.300d": 300,
    "google-gecko.text-embedding-preview-0409": 768,
    "google-gecko-256.text-embedding-preview-0409": 256,
    "gottbert-base": 768,    
    "gtr-t5-base": 768,
    "gtr-t5-large": 768,
    "gtr-t5-xl": 768,
    "gtr-t5-xxl": 768,
    "herbert-base-retrieval-v2": 768,
    "komninos": 300,
    "m3e-base": 768,
    "m3e-large": 768,
    "mistral-embed": 1024,
    "msmarco-bert-co-condensor": 768,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "multilingual-e5-base": 768,
    "multilingual-e5-small": 384,
    "multilingual-e5-large": 1024,
    "nb-bert-base": 768,
    "nb-bert-large": 1024,
    "nomic-embed-text-v1.5-64": 64,
    "nomic-embed-text-v1.5-128": 128,
    "nomic-embed-text-v1.5-256": 256,
    "nomic-embed-text-v1.5-512": 512,
    "norbert3-base": 768,
    "norbert3-large": 1024,
    "OpenSearch-text-hybrid": 1792,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-camembert-base": 768,
    "sentence-camembert-large": 1024,
    "sentence-croissant-llm-base": 2048,
    "sentence-bert-swedish-cased": 768,
    "sentence-t5-base": 768,
    "sentence-t5-large": 768,
    "sentence-t5-xl": 768,
    "sentence-t5-xxl": 768,
    "silver-retriever-base-v1": 768,
    "sup-simcse-bert-base-uncased": 768,
    "st-polish-paraphrase-from-distilroberta": 768,
    "st-polish-paraphrase-from-mpnet": 768,
    "text2vec-base-chinese": 768,
    "text2vec-large-chinese": 1024,
    "text-embedding-3-large": 3072,
    "text-embedding-3-large-256": 256,
    "text-embedding-3-small": 1536,
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
    "titan-embed-text-v1": 1536,
    "udever-bloom-1b1": 1536,
    "udever-bloom-560m": 1024,
    "universal-sentence-encoder-multilingual-3": 512,
    "universal-sentence-encoder-multilingual-large-3": 512,
    "unsup-simcse-bert-base-uncased": 768,
    "use-cmlm-multilingual": 768,
    "voyage-2": 1024,
    "voyage-code-2": 1536,
    "voyage-large-2-instruct": 1536,
    "voyage-law-2": 1024,
    "voyage-lite-01-instruct": 1024,
    "voyage-lite-02-instruct": 1024,
    "xlm-roberta-base":  768,
    "xlm-roberta-large":  1024,
}

EXTERNAL_MODEL_TO_SEQLEN = {
    "Baichuan-text-embedding": 512,
    "Cohere-embed-english-v3.0": 512,
    "Cohere-embed-multilingual-v3.0": 512,
    "Cohere-embed-multilingual-light-v3.0": 512,
    "DanskBERT": 514,
    "LASER2": "N/A",
    "LLM2Vec-Llama-supervised": 4096,
    "LLM2Vec-Llama-unsupervised": 4096,
    "LLM2Vec-Mistral-supervised": 32768,
    "LLM2Vec-Mistral-unsupervised": 32768,
    "LLM2Vec-Sheared-Llama-supervised": 4096,
    "LLM2Vec-Sheared-Llama-unsupervised": 4096,    
    "LaBSE": 512,
    "all-MiniLM-L12-v2": 512,
    "all-MiniLM-L6-v2": 512,
    "all-mpnet-base-v2": 514,
    "allenai-specter": 512,
    "bert-base-10lang-cased": 512,
    "bert-base-15lang-cased": 512,
    "bert-base-25lang-cased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-multilingual-uncased": 512,    
    "bert-base-swedish-cased": 512,
    "bert-base-uncased": 512,
    "bge-base-zh-v1.5": 512,
    "bge-large-en-v1.5": 512,
    "bge-large-zh-v1.5": 512,
    "bge-large-zh-noinstruct": 512,
    "bge-small-zh-v1.5": 512,
    "camembert-base": 512,
    "camembert-large": 512,
    "contriever-base-msmarco": 512,
    "cross-en-de-roberta-sentence-transformer": 514,
    "distilbert-base-25lang-cased": 512,
    "distilbert-base-en-fr-cased": 512,
    "distilbert-base-en-fr-es-pt-it-cased": 512,
    "distilbert-base-fr-cased": 512,
    "distilbert-base-uncased": 512,
    "dfm-encoder-large-v1": 512,
    "dfm-sentence-encoder-large-1": 512,
    "distiluse-base-multilingual-cased-v2": 512,
    "e5-base": 512,
    "e5-large": 512,
    "e5-mistral-7b-instruct": 32768,
    "e5-small": 512,
    "electra-small-nordic": 512,
    "electra-small-swedish-cased-discriminator": 512,
    "flaubert_base_cased": 512,
    "flaubert_base_uncased": 512,
    "flaubert_large_cased": 512,
    "gbert-base": 512,
    "gbert-large": 512,
    "gelectra-base": 512,
    "gelectra-large": 512,
    "google-gecko.text-embedding-preview-0409": 2048,
    "google-gecko-256.text-embedding-preview-0409": 2048,
    "gottbert-base": 512,
    "glove.6B.300d": "N/A",
    "gtr-t5-base": 512,
    "gtr-t5-large": 512,
    "gtr-t5-xl": 512,
    "gtr-t5-xxl": 512,
    "herbert-base-retrieval-v2": 514,
    "komninos": "N/A",
    "luotuo-bert-medium": 512,
    "m3e-base": 512,
    "m3e-large": 512,
#    "mistral-embed": "?",
    "msmarco-bert-co-condensor": 512,
    "multi-qa-MiniLM-L6-cos-v1": 512,
    "multilingual-e5-base": 514,
    "multilingual-e5-large": 514,    
    "multilingual-e5-small": 512,
    "nb-bert-base": 512,
    "nb-bert-large": 512,
    "nomic-embed-text-v1.5-64": 8192,
    "nomic-embed-text-v1.5-128": 8192,
    "nomic-embed-text-v1.5-256": 8192,
    "nomic-embed-text-v1.5-512": 8192,
    "norbert3-base": 512,
    "norbert3-large": 512,
    "OpenSearch-text-hybrid": 512,
    "paraphrase-multilingual-MiniLM-L12-v2": 512,
    "paraphrase-multilingual-mpnet-base-v2": 514,
    "sentence-camembert-base": 512,
    "sentence-camembert-large": 512,
    "sentence-croissant-llm-base": 2048,    
    "sentence-bert-swedish-cased": 512,
    "sentence-t5-base": 512,
    "sentence-t5-large": 512,
    "sentence-t5-xl": 512,
    "sentence-t5-xxl": 512,
    "silver-retriever-base-v1": 514,
    "sup-simcse-bert-base-uncased": 512,
    "st-polish-paraphrase-from-distilroberta": 514,
    "st-polish-paraphrase-from-mpnet": 514,
    "text2vec-base-chinese": 512,
    "text2vec-large-chinese": 512,
    "text-embedding-3-large": 8191,
    "text-embedding-3-large-256": 8191,
    "text-embedding-3-small": 8191,
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
    "titan-embed-text-v1": 8000,
    "udever-bloom-1b1": 2048,
    "udever-bloom-560m": 2048,
    "universal-sentence-encoder-multilingual-3": 512,
    "universal-sentence-encoder-multilingual-large-3": 512,    
    "use-cmlm-multilingual": 512,
    "unsup-simcse-bert-base-uncased": 512,
    "voyage-2": 1024,
    "voyage-code-2": 16000,
    "voyage-large-2-instruct": 16000,
    "voyage-law-2": 4000,
    "voyage-lite-01-instruct": 4000,
    "voyage-lite-02-instruct": 4000,
    "xlm-roberta-base": 514,
    "xlm-roberta-large": 514,
}

EXTERNAL_MODEL_TO_SIZE = {
    "DanskBERT": 125,
    "LASER2": 43,
    "LLM2Vec-Llama-supervised": 6607,
    "LLM2Vec-Llama-unsupervised": 6607,
    "LLM2Vec-Mistral-supervised": 7111,
    "LLM2Vec-Mistral-unsupervised": 7111,
    "LLM2Vec-Sheared-Llama-supervised": 1280,
    "LLM2Vec-Sheared-Llama-unsupervised": 1280,
    "LaBSE": 471,
    "allenai-specter": 110,
    "all-MiniLM-L12-v2": 33,
    "all-MiniLM-L6-v2": 23,
    "all-mpnet-base-v2": 110,
    "bert-base-10lang-cased": 138,
    "bert-base-15lang-cased": 138,
    "bert-base-25lang-cased": 138,
    "bert-base-multilingual-cased": 179,
    "bert-base-multilingual-uncased": 168,
    "bert-base-uncased": 110,
    "bert-base-swedish-cased": 125,
    "bge-base-zh-v1.5": 102,
    "bge-large-zh-v1.5": 326,
    "bge-large-zh-noinstruct": 326,
    "bge-small-zh-v1.5": 24,
    "camembert-base": 111,
    "camembert-large": 338,
    "cross-en-de-roberta-sentence-transformer": 278,
    "contriever-base-msmarco": 110,
    "distilbert-base-25lang-cased": 110,
    "distilbert-base-en-fr-cased": 110,
    "distilbert-base-en-fr-es-pt-it-cased": 110,
    "distilbert-base-fr-cased": 110,
    "distilbert-base-uncased": 110,
    "distiluse-base-multilingual-cased-v2": 135,
    "dfm-encoder-large-v1": 355,
    "dfm-sentence-encoder-large-1": 355,
    "e5-base": 110,
    "e5-large": 335,
    "e5-mistral-7b-instruct": 7111,
    "e5-small": 33,
    "electra-small-nordic": 23,
    "electra-small-swedish-cased-discriminator": 16,
    "flaubert_base_cased": 138,
    "flaubert_base_uncased": 138,
    "flaubert_large_cased": 372,
    "gbert-base": 110,
    "gbert-large": 337,
    "gelectra-base": 110,
    "gelectra-large": 335,
    "glove.6B.300d": 120,
    "google-gecko.text-embedding-preview-0409": 1200,
    "google-gecko-256.text-embedding-preview-0409": 1200,
    "gottbert-base": 127,
    "gtr-t5-base": 110,
    "gtr-t5-large": 168,
    "gtr-t5-xl": 1240,
    "gtr-t5-xxl": 4865,
    "herbert-base-retrieval-v2": 125,
    "komninos": 134,
    "luotuo-bert-medium": 328,
    "m3e-base": 102,
    "m3e-large": 102,
    "msmarco-bert-co-condensor": 110,
    "multi-qa-MiniLM-L6-cos-v1": 23,
    "multilingual-e5-base": 278,
    "multilingual-e5-small": 118,
    "multilingual-e5-large": 560,
    "nb-bert-base": 179,
    "nb-bert-large": 355,
    "nomic-embed-text-v1.5-64": 138,
    "nomic-embed-text-v1.5-128": 138,
    "nomic-embed-text-v1.5-256": 138,
    "nomic-embed-text-v1.5-512": 138,
    "norbert3-base": 131,
    "norbert3-large": 368,
    "paraphrase-multilingual-mpnet-base-v2": 278,
    "paraphrase-multilingual-MiniLM-L12-v2": 118,
    "sentence-camembert-base": 110,
    "sentence-camembert-large": 337,
    "sentence-croissant-llm-base": 1280,
    "sentence-bert-swedish-cased": 125,
    "sentence-t5-base": 110,
    "sentence-t5-large": 168,
    "sentence-t5-xl": 1240,
    "sentence-t5-xxl": 4865,
    "silver-retriever-base-v1": 125,
    "sup-simcse-bert-base-uncased": 110,
    "st-polish-paraphrase-from-distilroberta": 125,
    "st-polish-paraphrase-from-mpnet": 125,    
    "text2vec-base-chinese": 102,
    "text2vec-large-chinese": 326,
    "unsup-simcse-bert-base-uncased": 110,
    "use-cmlm-multilingual": 472,
    #"voyage-law-2": 1220,
    "voyage-lite-02-instruct": 1220,
    "xlm-roberta-base": 279,
    "xlm-roberta-large": 560,
}

PROPRIETARY_MODELS = {
    "Baichuan-text-embedding",
    "Cohere-embed-english-v3.0",
    "Cohere-embed-multilingual-v3.0",
    "Cohere-embed-multilingual-light-v3.0",
    "OpenSearch-text-hybrid",
    "mistral-embed",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-3-large-256",
    "text-embedding-ada-002",
    "text-similarity-ada-001",
    "text-similarity-babbage-001",
    "text-similarity-curie-001",
    "text-similarity-davinci-001",
    "text-search-ada-doc-001",
    "text-search-ada-query-001",
    "text-search-ada-001",
    "text-search-curie-001",
    "text-search-babbage-001",
    "text-search-davinci-001",
    "titan-embed-text-v1",
    "voyage-2",
    "voyage-code-2",
    "voyage-law-2",
    "voyage-lite-01-instruct",
    "voyage-lite-02-instruct",
    "google-gecko.text-embedding-preview-0409",
    "google-gecko-256.text-embedding-preview-0409",
}

PROPRIETARY_MODELS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, "https://huggingface.co/spaces/mteb/leaderboard"))
    for model in PROPRIETARY_MODELS
}

SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS = {
    "allenai-specter",
    "allenai-specter",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "bert-base-10lang-cased",
    "bert-base-15lang-cased",
    "bert-base-25lang-cased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "bert-base-swedish-cased",
    "bert-base-uncased",
    "bge-base-zh-v1.5",
    "bge-large-zh-v1.5",
    "bge-large-zh-noinstruct",
    "bge-small-zh-v1.5",
    "camembert-base",
    "camembert-large",
    "contriever-base-msmarco",
    "cross-en-de-roberta-sentence-transformer",
    "DanskBERT",
    "distilbert-base-25lang-cased",
    "distilbert-base-en-fr-cased",
    "distilbert-base-en-fr-es-pt-it-cased",
    "distilbert-base-fr-cased",
    "distilbert-base-uncased",
    "distiluse-base-multilingual-cased-v2",
    "dfm-encoder-large-v1",
    "dfm-sentence-encoder-large-1",
    "e5-base",
    "e5-large",
    "e5-mistral-7b-instruct",
    "e5-small",
    "electra-small-nordic",
    "electra-small-swedish-cased-discriminator",
    "flaubert_base_cased",
    "flaubert_base_uncased",
    "flaubert_large_cased",
    "gbert-base",
    "gbert-large",
    "gelectra-base",
    "gelectra-large",
    "glove.6B.300d",
    "gottbert-base",
    "gtr-t5-base",
    "gtr-t5-large",
    "gtr-t5-xl",
    "gtr-t5-xxl",
    "herbert-base-retrieval-v2",
    "komninos",
    "luotuo-bert-medium",
    "LaBSE",
    "m3e-base",
    "m3e-large",
    "msmarco-bert-co-condensor",
    "multi-qa-MiniLM-L6-cos-v1",
    "multilingual-e5-base",
    "multilingual-e5-large",
    "multilingual-e5-small",
    "nb-bert-base",
    "nb-bert-large",
    "nomic-embed-text-v1.5-64",
    "nomic-embed-text-v1.5-128",
    "nomic-embed-text-v1.5-256",
    "nomic-embed-text-v1.5-512",
    "norbert3-base",
    "norbert3-large",
    "paraphrase-multilingual-mpnet-base-v2",    
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-camembert-base",
    "sentence-camembert-large",
    "sentence-croissant-llm-base",
    "sentence-bert-swedish-cased",
    "sentence-t5-base",
    "sentence-t5-large",
    "sentence-t5-xl",
    "sentence-t5-xxl",
    "silver-retriever-base-v1",
    "sup-simcse-bert-base-uncased",
    "st-polish-paraphrase-from-distilroberta",
    "st-polish-paraphrase-from-mpnet",
    "text2vec-base-chinese",
    "text2vec-large-chinese",
    "udever-bloom-1b1",
    "udever-bloom-560m",
    "universal-sentence-encoder-multilingual-3",
    "universal-sentence-encoder-multilingual-large-3",
    "unsup-simcse-bert-base-uncased",
    "use-cmlm-multilingual",
    "xlm-roberta-base",
    "xlm-roberta-large",
}
SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS = {
    make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model, "https://huggingface.co/spaces/mteb/leaderboard"))
    for model in SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS
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
    "cgldo/semanticClone",
    "Malmuk1/e5-large-v2_Sharded",
    "jncraton/gte-small-ct2-int8",
    "Einas/einas_ashkar",
    "gruber/e5-small-v2-ggml",
    "jncraton/bge-small-en-ct2-int8",
    "vectoriseai/bge-small-en",
    "recipe/embeddings",
    "dhairya0907/thenlper-get-large",
    "Narsil/bge-base-en",
    "kozistr/fused-large-en",
    "sionic-ai/sionic-ai-v2", # Wait for https://huggingface.co/sionic-ai/sionic-ai-v2/discussions/1
    "sionic-ai/sionic-ai-v1", # Wait for https://huggingface.co/sionic-ai/sionic-ai-v2/discussions/1
    "BAAI/bge-large-en", # Deprecated in favor of v1.5
    "BAAI/bge-base-en", # Deprecated in favor of v1.5
    "BAAI/bge-small-en", # Deprecated in favor of v1.5
    "d0rj/e5-large-en-ru",
    "d0rj/e5-base-en-ru",
    "d0rj/e5-small-en-ru",
    "aident-ai/bge-base-en-onnx",
    "barisaydin/bge-base-en",
    "barisaydin/gte-large",
    "barisaydin/gte-base",
    "barisaydin/gte-small",
    "barisaydin/bge-small-en",
    "odunola/e5-base-v2",
    "goldenrooster/multilingual-e5-large",
    "davidpeer/gte-small",
    "barisaydin/bge-large-en",
    "jamesgpt1/english-large-v1",
    "vectoriseai/bge-large-en-v1.5",
    "vectoriseai/bge-base-en-v1.5",
    "vectoriseai/instructor-large",
    "vectoriseai/instructor-base",
    "vectoriseai/gte-large",
    "vectoriseai/gte-base",
    "vectoriseai/e5-large-v2",
    "vectoriseai/bge-small-en-v1.5",
    "vectoriseai/e5-base-v2",
    "vectoriseai/e5-large",
    "vectoriseai/multilingual-e5-large",
    "vectoriseai/gte-small",
    "vectoriseai/ember-v1",
    "vectoriseai/e5-base",
    "vectoriseai/e5-small-v2",
    "michaelfeil/ct2fast-bge-large-en-v1.5",
    "michaelfeil/ct2fast-bge-large-en-v1.5",
    "michaelfeil/ct2fast-bge-base-en-v1.5",
    "michaelfeil/ct2fast-gte-large",
    "michaelfeil/ct2fast-gte-base",
    "michaelfeil/ct2fast-bge-small-en-v1.5",
    "rizki/bgr-tf",
    "ef-zulla/e5-multi-sml-torch",
    "cherubhao/yogamodel",
    "morgendigital/multilingual-e5-large-quantized",
    "jncraton/gte-tiny-ct2-int8",
    "Research2NLP/electrical_stella",
    "Intel/bge-base-en-v1.5-sts-int8-static",
    "Intel/bge-base-en-v1.5-sts-int8-dynamic",
    "Intel/bge-base-en-v1.5-sst2",
    "Intel/bge-base-en-v1.5-sst2-int8-static",
    "Intel/bge-base-en-v1.5-sst2-int8-dynamic",
    "Intel/bge-small-en-v1.5-sst2",
    "Intel/bge-small-en-v1.5-sst2-int8-dynamic",
    "Intel/bge-small-en-v1.5-sst2-int8-static",
    "binqiangliu/EmbeddingModlebgelargeENv1.5",
    "DecisionOptimizationSystem/DeepFeatEmbeddingLargeContext",
    "woody72/multilingual-e5-base",
    "Severian/embed",
    "Frazic/udever-bloom-3b-sentence",
    "jamesgpt1/zzz",
    "karrar-alwaili/UAE-Large-V1",
    "odunola/UAE-Large-VI",
    "shubham-bgi/UAE-Large",
    "retrainai/instructor-xl",
    "weakit-v/bge-base-en-v1.5-onnx",
    "ieasybooks/multilingual-e5-large-onnx",
    "gizmo-ai/Cohere-embed-multilingual-v3.0",
    "jingyeom/korean_embedding_model",
    "barisaydin/text2vec-base-multilingual",
    "mlx-community/multilingual-e5-large-mlx",
    "mlx-community/multilingual-e5-base-mlx",
    "mlx-community/multilingual-e5-small-mlx",
    "maiyad/multilingual-e5-small",
    "khoa-klaytn/bge-base-en-v1.5-angle",
    "khoa-klaytn/bge-small-en-v1.5-angle",
    "mixamrepijey/instructor-small",
    "mixamrepijey/instructor-models",
    "lsf1000/bge-evaluation", # Empty
    "giulio98/placeholder", # Empty
    "Severian/nomic", # Copy
    "atian-chapters/Chapters-SFR-Embedding-Mistral", # Copy
    "rlsChapters/Chapters-SFR-Embedding-Mistral", # Copy
    "TitanML/jina-v2-base-en-embed", # Copy
    "MaziyarPanahi/GritLM-8x7B-GGUF", # GGUF variant
    "Geolumina/instructor-xl", # Duplicate
    "krilecy/e5-mistral-7b-instruct",
    "beademiguelperez/sentence-transformers-multilingual-e5-small",
    "arcdev/SFR-Embedding-Mistral",
    "arcdev/e5-mistral-7b-instruct",
    "Koat/gte-tiny",
    "SmartComponents/bge-micro-v2",
    "ildodeltaRule/multilingual-e5-large",
    "hsikchi/dump",
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse",
    "jncraton/GIST-small-Embedding-v0-ct2-int8",
    "jncraton/stella-base-en-v2-ct2-int8",
    "lightbird-ai/nomic",
    "jamesdborin/jina-v2-base-en-embed",
    "iampanda/Test",
}


def add_lang(examples):
    if not(examples["eval_language"]):
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"]
    else:
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"] + f' ({examples["eval_language"]})'
    return examples

def norm(names): return set([name.split(" ")[0] for name in names])

def add_task(examples):
    # Could be added to the dataset loading script instead
    if examples["mteb_dataset_name"] in norm(TASK_LIST_CLASSIFICATION + TASK_LIST_CLASSIFICATION_DA + TASK_LIST_CLASSIFICATION_FR + TASK_LIST_CLASSIFICATION_NB + TASK_LIST_CLASSIFICATION_PL + TASK_LIST_CLASSIFICATION_SV + TASK_LIST_CLASSIFICATION_ZH):
        examples["mteb_task"] = "Classification"
    elif examples["mteb_dataset_name"] in norm(TASK_LIST_CLUSTERING + TASK_LIST_CLUSTERING_DE + TASK_LIST_CLUSTERING_FR + TASK_LIST_CLUSTERING_PL + TASK_LIST_CLUSTERING_ZH):
        examples["mteb_task"] = "Clustering"
    elif examples["mteb_dataset_name"] in norm(TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_PAIR_CLASSIFICATION_FR + TASK_LIST_PAIR_CLASSIFICATION_PL + TASK_LIST_PAIR_CLASSIFICATION_ZH):
        examples["mteb_task"] = "PairClassification"
    elif examples["mteb_dataset_name"] in norm(TASK_LIST_RERANKING + TASK_LIST_RERANKING_FR + TASK_LIST_RERANKING_ZH):
        examples["mteb_task"] = "Reranking"
    elif examples["mteb_dataset_name"] in norm(TASK_LIST_RETRIEVAL_NORM + TASK_LIST_RETRIEVAL_FR + TASK_LIST_RETRIEVAL_PL + TASK_LIST_RETRIEVAL_ZH + TASK_LIST_RETRIEVAL_LAW):
        examples["mteb_task"] = "Retrieval"
    elif examples["mteb_dataset_name"] in norm(TASK_LIST_STS + TASK_LIST_STS_FR + TASK_LIST_STS_PL + TASK_LIST_STS_ZH):
        examples["mteb_task"] = "STS"
    elif examples["mteb_dataset_name"] in norm(TASK_LIST_SUMMARIZATION + TASK_LIST_SUMMARIZATION_FR):
        examples["mteb_task"] = "Summarization"
    elif examples["mteb_dataset_name"] in norm(TASK_LIST_BITEXT_MINING + TASK_LIST_BITEXT_MINING_DA):
        examples["mteb_task"] = "BitextMining"
    else:
        print("WARNING: Task not found for dataset", examples["mteb_dataset_name"])
        examples["mteb_task"] = "Unknown"
    return examples

if os.path.exists("EXTERNAL_MODEL_RESULTS.json"):
    with open("EXTERNAL_MODEL_RESULTS.json") as f:
        EXTERNAL_MODEL_RESULTS = json.load(f)
    # Update with models not contained
    models_to_run = []
    for model in EXTERNAL_MODELS:
        if model not in EXTERNAL_MODEL_RESULTS:
            models_to_run.append(model)
            EXTERNAL_MODEL_RESULTS[model] = {k: {v: []} for k, v in TASK_TO_METRIC.items()}
else:
    EXTERNAL_MODEL_RESULTS = {model: {k: {v: []} for k, v in TASK_TO_METRIC.items()} for model in EXTERNAL_MODELS}
    models_to_run = EXTERNAL_MODELS

pbar = tqdm(models_to_run, desc="Fetching external model results")
for model in pbar:
    pbar.set_description(f"Fetching external model results for {model!r}")
    ds = load_dataset("mteb/results", model, trust_remote_code=True)
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

# Save & cache EXTERNAL_MODEL_RESULTS
with open("EXTERNAL_MODEL_RESULTS.json", "w") as f:
    json.dump(EXTERNAL_MODEL_RESULTS, f)

def get_dim_seq_size(model):
    filenames = [sib.rfilename for sib in model.siblings]
    dim, seq = "", ""
    for filename in filenames:
        if re.match("\d+_Pooling/config.json", filename):
            st_config_path = hf_hub_download(model.modelId, filename=filename)
            dim = json.load(open(st_config_path)).get("word_embedding_dimension", "")
            break
    for filename in filenames:
        if re.match("\d+_Dense/config.json", filename):
            st_config_path = hf_hub_download(model.modelId, filename=filename)
            dim = json.load(open(st_config_path)).get("out_features", dim)
    if "config.json" in filenames:
        config_path = hf_hub_download(model.modelId, filename="config.json")
        config = json.load(open(config_path))
        if not dim:
            dim = config.get("hidden_dim", config.get("hidden_size", config.get("d_model", "")))
        seq = config.get("n_positions", config.get("max_position_embeddings", config.get("n_ctx", config.get("seq_length", ""))))
    # Get model file size without downloading. Parameters in million parameters and memory in GB
    parameters, memory = get_model_parameters_memory(model)
    return dim, seq, parameters, memory

def make_datasets_clickable(df):
    """Does not work"""
    if "BornholmBitextMining" in df.columns:
        link = "https://huggingface.co/datasets/strombergnlp/bornholmsk_parallel"
        df = df.rename(
            columns={f'BornholmBitextMining': '<a target="_blank" style="text-decoration: underline" href="{link}">BornholmBitextMining</a>',})
    return df

def add_rank(df):
    cols_to_rank = [col for col in df.columns if col not in ["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens"]]
    if len(cols_to_rank) == 1:
        df.sort_values(cols_to_rank[0], ascending=False, inplace=True)
    else:
        df.insert(len(df.columns) - len(cols_to_rank), "Average", df[cols_to_rank].mean(axis=1, skipna=False))
        df.sort_values("Average", ascending=False, inplace=True)
    df.insert(0, "Rank", list(range(1, len(df) + 1)))
    df = df.round(2)
    # Fill NaN after averaging
    df.fillna("", inplace=True)
    return df

def get_mteb_data(tasks=["Clustering"], langs=[], datasets=[], fillna=True, add_emb_dim=True, task_to_metric=TASK_TO_METRIC, rank=True):
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
                res["Model Size (Million Parameters)"] = EXTERNAL_MODEL_TO_SIZE.get(model, "")
                res["Memory Usage (GB, fp32)"] = round(res["Model Size (Million Parameters)"] * 1e6 * 4 / 1024**3, 2) if res["Model Size (Million Parameters)"] != "" else ""
                res["Embedding Dimensions"] = EXTERNAL_MODEL_TO_DIM.get(model, "")
                res["Max Tokens"] = EXTERNAL_MODEL_TO_SEQLEN.get(model, "")
            df_list.append(res)
    
    for model in models:
        if model.modelId in MODELS_TO_SKIP: continue
        print("MODEL", model)
        readme_path = hf_hub_download(model.modelId, filename="README.md")
        meta = metadata_load(readme_path)
        if "model-index" not in meta:
            continue
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
                try:
                    # Fails on gated repos, so we only include scores for them
                    out["Embedding Dimensions"], out["Max Tokens"], out["Model Size (Million Parameters)"], out["Memory Usage (GB, fp32)"] = get_dim_seq_size(model)
                except:
                    pass
            df_list.append(out)
        if model.library_name == "sentence-transformers" or "sentence-transformers" in model.tags or "modules.json" in {file.rfilename for file in model.siblings}:
            SENTENCE_TRANSFORMERS_COMPATIBLE_MODELS.add(out["Model"])
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
    global DATA_OVERALL, DATA_CLASSIFICATION_EN, DATA_CLUSTERING, DATA_PAIR_CLASSIFICATION, DATA_RERANKING, DATA_RETRIEVAL, DATA_STS_EN, DATA_SUMMARIZATION
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
        datasets=TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS + TASK_LIST_SUMMARIZATION,
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

    DATA_CLASSIFICATION_EN = add_rank(DATA_OVERALL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] + TASK_LIST_CLASSIFICATION])
    # Only keep rows with at least one score in addition to the "Model" & rank column
    DATA_CLASSIFICATION_EN = DATA_CLASSIFICATION_EN[DATA_CLASSIFICATION_EN.iloc[:, 4:].ne("").any(axis=1)]

    DATA_CLUSTERING = add_rank(DATA_OVERALL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_CLUSTERING])
    DATA_CLUSTERING = DATA_CLUSTERING[DATA_CLUSTERING.iloc[:, 4:].ne("").any(axis=1)]

    DATA_PAIR_CLASSIFICATION = add_rank(DATA_OVERALL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_PAIR_CLASSIFICATION])
    DATA_PAIR_CLASSIFICATION = DATA_PAIR_CLASSIFICATION[DATA_PAIR_CLASSIFICATION.iloc[:, 4:].ne("").any(axis=1)]

    DATA_RERANKING = add_rank(DATA_OVERALL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_RERANKING])
    DATA_RERANKING = DATA_RERANKING[DATA_RERANKING.iloc[:, 4:].ne("").any(axis=1)]

    DATA_RETRIEVAL = add_rank(DATA_OVERALL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_RETRIEVAL])
    DATA_RETRIEVAL = DATA_RETRIEVAL[DATA_RETRIEVAL.iloc[:, 4:].ne("").any(axis=1)]

    DATA_STS_EN = add_rank(DATA_OVERALL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_STS])
    DATA_STS_EN = DATA_STS_EN[DATA_STS_EN.iloc[:, 4:].ne("").any(axis=1)]

    DATA_SUMMARIZATION = add_rank(DATA_OVERALL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_SUMMARIZATION])
    DATA_SUMMARIZATION = DATA_SUMMARIZATION[DATA_SUMMARIZATION.iloc[:, 1:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL.fillna("", inplace=True)

    DATA_OVERALL = DATA_OVERALL[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens", f"Average ({len(TASK_LIST_EN)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION)} datasets)", f"Reranking Average ({len(TASK_LIST_RERANKING)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)", f"STS Average ({len(TASK_LIST_STS)} datasets)", f"Summarization Average ({len(TASK_LIST_SUMMARIZATION)} dataset)"]]
    DATA_OVERALL = DATA_OVERALL[DATA_OVERALL.iloc[:, 5:].ne("").any(axis=1)]

    return DATA_OVERALL

def get_mteb_average_zh():
    global DATA_OVERALL_ZH, DATA_CLASSIFICATION_ZH, DATA_CLUSTERING_ZH, DATA_PAIR_CLASSIFICATION_ZH, DATA_RERANKING_ZH, DATA_RETRIEVAL_ZH, DATA_STS_ZH
    DATA_OVERALL_ZH = get_mteb_data(
        tasks=[
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
            "STS",
        ],
        datasets=TASK_LIST_CLASSIFICATION_ZH + TASK_LIST_CLUSTERING_ZH + TASK_LIST_PAIR_CLASSIFICATION_ZH + TASK_LIST_RERANKING_ZH + TASK_LIST_RETRIEVAL_ZH + TASK_LIST_STS_ZH,
        fillna=False,
        add_emb_dim=True,
        rank=False,
    )
    # Debugging:
    # DATA_OVERALL_ZH.to_csv("overall.csv")
    
    DATA_OVERALL_ZH.insert(1, f"Average ({len(TASK_LIST_ZH)} datasets)", DATA_OVERALL_ZH[TASK_LIST_ZH].mean(axis=1, skipna=False))
    DATA_OVERALL_ZH.insert(2, f"Classification Average ({len(TASK_LIST_CLASSIFICATION_ZH)} datasets)", DATA_OVERALL_ZH[TASK_LIST_CLASSIFICATION_ZH].mean(axis=1, skipna=False))
    DATA_OVERALL_ZH.insert(3, f"Clustering Average ({len(TASK_LIST_CLUSTERING_ZH)} datasets)", DATA_OVERALL_ZH[TASK_LIST_CLUSTERING_ZH].mean(axis=1, skipna=False))
    DATA_OVERALL_ZH.insert(4, f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_ZH)} datasets)", DATA_OVERALL_ZH[TASK_LIST_PAIR_CLASSIFICATION_ZH].mean(axis=1, skipna=False))
    DATA_OVERALL_ZH.insert(5, f"Reranking Average ({len(TASK_LIST_RERANKING_ZH)} datasets)", DATA_OVERALL_ZH[TASK_LIST_RERANKING_ZH].mean(axis=1, skipna=False))
    DATA_OVERALL_ZH.insert(6, f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_ZH)} datasets)", DATA_OVERALL_ZH[TASK_LIST_RETRIEVAL_ZH].mean(axis=1, skipna=False))
    DATA_OVERALL_ZH.insert(7, f"STS Average ({len(TASK_LIST_STS_ZH)} datasets)", DATA_OVERALL_ZH[TASK_LIST_STS_ZH].mean(axis=1, skipna=False))
    DATA_OVERALL_ZH.sort_values(f"Average ({len(TASK_LIST_ZH)} datasets)", ascending=False, inplace=True)
    # Start ranking from 1
    DATA_OVERALL_ZH.insert(0, "Rank", list(range(1, len(DATA_OVERALL_ZH) + 1)))

    DATA_OVERALL_ZH = DATA_OVERALL_ZH.round(2)

    DATA_CLASSIFICATION_ZH = add_rank(DATA_OVERALL_ZH[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_CLASSIFICATION_ZH])
    # Only keep rows with at least one score in addition to the "Model" & rank column
    DATA_CLASSIFICATION_ZH = DATA_CLASSIFICATION_ZH[DATA_CLASSIFICATION_ZH.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_CLUSTERING_ZH = add_rank(DATA_OVERALL_ZH[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_CLUSTERING_ZH])
    DATA_CLUSTERING_ZH = DATA_CLUSTERING_ZH[DATA_CLUSTERING_ZH.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_PAIR_CLASSIFICATION_ZH = add_rank(DATA_OVERALL_ZH[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_PAIR_CLASSIFICATION_ZH])
    DATA_PAIR_CLASSIFICATION_ZH = DATA_PAIR_CLASSIFICATION_ZH[DATA_PAIR_CLASSIFICATION_ZH.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_RERANKING_ZH = add_rank(DATA_OVERALL_ZH[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_RERANKING_ZH])
    DATA_RERANKING_ZH = DATA_RERANKING_ZH[DATA_RERANKING_ZH.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_RETRIEVAL_ZH = add_rank(DATA_OVERALL_ZH[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_RETRIEVAL_ZH])
    DATA_RETRIEVAL_ZH = DATA_RETRIEVAL_ZH[DATA_RETRIEVAL_ZH.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_STS_ZH = add_rank(DATA_OVERALL_ZH[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_STS_ZH])
    DATA_STS_ZH = DATA_STS_ZH[DATA_STS_ZH.iloc[:, 4:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL_ZH.fillna("", inplace=True)

    DATA_OVERALL_ZH = DATA_OVERALL_ZH[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens", f"Average ({len(TASK_LIST_ZH)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION_ZH)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING_ZH)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_ZH)} datasets)", f"Reranking Average ({len(TASK_LIST_RERANKING_ZH)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_ZH)} datasets)", f"STS Average ({len(TASK_LIST_STS_ZH)} datasets)"]]
    DATA_OVERALL_ZH = DATA_OVERALL_ZH[DATA_OVERALL_ZH.iloc[:, 5:].ne("").any(axis=1)]

    return DATA_OVERALL_ZH

def get_mteb_average_fr():
    global DATA_OVERALL_FR, DATA_CLASSIFICATION_FR, DATA_CLUSTERING_FR, DATA_PAIR_CLASSIFICATION_FR, DATA_RERANKING_FR, DATA_RETRIEVAL_FR, DATA_STS_FR, DATA_SUMMARIZATION_FR
    DATA_OVERALL_FR = get_mteb_data(
        tasks=[
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
            "STS",
            "Summarization"
        ],
        datasets=TASK_LIST_CLASSIFICATION_FR + TASK_LIST_CLUSTERING_FR + TASK_LIST_PAIR_CLASSIFICATION_FR + TASK_LIST_RERANKING_FR + TASK_LIST_RETRIEVAL_FR + TASK_LIST_STS_FR + TASK_LIST_SUMMARIZATION_FR,
        fillna=False,
        add_emb_dim=True,
        rank=False,
    )
    # Debugging:
    # DATA_OVERALL_FR.to_csv("overall.csv")
    
    DATA_OVERALL_FR.insert(1, f"Average ({len(TASK_LIST_FR)} datasets)", DATA_OVERALL_FR[TASK_LIST_FR].mean(axis=1, skipna=False))
    DATA_OVERALL_FR.insert(2, f"Classification Average ({len(TASK_LIST_CLASSIFICATION_FR)} datasets)", DATA_OVERALL_FR[TASK_LIST_CLASSIFICATION_FR].mean(axis=1, skipna=False))
    DATA_OVERALL_FR.insert(3, f"Clustering Average ({len(TASK_LIST_CLUSTERING_FR)} datasets)", DATA_OVERALL_FR[TASK_LIST_CLUSTERING_FR].mean(axis=1, skipna=False))
    DATA_OVERALL_FR.insert(4, f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_FR)} datasets)", DATA_OVERALL_FR[TASK_LIST_PAIR_CLASSIFICATION_FR].mean(axis=1, skipna=False))
    DATA_OVERALL_FR.insert(5, f"Reranking Average ({len(TASK_LIST_RERANKING_FR)} datasets)", DATA_OVERALL_FR[TASK_LIST_RERANKING_FR].mean(axis=1, skipna=False))
    DATA_OVERALL_FR.insert(6, f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_FR)} datasets)", DATA_OVERALL_FR[TASK_LIST_RETRIEVAL_FR].mean(axis=1, skipna=False))
    DATA_OVERALL_FR.insert(7, f"STS Average ({len(TASK_LIST_STS_FR)} datasets)", DATA_OVERALL_FR[TASK_LIST_STS_FR].mean(axis=1, skipna=False))
    DATA_OVERALL_FR.insert(8, f"Summarization Average ({len(TASK_LIST_SUMMARIZATION_FR)} dataset)", DATA_OVERALL_FR[TASK_LIST_SUMMARIZATION_FR].mean(axis=1, skipna=False))
    DATA_OVERALL_FR.sort_values(f"Average ({len(TASK_LIST_FR)} datasets)", ascending=False, inplace=True)
    # Start ranking from 1
    DATA_OVERALL_FR.insert(0, "Rank", list(range(1, len(DATA_OVERALL_FR) + 1)))
    DATA_OVERALL_FR = DATA_OVERALL_FR.round(2)

    DATA_CLASSIFICATION_FR = add_rank(DATA_OVERALL_FR[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_CLASSIFICATION_FR])
    DATA_CLASSIFICATION_FR = DATA_CLASSIFICATION_FR[DATA_CLASSIFICATION_FR.iloc[:, 4:].ne("").any(axis=1)]

    DATA_CLUSTERING_FR = add_rank(DATA_OVERALL_FR[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_CLUSTERING_FR])
    DATA_CLUSTERING_FR = DATA_CLUSTERING_FR[DATA_CLUSTERING_FR.iloc[:, 4:].ne("").any(axis=1)]

    DATA_PAIR_CLASSIFICATION_FR = add_rank(DATA_OVERALL_FR[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_PAIR_CLASSIFICATION_FR])
    DATA_PAIR_CLASSIFICATION_FR = DATA_PAIR_CLASSIFICATION_FR[DATA_PAIR_CLASSIFICATION_FR.iloc[:, 4:].ne("").any(axis=1)]

    DATA_RERANKING_FR = add_rank(DATA_OVERALL_FR[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_RERANKING_FR])
    DATA_RERANKING_FR = DATA_RERANKING_FR[DATA_RERANKING_FR.iloc[:, 4:].ne("").any(axis=1)]

    DATA_RETRIEVAL_FR = add_rank(DATA_OVERALL_FR[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_RETRIEVAL_FR])
    DATA_RETRIEVAL_FR = DATA_RETRIEVAL_FR[DATA_RETRIEVAL_FR.iloc[:, 4:].ne("").any(axis=1)]

    DATA_STS_FR = add_rank(DATA_OVERALL_FR[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_STS_FR])
    DATA_STS_FR = DATA_STS_FR[DATA_STS_FR.iloc[:, 4:].ne("").any(axis=1)]

    DATA_SUMMARIZATION_FR = add_rank(DATA_OVERALL_FR[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_SUMMARIZATION_FR])
    DATA_SUMMARIZATION_FR = DATA_SUMMARIZATION_FR[DATA_SUMMARIZATION_FR.iloc[:, 1:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL_FR.fillna("", inplace=True)

    DATA_OVERALL_FR = DATA_OVERALL_FR[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens", f"Average ({len(TASK_LIST_FR)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION_FR)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING_FR)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_FR)} datasets)", f"Reranking Average ({len(TASK_LIST_RERANKING_FR)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_FR)} datasets)", f"STS Average ({len(TASK_LIST_STS_FR)} datasets)", f"Summarization Average ({len(TASK_LIST_SUMMARIZATION_FR)} dataset)"]]
    DATA_OVERALL_FR = DATA_OVERALL_FR[DATA_OVERALL_FR.iloc[:, 5:].ne("").any(axis=1)]

    return DATA_OVERALL_FR

def get_mteb_average_pl():
    global DATA_OVERALL_PL, DATA_CLASSIFICATION_PL, DATA_CLUSTERING_PL, DATA_PAIR_CLASSIFICATION_PL, DATA_RETRIEVAL_PL, DATA_STS_PL
    DATA_OVERALL_PL = get_mteb_data(
        tasks=[
            "Classification",
            "Clustering",
            "PairClassification",
            "Retrieval",
            "STS",
        ],
        datasets=TASK_LIST_CLASSIFICATION_PL + TASK_LIST_CLUSTERING_PL + TASK_LIST_PAIR_CLASSIFICATION_PL + TASK_LIST_RETRIEVAL_PL + TASK_LIST_STS_PL,
        fillna=False,
        add_emb_dim=True,
        rank=False,
    )
    # Debugging:
    # DATA_OVERALL_PL.to_csv("overall.csv")
    
    DATA_OVERALL_PL.insert(1, f"Average ({len(TASK_LIST_PL)} datasets)", DATA_OVERALL_PL[TASK_LIST_PL].mean(axis=1, skipna=False))
    DATA_OVERALL_PL.insert(2, f"Classification Average ({len(TASK_LIST_CLASSIFICATION_PL)} datasets)", DATA_OVERALL_PL[TASK_LIST_CLASSIFICATION_PL].mean(axis=1, skipna=False))
    DATA_OVERALL_PL.insert(3, f"Clustering Average ({len(TASK_LIST_CLUSTERING_PL)} datasets)", DATA_OVERALL_PL[TASK_LIST_CLUSTERING_PL].mean(axis=1, skipna=False))
    DATA_OVERALL_PL.insert(4, f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_PL)} datasets)", DATA_OVERALL_PL[TASK_LIST_PAIR_CLASSIFICATION_PL].mean(axis=1, skipna=False))
    DATA_OVERALL_PL.insert(5, f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_PL)} datasets)", DATA_OVERALL_PL[TASK_LIST_RETRIEVAL_PL].mean(axis=1, skipna=False))
    DATA_OVERALL_PL.insert(6, f"STS Average ({len(TASK_LIST_STS_PL)} datasets)", DATA_OVERALL_PL[TASK_LIST_STS_PL].mean(axis=1, skipna=False))
    DATA_OVERALL_PL.sort_values(f"Average ({len(TASK_LIST_PL)} datasets)", ascending=False, inplace=True)
    # Start ranking from 1
    DATA_OVERALL_PL.insert(0, "Rank", list(range(1, len(DATA_OVERALL_PL) + 1)))

    DATA_OVERALL_PL = DATA_OVERALL_PL.round(2)

    DATA_CLASSIFICATION_PL = add_rank(DATA_OVERALL_PL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_CLASSIFICATION_PL])
    # Only keep rows with at least one score in addition to the "Model" & rank column
    DATA_CLASSIFICATION_PL = DATA_CLASSIFICATION_PL[DATA_CLASSIFICATION_PL.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_CLUSTERING_PL = add_rank(DATA_OVERALL_PL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_CLUSTERING_PL])
    DATA_CLUSTERING_PL = DATA_CLUSTERING_PL[DATA_CLUSTERING_PL.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_PAIR_CLASSIFICATION_PL = add_rank(DATA_OVERALL_PL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_PAIR_CLASSIFICATION_PL])
    DATA_PAIR_CLASSIFICATION_PL = DATA_PAIR_CLASSIFICATION_PL[DATA_PAIR_CLASSIFICATION_PL.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_RETRIEVAL_PL = add_rank(DATA_OVERALL_PL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_RETRIEVAL_PL])
    DATA_RETRIEVAL_PL = DATA_RETRIEVAL_PL[DATA_RETRIEVAL_PL.iloc[:, 4:].ne("").any(axis=1)]
    
    DATA_STS_PL = add_rank(DATA_OVERALL_PL[["Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] +  TASK_LIST_STS_PL])
    DATA_STS_PL = DATA_STS_PL[DATA_STS_PL.iloc[:, 4:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL_PL.fillna("", inplace=True)

    DATA_OVERALL_PL = DATA_OVERALL_PL[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Embedding Dimensions", "Max Tokens", f"Average ({len(TASK_LIST_PL)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION_PL)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING_PL)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_PL)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_PL)} datasets)", f"STS Average ({len(TASK_LIST_STS_PL)} datasets)"]]
    DATA_OVERALL_PL = DATA_OVERALL_PL[DATA_OVERALL_PL.iloc[:, 5:].ne("").any(axis=1)]

    return DATA_OVERALL_PL

get_mteb_average()
get_mteb_average_fr()
get_mteb_average_pl()
get_mteb_average_zh()
DATA_BITEXT_MINING = get_mteb_data(["BitextMining"], [], TASK_LIST_BITEXT_MINING)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Average"] + TASK_LIST_BITEXT_MINING]
DATA_BITEXT_MINING_DA = get_mteb_data(["BitextMining"], [], TASK_LIST_BITEXT_MINING_DA)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)"] + TASK_LIST_BITEXT_MINING_DA]
DATA_CLASSIFICATION_DA = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_DA)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Average"] + TASK_LIST_CLASSIFICATION_DA]
DATA_CLASSIFICATION_NB = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_NB)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Average"] + TASK_LIST_CLASSIFICATION_NB]
DATA_CLASSIFICATION_SV = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_SV)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Average"] + TASK_LIST_CLASSIFICATION_SV]
DATA_CLASSIFICATION_OTHER = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_OTHER)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Average"] + TASK_LIST_CLASSIFICATION_OTHER]
DATA_CLUSTERING_DE = get_mteb_data(["Clustering"], [], TASK_LIST_CLUSTERING_DE)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Average"] + TASK_LIST_CLUSTERING_DE]
DATA_STS_OTHER = get_mteb_data(["STS"], [], TASK_LIST_STS_OTHER)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Average"] + TASK_LIST_STS_OTHER]
DATA_RETRIEVAL_LAW = get_mteb_data(["Retrieval"], [], TASK_LIST_RETRIEVAL_LAW)[["Rank", "Model", "Model Size (Million Parameters)", "Memory Usage (GB, fp32)", "Average"] + TASK_LIST_RETRIEVAL_LAW]

# Exact, add all non-nan integer values for every dataset
NUM_SCORES = 0
DATASETS = []
MODELS = []
# LANGUAGES = []
for d in [
    DATA_BITEXT_MINING,
    DATA_BITEXT_MINING_DA,
    DATA_CLASSIFICATION_EN,
    DATA_CLASSIFICATION_DA,
    DATA_CLASSIFICATION_FR,
    DATA_CLASSIFICATION_NB,
    DATA_CLASSIFICATION_PL,
    DATA_CLASSIFICATION_SV,
    DATA_CLASSIFICATION_ZH,
    DATA_CLASSIFICATION_OTHER,
    DATA_CLUSTERING,
    DATA_CLUSTERING_DE,
    DATA_CLUSTERING_FR,
    DATA_CLUSTERING_PL,
    DATA_CLUSTERING_ZH,
    DATA_PAIR_CLASSIFICATION,
    DATA_PAIR_CLASSIFICATION_FR,
    DATA_PAIR_CLASSIFICATION_PL,
    DATA_PAIR_CLASSIFICATION_ZH,
    DATA_RERANKING,
    DATA_RERANKING_FR,
    DATA_RERANKING_ZH,
    DATA_RETRIEVAL,
    DATA_RETRIEVAL_FR,
    DATA_RETRIEVAL_PL,
    DATA_RETRIEVAL_ZH,
    DATA_RETRIEVAL_LAW,
    DATA_STS_EN,
    DATA_STS_FR,
    DATA_STS_PL,
    DATA_STS_ZH,
    DATA_STS_OTHER,
    DATA_SUMMARIZATION,
    DATA_SUMMARIZATION_FR,
]:
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
- refresh: The function to refresh the leaderboard
"""

chinese_credits = "[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)"
french_credits = "[Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [Wissam Siblini](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)"
danish_credits = "[Kenneth Enevoldsen](https://github.com/KennethEnevoldsen), [scandinavian-embedding-benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)"
norwegian_credits = "[Kenneth Enevoldsen](https://github.com/KennethEnevoldsen), [scandinavian-embedding-benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)"
polish_credits = "[Rafał Poświata](https://github.com/rafalposwiata)"

data = {
    "Overall": {
        "metric": "Various, refer to task tabs",
        "data": [
            {
                "language": "English",
                "description": "**Overall MTEB English leaderboard** 🔮",
                "data": DATA_OVERALL,
                "refresh": get_mteb_average,
            },
            {
                "language": "Chinese",
                "data": DATA_OVERALL_ZH,
                "description": "**Overall MTEB Chinese leaderboard (C-MTEB)** 🔮🇨🇳",
                "credits": chinese_credits,
                "refresh": get_mteb_average_zh,
            },
            {
                "language": "French",
                "data": DATA_OVERALL_FR,
                "description": "**Overall MTEB French leaderboard (F-MTEB)** 🔮🇫🇷",
                "credits": french_credits,
                "refresh": get_mteb_average_fr,
            },
            {
                "language": "Polish",
                "data": DATA_OVERALL_PL,
                "description": "**Overall MTEB Polish leaderboard** 🔮🇵🇱",
                "refresh": get_mteb_average_pl,
            },
        ]
    },
    "Bitext Mining": {
        "metric": "[F1](https://huggingface.co/spaces/evaluate-metric/f1)",
        "data": [
            {
                "language": "English-X",
                "language_long": "117 (Pairs of: English & other language)",
                "description": "**Bitext Mining English-X Leaderboard** 🎌",
                "data": DATA_BITEXT_MINING,
                "refresh": partial(get_mteb_data, tasks=["BitextMining"], datasets=TASK_LIST_BITEXT_MINING),
            },
            {
                "language": "Danish",
                "language_long": "Danish & Bornholmsk (Danish Dialect)",
                "description": "**Bitext Mining Danish Leaderboard** 🎌🇩🇰",
                "credits": danish_credits,
                "data": DATA_BITEXT_MINING_DA,
                "refresh": partial(get_mteb_data, tasks=["BitextMining"], datasets=TASK_LIST_BITEXT_MINING_DA),
            }
        ]
    },
    "Classification": {
        "metric": "[Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)",
        "data": [
            {
                "language": "English",
                "description": "**Classification English Leaderboard** ❤️",
                "data": DATA_CLASSIFICATION_EN,
                "refresh": partial(get_mteb_data, tasks=["Classification"], langs=["en"])
            },
            {
                "language": "Chinese",
                "description": "**Classification Chinese Leaderboard** 🧡🇨🇳",
                "credits": chinese_credits,
                "data": DATA_CLASSIFICATION_ZH,
                "refresh": partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_ZH)
            },
            {
                "language": "Danish",
                "description": "**Classification Danish Leaderboard** 🤍🇩🇰",
                "credits": danish_credits,
                "data": DATA_CLASSIFICATION_DA,
                "refresh": partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_DA)
            },
            {
                "language": "French",
                "description": "**Classification French Leaderboard** 💙🇫🇷",
                "credits": french_credits,
                "data": DATA_CLASSIFICATION_FR,
                "refresh": partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_FR)
            },
            {
                "language": "Norwegian",
                "language_long": "Norwegian Bokmål",
                "description": "**Classification Norwegian Leaderboard** 💙🇳🇴",
                "credits": norwegian_credits,
                "data": DATA_CLASSIFICATION_NB,
                "refresh": partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_NB)
            },
            {
                "language": "Polish",
                "description": "**Classification Polish Leaderboard** 🤍🇵🇱",
                "credits": polish_credits,
                "data": DATA_CLASSIFICATION_PL,
                "refresh": partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_PL)
            },
            {
                "language": "Swedish",
                "description": "**Classification Swedish Leaderboard** 💛🇸🇪",
                "credits": norwegian_credits,
                "data": DATA_CLASSIFICATION_SV,
                "refresh": partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_SV)
            },
            {
                "language": "Other",
                "language_long": "47 (Only languages not included in the other tabs)",
                "description": "**Classification Other Languages Leaderboard** 💜💚💙",
                "data": DATA_CLASSIFICATION_OTHER,
                "refresh": partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_OTHER)
            }
        ]
    },
    "Clustering": {
        "metric": "Validity Measure (v_measure)",
        "data": [
            {
                "language": "English",
                "description": "**Clustering Leaderboard** ✨",
                "data": DATA_CLUSTERING,
                "refresh": partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING)
            },
            {
                "language": "Chinese",
                "description": "**Clustering Chinese Leaderboard** ✨🇨🇳",
                "credits": chinese_credits,
                "data": DATA_CLUSTERING_ZH,
                "refresh": partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING_ZH)
            },
            {
                "language": "French",
                "description": "**Clustering French Leaderboard** ✨🇫🇷",
                "credits": french_credits,
                "data": DATA_CLUSTERING_FR,
                "refresh": partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING_FR)
            },
            {
                "language": "German",
                "description": "**Clustering German Leaderboard** ✨🇩🇪",
                "credits": "[Silvan](https://github.com/slvnwhrl)",
                "data": DATA_CLUSTERING_DE,
                "refresh": partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING_DE)
            },
            {
                "language": "Polish",
                "description": "**Clustering Polish Leaderboard** ✨🇵🇱",
                "credits": polish_credits,
                "data": DATA_CLUSTERING_PL,
                "refresh": partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING_PL)
            },
        ]
    },
    "Pair Classification": {
        "metric": "Average Precision based on Cosine Similarities (cos_sim_ap)",
        "data": [
            {
                "language": "English",
                "description": "**Pair Classification English Leaderboard** 🎭",
                "data": DATA_PAIR_CLASSIFICATION,
                "refresh": partial(get_mteb_data, tasks=["PairClassification"], datasets=TASK_LIST_PAIR_CLASSIFICATION)
            },
            {
                "language": "Chinese",
                "description": "**Pair Classification Chinese Leaderboard** 🎭🇨🇳",
                "credits": chinese_credits,
                "data": DATA_PAIR_CLASSIFICATION_ZH,
                "refresh": partial(get_mteb_data, tasks=["PairClassification"], datasets=TASK_LIST_PAIR_CLASSIFICATION_ZH)
            },
            {
                "language": "French",
                "description": "**Pair Classification French Leaderboard** 🎭🇫🇷",
                "credits": french_credits,
                "data": DATA_PAIR_CLASSIFICATION_FR,
                "refresh": partial(get_mteb_data, tasks=["PairClassification"], datasets=TASK_LIST_PAIR_CLASSIFICATION_FR)
            },
            {
                "language": "Polish",
                "description": "**Pair Classification Polish Leaderboard** 🎭🇵🇱",
                "credits": polish_credits,
                "data": DATA_PAIR_CLASSIFICATION_PL,
                "refresh": partial(get_mteb_data, tasks=["PairClassification"], datasets=TASK_LIST_PAIR_CLASSIFICATION_PL)
            },
        ]
    },
    "Reranking": {
        "metric": "Mean Average Precision (MAP)",
        "data": [
            {
                "language": "English",
                "description": "**Reranking English Leaderboard** 🥈",
                "data": DATA_RERANKING,
                "refresh": partial(get_mteb_data, tasks=["Reranking"], datasets=TASK_LIST_RERANKING)
            },
            {
                "language": "Chinese",
                "description": "**Reranking Chinese Leaderboard** 🥈🇨🇳",
                "credits": chinese_credits,
                "data": DATA_RERANKING_ZH,
                "refresh": partial(get_mteb_data, tasks=["Reranking"], datasets=TASK_LIST_RERANKING_ZH)
            },
            {
                "language": "French",
                "description": "**Reranking French Leaderboard** 🥈🇫🇷",
                "credits": french_credits,
                "data": DATA_RERANKING_FR,
                "refresh": partial(get_mteb_data, tasks=["Reranking"], datasets=TASK_LIST_RERANKING_FR)
            }
        ]
    },
    "Retrieval": {
        "metric": "Normalized Discounted Cumulative Gain @ k (ndcg_at_10)",
        "data": [
            {
                "language": "English",
                "description": "**Retrieval English Leaderboard** 🔎",
                "data": DATA_RETRIEVAL,
                "refresh": partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL)
            },
            {
                "language": "Chinese",
                "description": "**Retrieval Chinese Leaderboard** 🔎🇨🇳",
                "credits": chinese_credits,
                "data": DATA_RETRIEVAL_ZH,
                "refresh": partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL_ZH)
            },
            {
                "language": "French",
                "description": "**Retrieval French Leaderboard** 🔎🇫🇷",
                "credits": french_credits,
                "data": DATA_RETRIEVAL_FR,
                "refresh": partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL_FR)
            },
            {
                "language": "Law",
                "language_long": "English, German, Chinese",
                "description": "**Retrieval Law Leaderboard** 🔎⚖️",
                "credits": "[Voyage AI](https://www.voyageai.com/)",
                "data": DATA_RETRIEVAL_LAW,
                "refresh": partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL_LAW)
            },
            {
                "language": "Polish",
                "description": "**Retrieval Polish Leaderboard** 🔎🇵🇱",
                "credits": polish_credits,
                "data": DATA_RETRIEVAL_PL,
                "refresh": partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL_PL)
            }
        ]
    },
    "STS": {
        "metric": "Spearman correlation based on cosine similarity",
        "data": [
            {
                "language": "English",
                "description": "**STS English Leaderboard** 🤖",
                "data": DATA_STS_EN,
                "refresh": partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS)
            },
            {
                "language": "Chinese",
                "description": "**STS Chinese Leaderboard** 🤖🇨🇳",
                "credits": chinese_credits,
                "data": DATA_STS_ZH,
                "refresh": partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS_ZH)
            },
            {
                "language": "French",
                "description": "**STS French Leaderboard** 🤖🇫🇷",
                "credits": french_credits,
                "data": DATA_STS_FR,
                "refresh": partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS_FR)
            },
            {
                "language": "Polish",
                "description": "**STS Polish Leaderboard** 🤖🇵🇱",
                "credits": polish_credits,
                "data": DATA_STS_PL,
                "refresh": partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS_PL)
            },
            {
                "language": "Other",
                "language_long": "Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Russian, Spanish (Only language combos not included in the other tabs)",
                "description": "**STS Other Leaderboard** 👽",
                "data": DATA_STS_OTHER,
                "refresh": partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS_OTHER)
            },
        ]
    },
    "Summarization": {
        "metric": "Spearman correlation	based on cosine similarity",
        "data": [
            {
                "language": "English",
                "description": "**Summarization Leaderboard** 📜",
                "data": DATA_SUMMARIZATION,
                "refresh": partial(get_mteb_data, tasks=TASK_LIST_SUMMARIZATION)
            },
            {
                "language": "French",
                "description": "**Summarization Leaderboard** 📜",
                "credits": french_credits,
                "data": DATA_SUMMARIZATION_FR,
                "refresh": partial(get_mteb_data, tasks=TASK_LIST_SUMMARIZATION_FR)
            }
        ]
    }
}

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
]

def filter_data(search_query, model_types, model_sizes, *full_dataframes):
    output_dataframes = []
    for df in full_dataframes:
        # Apply the search query
        if search_query:
            names = df["Model"].map(lambda x: re.match("<a .+?>(.+)</a>", x).group(1))
            masks = []
            for query in search_query.split(";"):
                masks.append(names.str.contains(query))
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
            with gr.Tab(task, id=task_tab_id) as task_tab:
                # For updating the 'task' in the URL
                task_tab.select(update_url_task, [current_task_language, language_per_task], [current_task_language, language_per_task]).then(None, [current_task_language], [], js=set_window_url_params)

                with gr.Tabs() as task_tabs:
                    # Store the task tabs for updating them on load based on URL parameters
                    tabs.append(task_tabs)

                    for item in task_values["data"]:
                        item_tab_id = item["language"].lower().replace(" ", "-")

                        # English, Chinese, French, etc.
                        with gr.Tab(item["language"], id=item_tab_id) as item_tab:
                            # For updating the 'language' in the URL
                            item_tab.select(update_url_language, [current_task_language, language_per_task], [current_task_language, language_per_task], trigger_mode="always_last").then(None, [current_task_language], [], js=set_window_url_params)

                            with gr.Row():
                                gr.Markdown(f"""
                                {item['description']}

                                - **Metric:** {metric}
                                - **Languages:** {item['language_long'] if 'language_long' in item else item['language']}
                                {"- **Credits:** " + item['credits'] if "credits" in item else ''}
                                """)

                            with gr.Row():
                                datatype = ["number", "markdown"] + ["number"] * len(item["data"])
                                dataframe = gr.Dataframe(item["data"], datatype=datatype, type="pandas", height=500)
                                dataframes.append(dataframe)

                                full_dataframe = gr.Dataframe(item["data"], datatype=datatype, type="pandas", visible=False)
                                full_dataframes.append(full_dataframe)

                            with gr.Row():
                                refresh_button = gr.Button("Refresh")
                                refresh_button.click(item["refresh"], inputs=None, outputs=dataframe, concurrency_limit=20)

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

# Possible changes:
# Could add graphs / other visual content
# Could add verification marks

# Sources:
# https://huggingface.co/spaces/gradio/leaderboard
# https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard
# https://getemoji.com/
