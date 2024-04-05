from functools import partial
import json
import os

from datasets import load_dataset
import gradio as gr
from huggingface_hub import get_hf_file_metadata, HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.repocard import metadata_load
import pandas as pd
from tqdm.autonotebook import tqdm

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
    "voyage-lite-01-instruct",
    "voyage-lite-02-instruct",
    "xlm-roberta-base",
    "xlm-roberta-large",
]

EXTERNAL_MODEL_TO_LINK = {
    "Cohere-embed-english-v3.0": "https://huggingface.co/Cohere/Cohere-embed-english-v3.0",
    "Cohere-embed-multilingual-v3.0": "https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0",
    "Cohere-embed-multilingual-light-v3.0": "https://huggingface.co/Cohere/Cohere-embed-multilingual-light-v3.0",
    "allenai-specter": "https://huggingface.co/sentence-transformers/allenai-specter",
    "allenai-specter": "https://huggingface.co/sentence-transformers/allenai-specter",
    "all-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    "Baichuan-text-embedding": "https://platform.baichuan-ai.com/docs/text-Embedding",
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
    "DanskBERT": "https://huggingface.co/vesteinn/DanskBERT",
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
    "LASER2": "https://github.com/facebookresearch/LASER",
    "LaBSE": "https://huggingface.co/sentence-transformers/LaBSE",
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
    "OpenSearch-text-hybrid": "https://help.aliyun.com/zh/open-search/vector-search-edition/hybrid-retrieval",
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
    "voyage-lite-01-instruct": "https://docs.voyageai.com/embeddings/",
    "voyage-lite-02-instruct": "https://docs.voyageai.com/embeddings/",
    "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base",
    "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large",
}

EXTERNAL_MODEL_TO_DIM = {
    "Cohere-embed-english-v3.0": 1024,
    "Cohere-embed-multilingual-v3.0": 1024,
    "Cohere-embed-multilingual-light-v3.0": 384,
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "allenai-specter": 768,
    "Baichuan-text-embedding": 1024,
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
    "DanskBERT": 768,
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
    "LASER2": 1024,
    "LaBSE": 768,
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
    "voyage-lite-01-instruct": 1024,
    "voyage-lite-02-instruct": 1024,
    "xlm-roberta-base":  768,
    "xlm-roberta-large":  1024,
}

EXTERNAL_MODEL_TO_SEQLEN = {
    "Cohere-embed-english-v3.0": 512,
    "Cohere-embed-multilingual-v3.0": 512,
    "Cohere-embed-multilingual-light-v3.0": 512,    
    "all-MiniLM-L12-v2": 512,
    "all-MiniLM-L6-v2": 512,
    "all-mpnet-base-v2": 514,
    "allenai-specter": 512,
    "Baichuan-text-embedding": 512,
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
    "DanskBERT": 514,
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
    "LASER2": "N/A",
    "LaBSE": 512,
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
    "voyage-lite-01-instruct": 4000,
    "voyage-lite-02-instruct": 4000,
    "xlm-roberta-base": 514,
    "xlm-roberta-large": 514,
}

EXTERNAL_MODEL_TO_SIZE = {
    "allenai-specter": 0.44,
    "all-MiniLM-L12-v2": 0.13,
    "all-MiniLM-L6-v2": 0.09,
    "all-mpnet-base-v2": 0.44,
    "bert-base-10lang-cased": 0.61,
    "bert-base-15lang-cased": 0.61,
    "bert-base-25lang-cased": 0.61,
    "bert-base-multilingual-cased": 0.71,
    "bert-base-multilingual-uncased": 0.67,
    "bert-base-uncased": 0.44,
    "bert-base-swedish-cased": 0.50,
    "bge-base-zh-v1.5": 0.41,
    "bge-large-en-v1.5": 1.30,
    "bge-large-zh-v1.5": 1.30,
    "bge-large-zh-noinstruct": 1.30,
    "bge-small-zh-v1.5": 0.10,
    "camembert-base": 0.45,
    "camembert-large": 1.35,
    "cross-en-de-roberta-sentence-transformer": 1.11,
    "contriever-base-msmarco": 0.44,
    "distilbert-base-25lang-cased": 0.44,
    "distilbert-base-en-fr-cased": 0.44,
    "distilbert-base-en-fr-es-pt-it-cased": 0.44,
    "distilbert-base-fr-cased": 0.44,
    "distilbert-base-uncased": 0.44,
    "DanskBERT": 0.50,
    "distiluse-base-multilingual-cased-v2": 0.54,
    "dfm-encoder-large-v1": 1.42,
    "dfm-sentence-encoder-large-1": 1.63,
    "e5-base": 0.44,
    "e5-large": 1.34,
    "e5-mistral-7b-instruct": 14.22,
    "e5-small": 0.13,
    "electra-small-nordic": 0.09,
    "electra-small-swedish-cased-discriminator": 0.06,
    "flaubert_base_cased": 0.55,
    "flaubert_base_uncased": 0.55,
    "flaubert_large_cased": 1.49,
    "gbert-base": 0.44,
    "gbert-large": 1.35,
    "gelectra-base": 0.44,
    "gelectra-large": 1.34,
    "glove.6B.300d": 0.48,
    "google-gecko.text-embedding-preview-0409": 2.29,
    "google-gecko-256.text-embedding-preview-0409": 2.29,
    "gottbert-base": 0.51,
    "gtr-t5-base": 0.22,
    "gtr-t5-large": 0.67,
    "gtr-t5-xl": 2.48,
    "gtr-t5-xxl": 9.73,
    "herbert-base-retrieval-v2": 0.50,
    "komninos": 0.27,
    "luotuo-bert-medium": 1.31,    
    "LASER2": 0.17,
    "LaBSE": 1.88,
    "m3e-base": 0.41,
    "m3e-large": 0.41,
    "msmarco-bert-co-condensor": 0.44,
    "multi-qa-MiniLM-L6-cos-v1": 0.09,
    "multilingual-e5-base": 1.11,
    "multilingual-e5-small": 0.47,
    "multilingual-e5-large": 2.24,
    "nb-bert-base": 0.71,
    "nb-bert-large": 1.42,
    "nomic-embed-text-v1.5-64": 0.55,
    "nomic-embed-text-v1.5-128": 0.55,
    "nomic-embed-text-v1.5-256": 0.55,
    "nomic-embed-text-v1.5-512": 0.55,
    "norbert3-base": 0.52,
    "norbert3-large": 1.47,
    "paraphrase-multilingual-mpnet-base-v2": 1.11,
    "paraphrase-multilingual-MiniLM-L12-v2": 0.47,
    "sentence-camembert-base": 0.44,
    "sentence-camembert-large": 1.35,
    "sentence-croissant-llm-base": 5.12,
    "sentence-bert-swedish-cased": 0.50,
    "sentence-t5-base": 0.22,
    "sentence-t5-large": 0.67,
    "sentence-t5-xl": 2.48,
    "sentence-t5-xxl": 9.73,
    "silver-retriever-base-v1": 0.50,
    "sup-simcse-bert-base-uncased": 0.44,
    "st-polish-paraphrase-from-distilroberta": 0.50,
    "st-polish-paraphrase-from-mpnet": 0.50,    
    "text2vec-base-chinese": 0.41,
    "text2vec-large-chinese": 1.30,
    "unsup-simcse-bert-base-uncased": 0.44,
    "use-cmlm-multilingual": 1.89,
    "voyage-lite-02-instruct": 2.45,
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
    elif examples["mteb_dataset_name"] in norm(TASK_LIST_BITEXT_MINING + TASK_LIST_BITEXT_MINING_OTHER):
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
    elif "model.safetensors" in filenames:
        url = hf_hub_url(model.modelId, filename="model.safetensors")
        meta = get_hf_file_metadata(url)
        size = round(meta.size / 1e9, 2)
    elif "model.safetensors.index.json" in filenames:
        index_path = hf_hub_download(model.modelId, filename="model.safetensors.index.json")
        """
        {
        "metadata": {
            "total_size": 14483464192
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
    cols_to_rank = [col for col in df.columns if col not in ["Model", "Model Size (GB)", "Embedding Dimensions", "Max Tokens"]]
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
                    out["Embedding Dimensions"], out["Max Tokens"], out["Model Size (GB)"] = get_dim_seq_size(model)
                except:
                    pass
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

    DATA_CLASSIFICATION_EN = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_CLASSIFICATION])
    # Only keep rows with at least one score in addition to the "Model" & rank column
    DATA_CLASSIFICATION_EN = DATA_CLASSIFICATION_EN[DATA_CLASSIFICATION_EN.iloc[:, 2:].ne("").any(axis=1)]

    DATA_CLUSTERING = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_CLUSTERING])
    DATA_CLUSTERING = DATA_CLUSTERING[DATA_CLUSTERING.iloc[:, 2:].ne("").any(axis=1)]

    DATA_PAIR_CLASSIFICATION = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_PAIR_CLASSIFICATION])
    DATA_PAIR_CLASSIFICATION = DATA_PAIR_CLASSIFICATION[DATA_PAIR_CLASSIFICATION.iloc[:, 2:].ne("").any(axis=1)]

    DATA_RERANKING = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_RERANKING])
    DATA_RERANKING = DATA_RERANKING[DATA_RERANKING.iloc[:, 2:].ne("").any(axis=1)]

    DATA_RETRIEVAL = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_RETRIEVAL])
    DATA_RETRIEVAL = DATA_RETRIEVAL[DATA_RETRIEVAL.iloc[:, 2:].ne("").any(axis=1)]

    DATA_STS_EN = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_STS])
    DATA_STS_EN = DATA_STS_EN[DATA_STS_EN.iloc[:, 2:].ne("").any(axis=1)]

    DATA_SUMMARIZATION = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_SUMMARIZATION])
    DATA_SUMMARIZATION = DATA_SUMMARIZATION[DATA_SUMMARIZATION.iloc[:, 1:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL.fillna("", inplace=True)

    DATA_OVERALL = DATA_OVERALL[["Rank", "Model", "Model Size (GB)", "Embedding Dimensions", "Max Tokens", f"Average ({len(TASK_LIST_EN)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION)} datasets)", f"Reranking Average ({len(TASK_LIST_RERANKING)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)", f"STS Average ({len(TASK_LIST_STS)} datasets)", f"Summarization Average ({len(TASK_LIST_SUMMARIZATION)} dataset)"]]
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

    DATA_CLASSIFICATION_ZH = add_rank(DATA_OVERALL_ZH[["Model"] + TASK_LIST_CLASSIFICATION_ZH])
    # Only keep rows with at least one score in addition to the "Model" & rank column
    DATA_CLASSIFICATION_ZH = DATA_CLASSIFICATION_ZH[DATA_CLASSIFICATION_ZH.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_CLUSTERING_ZH = add_rank(DATA_OVERALL_ZH[["Model"] + TASK_LIST_CLUSTERING_ZH])
    DATA_CLUSTERING_ZH = DATA_CLUSTERING_ZH[DATA_CLUSTERING_ZH.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_PAIR_CLASSIFICATION_ZH = add_rank(DATA_OVERALL_ZH[["Model"] + TASK_LIST_PAIR_CLASSIFICATION_ZH])
    DATA_PAIR_CLASSIFICATION_ZH = DATA_PAIR_CLASSIFICATION_ZH[DATA_PAIR_CLASSIFICATION_ZH.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_RERANKING_ZH = add_rank(DATA_OVERALL_ZH[["Model"] + TASK_LIST_RERANKING_ZH])
    DATA_RERANKING_ZH = DATA_RERANKING_ZH[DATA_RERANKING_ZH.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_RETRIEVAL_ZH = add_rank(DATA_OVERALL_ZH[["Model"] + TASK_LIST_RETRIEVAL_ZH])
    DATA_RETRIEVAL_ZH = DATA_RETRIEVAL_ZH[DATA_RETRIEVAL_ZH.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_STS_ZH = add_rank(DATA_OVERALL_ZH[["Model"] + TASK_LIST_STS_ZH])
    DATA_STS_ZH = DATA_STS_ZH[DATA_STS_ZH.iloc[:, 2:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL_ZH.fillna("", inplace=True)

    DATA_OVERALL_ZH = DATA_OVERALL_ZH[["Rank", "Model", "Model Size (GB)", "Embedding Dimensions", "Max Tokens", f"Average ({len(TASK_LIST_ZH)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION_ZH)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING_ZH)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_ZH)} datasets)", f"Reranking Average ({len(TASK_LIST_RERANKING_ZH)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_ZH)} datasets)", f"STS Average ({len(TASK_LIST_STS_ZH)} datasets)"]]
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

    DATA_CLASSIFICATION_FR = add_rank(DATA_OVERALL_FR[["Model"] + TASK_LIST_CLASSIFICATION_FR])
    DATA_CLASSIFICATION_FR = DATA_CLASSIFICATION_FR[DATA_CLASSIFICATION_FR.iloc[:, 2:].ne("").any(axis=1)]

    DATA_CLUSTERING_FR = add_rank(DATA_OVERALL_FR[["Model"] + TASK_LIST_CLUSTERING_FR])
    DATA_CLUSTERING_FR = DATA_CLUSTERING_FR[DATA_CLUSTERING_FR.iloc[:, 2:].ne("").any(axis=1)]

    DATA_PAIR_CLASSIFICATION_FR = add_rank(DATA_OVERALL_FR[["Model"] + TASK_LIST_PAIR_CLASSIFICATION_FR])
    DATA_PAIR_CLASSIFICATION_FR = DATA_PAIR_CLASSIFICATION_FR[DATA_PAIR_CLASSIFICATION_FR.iloc[:, 2:].ne("").any(axis=1)]

    DATA_RERANKING_FR = add_rank(DATA_OVERALL_FR[["Model"] + TASK_LIST_RERANKING_FR])
    DATA_RERANKING_FR = DATA_RERANKING_FR[DATA_RERANKING_FR.iloc[:, 2:].ne("").any(axis=1)]

    DATA_RETRIEVAL_FR = add_rank(DATA_OVERALL_FR[["Model"] + TASK_LIST_RETRIEVAL_FR])
    DATA_RETRIEVAL_FR = DATA_RETRIEVAL_FR[DATA_RETRIEVAL_FR.iloc[:, 2:].ne("").any(axis=1)]

    DATA_STS_FR = add_rank(DATA_OVERALL_FR[["Model"] + TASK_LIST_STS_FR])
    DATA_STS_FR = DATA_STS_FR[DATA_STS_FR.iloc[:, 2:].ne("").any(axis=1)]

    DATA_SUMMARIZATION_FR = add_rank(DATA_OVERALL_FR[["Model"] + TASK_LIST_SUMMARIZATION_FR])
    DATA_SUMMARIZATION_FR = DATA_SUMMARIZATION_FR[DATA_SUMMARIZATION_FR.iloc[:, 1:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL_FR.fillna("", inplace=True)

    DATA_OVERALL_FR = DATA_OVERALL_FR[["Rank", "Model", "Model Size (GB)", "Embedding Dimensions", "Max Tokens", f"Average ({len(TASK_LIST_FR)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION_FR)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING_FR)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_FR)} datasets)", f"Reranking Average ({len(TASK_LIST_RERANKING_FR)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_FR)} datasets)", f"STS Average ({len(TASK_LIST_STS_FR)} datasets)", f"Summarization Average ({len(TASK_LIST_SUMMARIZATION_FR)} dataset)"]]
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

    DATA_CLASSIFICATION_PL = add_rank(DATA_OVERALL_PL[["Model"] + TASK_LIST_CLASSIFICATION_PL])
    # Only keep rows with at least one score in addition to the "Model" & rank column
    DATA_CLASSIFICATION_PL = DATA_CLASSIFICATION_PL[DATA_CLASSIFICATION_PL.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_CLUSTERING_PL = add_rank(DATA_OVERALL_PL[["Model"] + TASK_LIST_CLUSTERING_PL])
    DATA_CLUSTERING_PL = DATA_CLUSTERING_PL[DATA_CLUSTERING_PL.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_PAIR_CLASSIFICATION_PL = add_rank(DATA_OVERALL_PL[["Model"] + TASK_LIST_PAIR_CLASSIFICATION_PL])
    DATA_PAIR_CLASSIFICATION_PL = DATA_PAIR_CLASSIFICATION_PL[DATA_PAIR_CLASSIFICATION_PL.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_RETRIEVAL_PL = add_rank(DATA_OVERALL_PL[["Model"] + TASK_LIST_RETRIEVAL_PL])
    DATA_RETRIEVAL_PL = DATA_RETRIEVAL_PL[DATA_RETRIEVAL_PL.iloc[:, 2:].ne("").any(axis=1)]
    
    DATA_STS_PL = add_rank(DATA_OVERALL_PL[["Model"] + TASK_LIST_STS_PL])
    DATA_STS_PL = DATA_STS_PL[DATA_STS_PL.iloc[:, 2:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL_PL.fillna("", inplace=True)

    DATA_OVERALL_PL = DATA_OVERALL_PL[["Rank", "Model", "Model Size (GB)", "Embedding Dimensions", "Max Tokens", f"Average ({len(TASK_LIST_PL)} datasets)", f"Classification Average ({len(TASK_LIST_CLASSIFICATION_PL)} datasets)", f"Clustering Average ({len(TASK_LIST_CLUSTERING_PL)} datasets)", f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION_PL)} datasets)", f"Retrieval Average ({len(TASK_LIST_RETRIEVAL_PL)} datasets)", f"STS Average ({len(TASK_LIST_STS_PL)} datasets)"]]
    DATA_OVERALL_PL = DATA_OVERALL_PL[DATA_OVERALL_PL.iloc[:, 5:].ne("").any(axis=1)]

    return DATA_OVERALL_PL

get_mteb_average()
get_mteb_average_fr()
get_mteb_average_pl()
get_mteb_average_zh()

DATA_BITEXT_MINING = get_mteb_data(["BitextMining"], [], TASK_LIST_BITEXT_MINING)
DATA_BITEXT_MINING_OTHER = get_mteb_data(["BitextMining"], [], TASK_LIST_BITEXT_MINING_OTHER)
DATA_CLASSIFICATION_DA = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_DA)
DATA_CLASSIFICATION_NB = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_NB)
DATA_CLASSIFICATION_SV = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_SV)
DATA_CLASSIFICATION_OTHER = get_mteb_data(["Classification"], [], TASK_LIST_CLASSIFICATION_OTHER)
DATA_CLUSTERING_DE = get_mteb_data(["Clustering"], [], TASK_LIST_CLUSTERING_DE)
DATA_STS_OTHER = get_mteb_data(["STS"], [], TASK_LIST_STS_OTHER)
DATA_RETRIEVAL_LAW = get_mteb_data(["Retrieval"], [], TASK_LIST_RETRIEVAL_LAW)

# Exact, add all non-nan integer values for every dataset
NUM_SCORES = 0
DATASETS = []
MODELS = []
# LANGUAGES = []
for d in [
    DATA_BITEXT_MINING,
    DATA_BITEXT_MINING_OTHER,
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
    cols_to_ignore = 3 if "Average" in d.columns else 2
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
"""

block = gr.Blocks(css=css)
with block:
    gr.Markdown(f"""
    Massive Text Embedding Benchmark (MTEB) Leaderboard. To submit, refer to the <a href="https://github.com/embeddings-benchmark/mteb#leaderboard" target="_blank" style="text-decoration: underline">MTEB GitHub repository</a> 🤗 Refer to the [MTEB paper](https://arxiv.org/abs/2210.07316) for details on metrics, tasks and models.
    """)
    with gr.Tabs():
        with gr.TabItem("Overall"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Overall MTEB English leaderboard** 🔮
                    
                    - **Metric:** Various, refer to task tabs
                    - **Languages:** English
                    """)
                with gr.Row():
                    data_overall = gr.components.Dataframe(
                        DATA_OVERALL,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_OVERALL.columns),
                        type="pandas",
                        height=600,
                    )
                with gr.Row():
                    data_run_overall = gr.Button("Refresh")
                    data_run_overall.click(get_mteb_average, inputs=None, outputs=data_overall)
            with gr.TabItem("Chinese"):
                with gr.Row():
                    gr.Markdown("""
                    **Overall MTEB Chinese leaderboard (C-MTEB)** 🔮🇨🇳
                    
                    - **Metric:** Various, refer to task tabs
                    - **Languages:** Chinese
                    - **Credits:** [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
                    """)
                with gr.Row():
                    data_overall_zh = gr.components.Dataframe(
                        DATA_OVERALL_ZH,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_OVERALL_ZH.columns),
                        type="pandas",
                        height=600,
                    )
                with gr.Row():
                    data_run_overall_zh = gr.Button("Refresh")
                    data_run_overall_zh.click(get_mteb_average_zh, inputs=None, outputs=data_overall_zh)    
            with gr.TabItem("French"):
                with gr.Row():
                    gr.Markdown("""
                    **Overall MTEB French leaderboard (F-MTEB)** 🔮🇫🇷
                    
                    - **Metric:** Various, refer to task tabs
                    - **Languages:** French
                    - **Credits:** [Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [Wissam Siblini](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)
                    """)
                with gr.Row():
                    data_overall_fr = gr.components.Dataframe(
                        DATA_OVERALL_FR,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_OVERALL_FR.columns),
                        type="pandas",
                        height=600,
                    )
                with gr.Row():
                    data_overall_fr = gr.Button("Refresh")
                    data_overall_fr.click(get_mteb_average_fr, inputs=None, outputs=data_overall_fr)
            with gr.TabItem("Polish"):
                with gr.Row():
                    gr.Markdown("""
                    **Overall MTEB Polish leaderboard (PL-MTEB)** 🔮🇵🇱
                    
                    - **Metric:** Various, refer to task tabs
                    - **Languages:** Polish
                    - **Credits:** [Rafał Poświata](https://github.com/rafalposwiata), [Konrad Wojtasik](https://github.com/kwojtasi) & [BEIR-PL](https://arxiv.org/abs/2305.19840)
                    """)
                with gr.Row():
                    data_overall_pl = gr.components.Dataframe(
                        DATA_OVERALL_PL,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_OVERALL_PL.columns),
                        type="pandas",
                        height=600,                        
                    )
                with gr.Row():
                    data_run_overall_pl = gr.Button("Refresh")
                    data_run_overall_pl.click(get_mteb_average_pl, inputs=None, outputs=data_overall_pl)                    
        with gr.TabItem("Bitext Mining"):
            with gr.TabItem("English-X"):
                with gr.Row():
                        gr.Markdown("""
                        **Bitext Mining English-X Leaderboard** 🎌
                        
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
                    data_run_bitext_mining = gr.Button("Refresh")
                    data_run_bitext_mining.click(
                        partial(get_mteb_data, tasks=["BitextMining"], datasets=TASK_LIST_BITEXT_MINING),
                        outputs=data_bitext_mining,
                    )
            with gr.TabItem("Danish"):
                with gr.Row():
                        gr.Markdown("""
                        **Bitext Mining Danish Leaderboard** 🎌🇩🇰
                        
                        - **Metric:** [F1](https://huggingface.co/spaces/evaluate-metric/f1)
                        - **Languages:** Danish & Bornholmsk (Danish Dialect)
                        - **Credits:** [Kenneth Enevoldsen](https://github.com/KennethEnevoldsen), [scandinavian-embedding-benchmark](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)
                        """)
                with gr.Row():
                    data_bitext_mining_da = gr.components.Dataframe(
                        DATA_BITEXT_MINING_OTHER,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_BITEXT_MINING_OTHER.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_bitext_mining_da = gr.Button("Refresh")
                    data_run_bitext_mining_da.click(
                        partial(get_mteb_data, tasks=["BitextMining"], datasets=TASK_LIST_BITEXT_MINING_OTHER),
                        outputs=data_bitext_mining_da,
                    )
        with gr.TabItem("Classification"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification English Leaderboard** ❤️
                    
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
                    data_run_classification_en.click(
                        partial(get_mteb_data, tasks=["Classification"], langs=["en"]),
                        outputs=data_classification_en,
                    )
            with gr.TabItem("Chinese"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Chinese Leaderboard** 🧡🇨🇳
                    
                    - **Metric:** [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
                    - **Languages:** Chinese
                    - **Credits:** [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
                    """)
                with gr.Row():
                    data_classification_zh = gr.components.Dataframe(
                        DATA_CLASSIFICATION_ZH,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLASSIFICATION_ZH.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_classification_zh = gr.Button("Refresh")
                    data_run_classification_zh.click(
                        partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_ZH),
                        outputs=data_classification_zh,
                    )
            with gr.TabItem("Danish"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Danish Leaderboard** 🤍🇩🇰
                    
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
                    data_run_classification_da.click(
                        partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_DA),
                        outputs=data_run_classification_da,
                    )
            with gr.TabItem("French"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification French Leaderboard** 💙🇫🇷
                    
                    - **Metric:** [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
                    - **Languages:** French
                    - **Credits:** [Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [wissam-sib](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)
                    """)
                with gr.Row():
                    data_classification_fr = gr.components.Dataframe(
                        DATA_CLASSIFICATION_FR,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLASSIFICATION_FR.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_classification_fr = gr.Button("Refresh")
                    data_run_classification_fr.click(
                        partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_FR),
                        outputs=data_run_classification_fr,
                    )                    
            with gr.TabItem("Norwegian"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Norwegian Leaderboard** 💙🇳🇴
                    
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
                    data_run_classification_nb.click(
                        partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_NB),
                        outputs=data_classification_nb,
                    )
            with gr.TabItem("Polish"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Polish Leaderboard** 🤍🇵🇱
                    
                    - **Metric:** [Accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
                    - **Languages:** Polish
                    - **Credits:** [Rafał Poświata](https://github.com/rafalposwiata)
                    """)
                with gr.Row():
                    data_classification_pl = gr.components.Dataframe(
                        DATA_CLASSIFICATION_PL,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLASSIFICATION_PL.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_classification_pl = gr.Button("Refresh")
                    data_run_classification_pl.click(
                        partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_PL),
                        outputs=data_classification_pl,
                    )
            with gr.TabItem("Swedish"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Swedish Leaderboard** 💛🇸🇪
                    
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
                    data_run_classification_sv.click(
                        partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_SV),
                        outputs=data_classification_sv,
                    )
            with gr.TabItem("Other"):
                with gr.Row():
                    gr.Markdown("""
                    **Classification Other Languages Leaderboard** 💜💚💙
                    
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
                    data_run_classification = gr.Button("Refresh")
                    data_run_classification.click(
                        partial(get_mteb_data, tasks=["Classification"], datasets=TASK_LIST_CLASSIFICATION_OTHER),
                        outputs=data_classification,
                    )                                    
        with gr.TabItem("Clustering"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Clustering Leaderboard** ✨
                    
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
                    data_run_clustering_en = gr.Button("Refresh")
                    data_run_clustering_en.click(
                        partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING),
                        outputs=data_clustering,
                    )
            with gr.TabItem("Chinese"):
                with gr.Row():
                    gr.Markdown("""
                    **Clustering Chinese Leaderboard** ✨🇨🇳
                    
                    - **Metric:** Validity Measure (v_measure)
                    - **Languages:** Chinese
                    - **Credits:** [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
                    """)
                with gr.Row():
                    data_clustering_zh = gr.components.Dataframe(
                        DATA_CLUSTERING_ZH,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLUSTERING_ZH.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_clustering_zh = gr.Button("Refresh")
                    data_run_clustering_zh.click(
                        partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING_ZH),
                        outputs=data_clustering_zh,
                    )
            with gr.TabItem("French"):
                with gr.Row():
                    gr.Markdown("""
                    **Clustering French Leaderboard** ✨🇫🇷
                    
                    - **Metric:** Validity Measure (v_measure)
                    - **Languages:** French
                    - **Credits:** [Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [wissam-sib](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)
                    """)
                with gr.Row():
                    data_clustering_fr = gr.components.Dataframe(
                        DATA_CLUSTERING_FR,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLUSTERING_FR.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_clustering_fr = gr.Button("Refresh")
                    data_run_clustering_fr.click(
                        partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING_FR),
                        outputs=data_clustering_fr,
                    )
            with gr.TabItem("German"):
                with gr.Row():
                    gr.Markdown("""
                    **Clustering German Leaderboard** ✨🇩🇪
                    
                    - **Metric:** Validity Measure (v_measure)
                    - **Languages:** German
                    - **Credits:** [Silvan](https://github.com/slvnwhrl)
                    """)
                with gr.Row():
                    data_clustering_de = gr.components.Dataframe(
                        DATA_CLUSTERING_DE,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLUSTERING_DE.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_clustering_de = gr.Button("Refresh")
                    data_run_clustering_de.click(
                        partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING_DE),
                        outputs=data_clustering_de,
                    )
            with gr.TabItem("Polish"):
                with gr.Row():
                    gr.Markdown("""
                    **Clustering Polish Leaderboard** ✨🇵🇱
                    
                    - **Metric:** Validity Measure (v_measure)
                    - **Languages:** Polish
                    - **Credits:** [Rafał Poświata](https://github.com/rafalposwiata)
                    """)
                with gr.Row():
                    data_clustering_pl = gr.components.Dataframe(
                        DATA_CLUSTERING_PL,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_CLUSTERING_PL.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_clustering_pl = gr.Button("Refresh")
                    data_run_clustering_pl.click(
                        partial(get_mteb_data, tasks=["Clustering"], datasets=TASK_LIST_CLUSTERING_PL),
                        outputs=data_clustering_pl,
                    )
        with gr.TabItem("Pair Classification"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Pair Classification English Leaderboard** 🎭
                    
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
                    data_run_pair_classification = gr.Button("Refresh")
                    data_run_pair_classification.click(
                        partial(get_mteb_data, tasks=["PairClassification"], datasets=TASK_LIST_PAIR_CLASSIFICATION),
                        outputs=data_pair_classification,
                    )
            with gr.TabItem("Chinese"):
                with gr.Row():
                    gr.Markdown("""
                    **Pair Classification Chinese Leaderboard** 🎭🇨🇳
                    
                    - **Metric:** Average Precision based on Cosine Similarities (cos_sim_ap)
                    - **Languages:** Chinese
                    - **Credits:** [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
                    """)
                with gr.Row():
                    data_pair_classification_zh = gr.components.Dataframe(
                        DATA_PAIR_CLASSIFICATION_ZH,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_PAIR_CLASSIFICATION_ZH.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_pair_classification_zh = gr.Button("Refresh")
                    data_run_pair_classification_zh.click(
                        partial(get_mteb_data, tasks=["PairClassification"], datasets=TASK_LIST_PAIR_CLASSIFICATION_ZH),
                        outputs=data_pair_classification_zh,
                    )
            with gr.TabItem("French"):
                with gr.Row():
                    gr.Markdown("""
                    **Pair Classification French Leaderboard** 🎭🇫🇷
                    
                    - **Metric:** Average Precision based on Cosine Similarities (cos_sim_ap)
                    - **Languages:** French
                    - **Credits:** [Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [wissam-sib](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)
                    """)
                with gr.Row():
                    data_pair_classification_fr = gr.components.Dataframe(
                        DATA_PAIR_CLASSIFICATION_FR,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_PAIR_CLASSIFICATION_FR.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_pair_classification_fr = gr.Button("Refresh")
                    data_run_pair_classification_fr.click(
                        partial(get_mteb_data, tasks=["PairClassification"], datasets=TASK_LIST_PAIR_CLASSIFICATION_FR),
                        outputs=data_pair_classification_fr,
                    )
            with gr.TabItem("Polish"):
                with gr.Row():
                    gr.Markdown("""
                    **Pair Classification Polish Leaderboard** 🎭🇵🇱
                    
                    - **Metric:** Average Precision based on Cosine Similarities (cos_sim_ap)
                    - **Languages:** Polish
                    - **Credits:** [Rafał Poświata](https://github.com/rafalposwiata)
                    """)
                with gr.Row():
                    data_pair_classification_pl = gr.components.Dataframe(
                        DATA_PAIR_CLASSIFICATION_PL,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_PAIR_CLASSIFICATION_PL.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_pair_classification_pl = gr.Button("Refresh")
                    data_run_pair_classification_pl.click(
                        partial(get_mteb_data, tasks=["PairClassification"], datasets=TASK_LIST_PAIR_CLASSIFICATION_PL),
                        outputs=data_pair_classification_pl,
                    )               
        with gr.TabItem("Reranking"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Reranking English Leaderboard** 🥈
                    
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
                    data_run_reranking = gr.Button("Refresh")
                    data_run_reranking.click(
                        partial(get_mteb_data, tasks=["Reranking"], datasets=TASK_LIST_RERANKING),
                        outputs=data_reranking,
                    )
            with gr.TabItem("Chinese"):
                with gr.Row():
                    gr.Markdown("""
                    **Reranking Chinese Leaderboard** 🥈🇨🇳
                    
                    - **Metric:** Mean Average Precision (MAP)
                    - **Languages:** Chinese
                    - **Credits:** [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
                    """)
                with gr.Row():
                    data_reranking_zh = gr.components.Dataframe(
                        DATA_RERANKING_ZH,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_RERANKING_ZH.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_reranking_zh = gr.Button("Refresh")
                    data_run_reranking_zh.click(
                        partial(get_mteb_data, tasks=["Reranking"], datasets=TASK_LIST_RERANKING_ZH),
                        outputs=data_reranking_zh,
                    )
            with gr.TabItem("French"):
                with gr.Row():
                    gr.Markdown("""
                    **Reranking French Leaderboard** 🥈🇫🇷
                    
                    - **Metric:** Mean Average Precision (MAP)
                    - **Languages:** French
                    - **Credits:** [Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [wissam-sib](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)
                    """)
                with gr.Row():
                    data_reranking_fr = gr.components.Dataframe(
                        DATA_RERANKING_FR,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_RERANKING_FR.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_reranking_fr = gr.Button("Refresh")
                    data_run_reranking_fr.click(
                        partial(get_mteb_data, tasks=["Reranking"], datasets=TASK_LIST_RERANKING_FR),
                        outputs=data_reranking_fr,
                    )
        with gr.TabItem("Retrieval"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Retrieval English Leaderboard** 🔎
                    
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
                    data_run_retrieval = gr.Button("Refresh")
                    data_run_retrieval.click(
                        partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL),
                        outputs=data_retrieval,
                    )
            with gr.TabItem("Chinese"):
                with gr.Row():
                    gr.Markdown("""
                    **Retrieval Chinese Leaderboard** 🔎🇨🇳
                    
                    - **Metric:** Normalized Discounted Cumulative Gain @ k (ndcg_at_10)
                    - **Languages:** Chinese
                    - **Credits:** [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
                    """)
                with gr.Row():
                    data_retrieval_zh = gr.components.Dataframe(
                        DATA_RETRIEVAL_ZH,
                        # Add support for more columns than existing as a buffer for CQADupstack & other Retrieval tasks (e.g. MSMARCOv2)
                        datatype=["number", "markdown"] + ["number"] * len(DATA_RETRIEVAL_ZH.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_retrieval_zh = gr.Button("Refresh")
                    data_run_retrieval_zh.click(
                        partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL_ZH),
                        outputs=data_retrieval_zh,
                    )
            with gr.TabItem("French"):
                with gr.Row():
                    gr.Markdown("""
                    **Retrieval French Leaderboard** 🔎🇫🇷
                    
                    - **Metric:** Normalized Discounted Cumulative Gain @ k (ndcg_at_10)
                    - **Languages:** French
                    - **Credits:** [Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [wissam-sib](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)
                    """)
                with gr.Row():
                    data_retrieval_fr = gr.components.Dataframe(
                        DATA_RETRIEVAL_FR,
                        # Add support for more columns than existing as a buffer for CQADupstack & other Retrieval tasks (e.g. MSMARCOv2)
                        datatype=["number", "markdown"] + ["number"] * len(DATA_RETRIEVAL_FR.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_retrieval_fr = gr.Button("Refresh")
                    data_run_retrieval_fr.click(
                        partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL_FR),
                        outputs=data_retrieval_fr,
                    )    
            with gr.TabItem("Law"):
                with gr.Row():
                    gr.Markdown("""
                    **Retrieval Law Leaderboard** 🔎⚖️
                    
                    - **Metric:** Normalized Discounted Cumulative Gain @ k (ndcg_at_10)
                    - **Languages:** English, German, Chinese
                    - **Credits:** [Voyage AI](https://www.voyageai.com/)
                    """)
                with gr.Row():
                    data_retrieval_law = gr.components.Dataframe(
                        DATA_RETRIEVAL_LAW,
                        # Add support for more columns than existing as a buffer for CQADupstack & other Retrieval tasks (e.g. MSMARCOv2)
                        datatype=["number", "markdown"] + ["number"] * len(DATA_RETRIEVAL_LAW.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_retrieval_law = gr.Button("Refresh")
                    data_run_retrieval_law.click(
                        partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL_LAW),
                        outputs=data_retrieval_law,
                    )
            with gr.TabItem("Polish"):
                with gr.Row():
                    gr.Markdown("""
                    **Retrieval Polish Leaderboard** 🔎🇵🇱
                    
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
                    data_run_retrieval_pl = gr.Button("Refresh")
                    data_run_retrieval_pl.click(
                        partial(get_mteb_data, tasks=["Retrieval"], datasets=TASK_LIST_RETRIEVAL_PL),
                        outputs=data_retrieval_pl,
                    )
        with gr.TabItem("STS"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **STS English Leaderboard** 🤖
                    
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
                    data_run_sts_en.click(
                        partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS),
                        outputs=data_sts_en,
                    )
            with gr.TabItem("Chinese"):
                with gr.Row():
                    gr.Markdown("""
                    **STS Chinese Leaderboard** 🤖🇨🇳
                    
                    - **Metric:** Spearman correlation based on cosine similarity
                    - **Languages:** Chinese
                    - **Credits:** [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
                    """)
                with gr.Row():
                    data_sts_zh = gr.components.Dataframe(
                        DATA_STS_ZH,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_STS_ZH.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_sts_zh = gr.Button("Refresh")
                    data_run_sts_zh.click(
                        partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS_ZH),
                        outputs=data_sts_zh,
                    )
            with gr.TabItem("French"):
                with gr.Row():
                    gr.Markdown("""
                    **STS French Leaderboard** 🤖🇫🇷
                    
                    - **Metric:** Spearman correlation based on cosine similarity
                    - **Languages:** French
                    - **Credits:** [Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [wissam-sib](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)
                    """)
                with gr.Row():
                    data_sts_fr = gr.components.Dataframe(
                        DATA_STS_FR,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_STS_FR.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_sts_fr = gr.Button("Refresh")
                    data_run_sts_fr.click(
                        partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS_FR),
                        outputs=data_sts_fr,
                    )                    
            with gr.TabItem("Polish"):
                with gr.Row():
                    gr.Markdown("""
                    **STS Polish Leaderboard** 🤖🇵🇱
                    
                    - **Metric:** Spearman correlation based on cosine similarity
                    - **Languages:** Polish
                    - **Credits:** [Rafał Poświata](https://github.com/rafalposwiata)
                    """)
                with gr.Row():
                    data_sts_pl = gr.components.Dataframe(
                        DATA_STS_PL,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_STS_PL.columns),
                        type="pandas",
                    )
                with gr.Row():
                    data_run_sts_pl = gr.Button("Refresh")
                    data_run_sts_pl.click(
                        partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS_PL),
                        outputs=data_sts_pl,
                    )
            with gr.TabItem("Other"):
                with gr.Row():
                    gr.Markdown("""
                    **STS Other Leaderboard** 👽
                    
                    - **Metric:** Spearman correlation based on cosine similarity
                    - **Languages:** Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Russian, Spanish (Only language combos not included in the other tabs)
                    """)
                with gr.Row():
                    data_sts_other = gr.components.Dataframe(
                        DATA_STS_OTHER,
                        datatype=["number", "markdown"] + ["number"] * len(DATA_STS_OTHER.columns) * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_sts_other = gr.Button("Refresh")
                    data_run_sts_other.click(
                        partial(get_mteb_data, tasks=["STS"], datasets=TASK_LIST_STS_OTHER),
                        outputs=data_sts_other,
                    )
        with gr.TabItem("Summarization"):
            with gr.TabItem("English"):
                with gr.Row():
                    gr.Markdown("""
                    **Summarization Leaderboard** 📜
                    
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
                    data_run.click(
                        partial(get_mteb_data, tasks=TASK_LIST_SUMMARIZATION),
                        outputs=data_summarization,
                    )
            with gr.TabItem("French"):
                with gr.Row():
                    gr.Markdown("""
                    **Summarization Leaderboard** 📜
                    
                    - **Metric:** Spearman correlation based on cosine similarity
                    - **Languages:** French
                    - **Credits:** [Lyon-NLP](https://github.com/Lyon-NLP): [Gabriel Sequeira](https://github.com/GabrielSequeira), [Imene Kerboua](https://github.com/imenelydiaker), [wissam-sib](https://github.com/wissam-sib), [Mathieu Ciancone](https://github.com/MathieuCiancone), [Marion Schaeffer](https://github.com/schmarion)                            
                    """)
                with gr.Row():
                    data_summarization_fr = gr.components.Dataframe(
                        DATA_SUMMARIZATION_FR,
                        datatype=["number", "markdown"] + ["number"] * 2,
                        type="pandas",
                    )
                with gr.Row():
                    data_run_summarization_fr = gr.Button("Refresh")
                    data_run_summarization_fr.click(
                        partial(get_mteb_data, tasks=TASK_LIST_SUMMARIZATION_FR),
                        outputs=data_run_summarization_fr,
                    )
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
    # Running the functions on page load in addition to when the button is clicked
    # This is optional - If deactivated the data loaded at "Build time" is shown like for Overall tab
    """
    block.load(get_mteb_data, inputs=[task_bitext_mining], outputs=data_bitext_mining)
    """

block.queue(max_size=10)
block.launch()


# Possible changes:
# Could add graphs / other visual content
# Could add verification marks

# Sources:
# https://huggingface.co/spaces/gradio/leaderboard
# https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard
# https://getemoji.com/
