from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings

import logging


def load_embedding_model(embedding_model_name: str):
    if embedding_model_name.startswith("bge"):
        # local_dir = "../data/embedding/"
        # if not os.path.exists(local_dir):
        #     model_dir = snapshot_download(embedding_model_name, local_dir=local_dir)
        # else:
        #     model_dir = snapshot_download(embedding_model_name, local_files_only=True, local_dir=local_dir)
        # print(model_dir)
        # embeddings = SentenceTransformerEmbeddings(model_name=model_dir, trust_remote_code=True)
        # model_dir="/data_G/zhenyang/Project_all/model/bge-large-zh-v1.5"
        model_dir="/mnt/sdb/Disk_A/zhijian/project/model/bge-large-zh-v1.5"

        embeddings = SentenceTransformerEmbeddings(model_name=model_dir)
        # dimension = 768
        dimension = 1024
        #
        logging.info(f"Embedding: Using modelscope Embeddings{embedding_model_name} , Dimension:{dimension}")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logging.info(f"Embedding: Using OpenAI Embeddings , Dimension:{dimension}")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name=r"E:\model\all-MiniLM-L6-v2"
        )
        dimension = 384
        logging.info(f"Embedding: Using SentenceTransformer , Dimension:{dimension}")
    return embeddings, dimension