import logging
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI, AzureChatOpenAI
# from langchain_google_vertexai import ChatVertexAI
# from langchain_groq import ChatGroq
# from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
# from langchain_anthropic import ChatAnthropic
# from langchain_fireworks import ChatFireworks
# from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models.tongyi import ChatTongyi
# import boto3
# import google.auth

from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from langchain_experimental.graph_transformers import LLMGraphTransformer

# from ktem.pages.src.graph_transformers.llm import LLMGraphTransformer
from ktem.pages.src.mcqgenerator.constants import MODEL_VERSIONS


def get_llm(model_version: str):
    """Retrieve the specified language model based on the model name."""
    env_key = "LLM_MODEL_CONFIG_" + model_version
    env_value = os.environ.get(env_key)
    logging.info("Model: {}".format(env_key))
    model_name = MODEL_VERSIONS[model_version]
    if "Ollama" in model_version:
        # model_name, base_url = env_value.split(",")
        llm = ChatOpenAI(api_key=os.environ.get('OLLAMA_API_KEY'),
                         base_url=os.environ.get('OLLAMA_API_URL'),
                         model=model_name,
                         # top_p=0.7,
                         temperature=0.7)
    elif "glm" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('ZHIPUAI_API_KEY'),
                         base_url=os.environ.get('ZHIPUAI_API_URL'),
                         model=model_name,
                         # top_p=0.7,
                         temperature=0.98)

    elif "moonshot" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('MOONSHOT_API_KEY'),
                         base_url=os.environ.get('MOONSHOT_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95)
    elif "Baichuan" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('BAICHUAN_API_KEY'),
                         base_url=os.environ.get('BAICHUAN_API_URL'),
                         model=model_name,
                         # top_p=0.7,
                         temperature=0.95)
    elif "yi-large" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('LINGYIWANWU_API_KEY'),
                         base_url=os.environ.get('LINGYIWANWU_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95)
    elif "deepseek" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'),
                         base_url=os.environ.get('DEEPSEEK_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95)
    elif "qwen" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('QWEN_API_KEY'),
                         base_url=os.environ.get('QWEN_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95
                         )
    elif "Doubao" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('DOUBAO_API_KEY'),
                         base_url=os.environ.get('DOUBAO_API_URL'),
                         model=os.environ.get('ENDPOINT_ID'),
                         # top_p=0.7,
                         # temperature=0.95
                         )

    elif "openai" in model_version:
        model_name = MODEL_VERSIONS[model_version]
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model_name,
            temperature=0,
        )

    elif "azure" in model_version:
        model_name, api_endpoint, api_key, api_version = env_value.split(",")
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=api_endpoint,
            azure_deployment=model_name,  # takes precedence over model parameter
            api_version=api_version,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )
    else:
        model_name = "diffbot"
        llm = DiffbotGraphTransformer(
            diffbot_api_key=os.environ.get("DIFFBOT_API_KEY"),
            extract_types=["entities", "facts"],
        )
    logging.info(f"Model created - Model Version: {model_version}")
    return llm, model_name


def get_combined_chunks(chunkId_chunkDoc_list):
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship, use_function=True
):
    use_function =True
    futures = []
    graph_document_list = []
    if not use_function:
        node_properties = False
    else:
        node_properties = ["description"]
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=node_properties,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
        # use_function_call=use_function
    )
    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in combined_chunk_document_list:
            chunk_doc = Document(
                page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
            )
            futures.append(
                executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
            )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])

    return graph_document_list


def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    llm, model_name = get_llm(model)
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    graph_document_list = get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship
    )
    return graph_document_list


def get_graph_document(model, document, allowedNodes, allowedRelationship):
    llm, model_name = get_llm(model)
    use_function = True
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
        # use_function_call=use_function
    )
    document = [Document(page_content=document)]

    graph_document = llm_transformer.convert_to_graph_documents(document)
    return graph_document

def extract_graph_document(model, document, allowedNodes, allowedRelationship=None):
    graph_document = get_graph_document(model, document, allowedNodes, allowedRelationship)
    nodes = []
    relationships = {"triples": []}
    for node in graph_document[0].nodes:
        nodes.append(node.id)
    # for relationship in graph_document[0].relationships:
    #     relationships.get("triples").append({"start": relationship.source.id,
    #                                          "relationship": relationship.type,
    #                                          "end": relationship.target.id})
    return nodes

if __name__ == "__main__":
    llm, _=get_llm("Ollama")
    response=llm.invoke("nihao ")
    print(response)