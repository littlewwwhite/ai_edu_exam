from datetime import datetime
import json
import os
import time
import logging
import asyncio
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.vectorstores import Neo4jVector

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableWithMessageHistory
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_text_splitters import TokenTextSplitter
from langchain_core.messages import HumanMessage,AIMessage
from langchain.chains import GraphCypherQAChain
from ktem.pages.src.mcqgenerator.llm import extract_graph_document,get_llm
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings


from ktem.pages.src.mcqgenerator.constants import *
from ktem.pages.src.mcqgenerator.embedding import load_embedding_model

from dotenv import load_dotenv
load_dotenv()



uri = os.getenv('NEO4J_URI')
userName = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')
database = os.getenv('NEO4J_DATABASE')


async def RAG(question, mode="graph+vector", document_names="[]", session_id="001", model="深度求索", graph_type="Microsoft"):

    graph = Neo4jGraph(url=uri, username=userName, password=password, database=database, sanitize=True,
                       refresh_schema=True)

    # 替代 asyncio.to_thread 使用 loop.run_in_executor
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, QA_RAG, graph, model, question, document_names, session_id, mode, graph_type)

    return result


def extract_info(graph, question, model):
    allow_nodes= []
    for item in graph.query(NODES_QUERY):
        allow_nodes.append(item["label"])

    # allow_relationships=graph.query(RELATIONSHIPS_QUERY)
    nodes = extract_graph_document(model, question, allow_nodes)
    return nodes

def create_neo4j_chat_message_history(graph, session_id):
    """
    Creates and returns a Neo4jChatMessageHistory instance.

    """
    try:

        history = Neo4jChatMessageHistory(
            graph=graph,
            session_id=session_id
        )
        return history

    except Exception as e:
        logging.error(f"Error creating Neo4jChatMessageHistory: {e}")
    return None

def create_graph_chain(model, graph):
    try:
        logging.info(f"Graph QA Chain using LLM model: {model}")

        cypher_llm,model_name = get_llm(model)
        qa_llm,model_name = get_llm(model)
        graph_chain = GraphCypherQAChain.from_llm(
            cypher_llm=cypher_llm,
            qa_llm=qa_llm,
            validate_cypher= True,
            graph=graph,
            verbose=True,
            return_intermediate_steps = True,
            top_k=3
        )

        logging.info("GraphCypherQAChain instance created successfully.")
        return graph_chain,qa_llm,model_name

    except Exception as e:
        logging.error(f"An error occurred while creating the GraphCypherQAChain instance. : {e}")


def get_graph_response(graph_chain, question):
    try:
        cypher_res = graph_chain.invoke({"query": question})

        response = cypher_res.get("result")
        cypher_query = ""
        context = []

        for step in cypher_res.get("intermediate_steps", []):
            if "query" in step:
                cypher_string = step["query"]
                cypher_query = cypher_string.replace("cypher\n", "").replace("\n", " ").strip()
            elif "context" in step:
                context = step["context"]
        return {
            "response": response,
            "cypher_query": cypher_query,
            "context": context
        }

    except Exception as e:
        logging.error("An error occurred while getting the graph response : {e}")

def format_documents(documents,model):
    prompt_token_cutoff = 4
    for models,value in CHAT_TOKEN_CUT_OFF.items():
        if model in models:
            prompt_token_cutoff = value

    sorted_documents = sorted(documents, key=lambda doc: doc.state["query_similarity_score"], reverse=True)
    sorted_documents = sorted_documents[:prompt_token_cutoff]

    formatted_docs = []
    sources = set()

    for doc in sorted_documents:
        source = doc.metadata['source']
        sources.add(source)

        formatted_doc = (
            "Document start\n"
            f"This Document belongs to the source {source}\n"
            f"Content: {doc.page_content}\n"
            "Document end\n"
        )
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs), sources

def get_rag_chain(llm,system_template=CHAT_SYSTEM_TEMPLATE):
    question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
        (
                "human",
                "User question: {input}"
            ),
    ]
    )
    question_answering_chain = question_answering_prompt | llm

    return question_answering_chain

def get_sources_and_chunks(sources_used, docs):
    chunkdetails_list = []
    sources_used_set = set(sources_used)

    for doc in docs:
        source = doc.metadata["source"]
        chunkdetails = doc.metadata["chunkdetails"]
        if source in sources_used_set:
            chunkdetails = [{**chunkdetail, "score": round(chunkdetail["score"], 4)} for chunkdetail in chunkdetails]
            chunkdetails_list.extend(chunkdetails)

    result = {
        'sources': sources_used,
        'chunkdetails': chunkdetails_list
    }
    return result

def process_documents(docs, question, llm,model):
    start_time = time.time()
    formatted_docs, sources = format_documents(docs,model)
    # rag_chain = get_rag_chain(llm=llm)
    # ai_response = rag_chain.invoke({
    #     "messages": messages[:-1],
    #     "context": formatted_docs,
    #     "input": question
    # })
    # result = get_sources_and_chunks(sources, docs)
    # content = ai_response.content


    predict_time = time.time() - start_time
    logging.info(f"Final Response predicted in {predict_time:.2f} seconds")

    return formatted_docs

def QA_RAG(graph, model, question, document_names,session_id, mode, graph_type):
    try:
        # logging.info(f"Chat Mode : {mode}")
        # history = create_neo4j_chat_message_history(graph, session_id)
        # messages = history.messages
        # user_question = HumanMessage(content=question)
        # messages.append(user_question)


        if mode == "graph":
            graph_chain, qa_llm,model_version = create_graph_chain(model,graph)
            graph_response = get_graph_response(graph_chain,question)
            # ai_response = AIMessage(content=graph_response["response"]) if graph_response["response"] else AIMessage(content="Something went wrong")
            # messages.append(ai_response)
            # summarize_and_log(history, messages, qa_llm)

            result = {
                "session_id": session_id,
                "message": graph_response["response"],
                "info": {
                    "model": model_version,
                    'cypher_query':graph_response["cypher_query"],
                    "context" : graph_response["context"],
                    "mode" : mode,
                    "response_time": 0
                },
                "user": "RAG"
            }
            return result
        elif mode == "vector":
            retrieval_query = VECTOR_SEARCH_QUERY
        else:
            if graph_type == "langchain":
                # 提取查询语句中的实体和关系
                info_in_question = extract_info(graph, question, model)
                retrieval_query = VECTOR_GRAPH_SEARCH_QUERY.format(no_of_entites=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT,
                                                                   nodes=info_in_question)
            else:
                # retrieval_query = lc_retrieval_query
                retrieval_query=test
        # print("retrieval query: {}".format(retrieval_query))
        llm, doc_retriever, model_version = setup_chat(model, graph, session_id, document_names,retrieval_query, graph_type=graph_type)

        docs = retrieve_documents(doc_retriever, question)

        if docs:
            if graph_type=="langchain":
                content= process_documents(docs, question, llm,model)

            else:
                content= docs
        else:
            content = "I couldn't find any relevant documents to answer your question."
            # result = {"sources": [], "chunkdetails": []}
            total_tokens = 0

        # ai_response = AIMessage(content=content)
        # messages.append(ai_response)
        # summarize_and_log(history, messages, llm)
        # print(content)
        return {
            "session_id": session_id,
            "message": content,
            "info": {
                "model": model_version,
                "response_time": 0,
                "mode": mode
            },
            "user": "RAG"
        }

    except Exception as e:
        logging.exception(f"Exception in QA component at {datetime.now()}: {str(e)}")
        error_name = type(e).__name__
        return {
            "session_id": session_id,
            "message": "Something went wrong",
            "info": {
                "sources": [],
                "chunkids": [],
                "error": f"{error_name} :- {str(e)}",
                "mode": mode
            },
            "user": "chatbot"
        }


def setup_chat(model, graph, session_id, document_names,retrieval_query, graph_type):
    start_time = time.time()
    if model in ["diffbot"]:
        model = "openai-gpt-4o"
    llm,model_name = get_llm(model)
    logging.info(f"Model called in chat {model} and model version is {model_name}")
    retriever, EMBEDDING_FUNCTION = get_neo4j_retriever(graph=graph,retrieval_query=retrieval_query,document_names=document_names,graph_type=graph_type)
    # doc_retriever = create_document_retriever_chain(llm, retriever, EMBEDDING_FUNCTION)

    chat_setup_time = time.time() - start_time
    logging.info(f"Chat setup completed in {chat_setup_time:.2f} seconds")

    return llm, retriever, model_name

def get_neo4j_retriever(graph, retrieval_query,document_names,index_name="vector", search_k=CHAT_SEARCH_KWARG_K, score_threshold=CHAT_SEARCH_KWARG_SCORE_THRESHOLD,graph_type="langchain"):
    try:
        if graph_type == "Microsoft":
            EMBEDDING_FUNCTION = OpenAIEmbeddings(model="embedding-2",
                                                  api_key="67f836edc96405d0c4eea5d8eeff70d0.pVev8zx1tG17ohTQ",
                                                  openai_api_base="https://open.bigmodel.cn/api/paas/v4/")
        # print("graph_type:",graph_type)
        EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
        EMBEDDING_FUNCTION, _ = load_embedding_model(EMBEDDING_MODEL)
        neo_db = Neo4jVector.from_existing_index(
            embedding=EMBEDDING_FUNCTION,
            index_name=index_name,
            retrieval_query=retrieval_query,
            graph=graph
        )
        logging.info(f"Successfully retrieved Neo4jVector index '{index_name}'")
        document_names= list(map(str.strip, json.loads(document_names)))

        if document_names:
            retriever = neo_db.as_retriever(search_kwargs={'k': search_k, "score_threshold": score_threshold,'filter':{'fileName': {'$in': document_names}}})
            logging.info(f"Successfully created retriever for index '{index_name}' with search_k={search_k}, score_threshold={score_threshold} for documents {document_names}")
        else:
            retriever = neo_db.as_retriever(search_kwargs={'k': search_k, "score_threshold": score_threshold})
            logging.info(f"Successfully created retriever for index '{index_name}' with search_k={search_k}, score_threshold={score_threshold}")
        return retriever, EMBEDDING_FUNCTION
    except Exception as e:
        logging.error(f"Error retrieving Neo4jVector index '{index_name}' or creating retriever: {e}")
        return None

def create_document_retriever_chain(llm,retriever, EMBEDDING_FUNCTION):
    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUESTION_TRANSFORM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    output_parser = StrOutputParser()

    splitter = TokenTextSplitter(chunk_size=CHAT_DOC_SPLIT_SIZE, chunk_overlap=0)
    embeddings_filter = EmbeddingsFilter(embeddings=EMBEDDING_FUNCTION, similarity_threshold=CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD)

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, embeddings_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | compression_retriever,
        ),
        query_transform_prompt | llm | output_parser | compression_retriever,
    ).with_config(run_name="chat_retriever_chain")

    return query_transforming_retriever_chain


def summarize_and_log(history, messages, llm):
    start_time = time.time()
    summarize_messages(llm, history, messages)
    history_summarized_time = time.time() - start_time
    logging.info(f"Chat History summarized in {history_summarized_time:.2f} seconds")


def summarize_messages(llm,history,stored_messages):
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Summarize the above chat messages into a concise message, focusing on key points and relevant details that could be useful for future conversations. Exclude all introductions and extraneous information."
            ),
        ]
    )

    summarization_chain = summarization_prompt | llm

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    history.clear()
    history.add_user_message("Our current convertaion summary till now")
    history.add_message(summary_message)
    return True


def retrieve_documents(doc_retriever, question):
    start_time = time.time()
    docs = doc_retriever.invoke(question)
    doc_retrieval_time = time.time() - start_time
    logging.info(f"Documents retrieved in {doc_retrieval_time:.2f} seconds")
    return docs


store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def generate_response(history_list, information, session_id="1",model="深度求索"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_pro),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    # model = os.environ.get('default_model')
    llm, _ = get_llm(model)
    # Set up the language model and memory chain
    chain = prompt | llm
    chain_with_message = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="question",
                                                         history_messages_key="history", )
    # 获取当前会话的历史记录
    # history = get_session_history(session_id)

    # 生成响应
    history_list[-1][1] = ""
    # 提取响应内容
    for chunk in chain_with_message.stream({"question": history_list[-1][0], "info": information},
                                                config={"configurable": {"session_id": session_id}}):
        history_list[-1][1] += chunk.content
        yield history_list

import asyncio

# 假设 RAG 函数已经定义好了

if __name__ == "__main__":
    # 使用 asyncio.run() 来调用并等待异步函数 RAG 完成
    result = asyncio.run(RAG("神经网络的发展历程"))
    print(result["message"][0].page_content)