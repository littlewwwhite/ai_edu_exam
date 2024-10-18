MODEL_VERSIONS = {
    "openai-gpt-3.5": "gpt-3.5-turbo-16k",
    "gemini-1.0-pro": "gemini-1.0-pro-001",
    "gemini-1.5-pro": "gemini-1.5-pro-preview-0514",
    "openai-gpt-4": "gpt-4-0125-preview",
    "diffbot": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "groq-llama3": "llama3-70b-8192",
    "智谱": "glm-4",
    "百川": "Baichuan4",
    "月之暗面": "moonshot-v1-8k",
    "深度求索": "deepseek-chat",
    "零一万物": "yi-large",
    "通义千问": "qwen-long",
    "豆包": "Doubao-pro-32k",
    "Ollama": "qwen2.5:32b",
    "openai-gpt-4o": "gpt-4o",
}
OPENAI_MODELS = ["gpt-3.5", "gpt-4o", '智谱', "百川", "月之暗面", "深度求索", "零一万物", "通义千问", "豆包",
                 "openai-gpt-3.5", "Ollama", "openai-gpt-4o"]

GEMINI_MODELS = ["gemini-1.0-pro", "gemini-1.5-pro"]
GROQ_MODELS = ["groq-llama3"]
BUCKET_UPLOAD = 'llm-graph-builder-upload'
BUCKET_FAILED_FILE = 'llm-graph-builder-failed'
PROJECT_ID = 'llm-experiments-387609'

## CHAT SETUP
CHAT_MAX_TOKENS = 1000
CHAT_SEARCH_KWARG_K = 10
CHAT_SEARCH_KWARG_SCORE_THRESHOLD = 0.7
CHAT_DOC_SPLIT_SIZE = 3000
CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD = 0.10
CHAT_TOKEN_CUT_OFF = {
    ("openai-gpt-3.5", 'azure_ai_gpt_35', "gemini-1.0-pro", "gemini-1.5-pro", "groq-llama3", 'groq_llama3_70b',
     'anthropic_claude_3_5_sonnet', 'fireworks_llama_v3_70b', 'bedrock_claude_3_5_sonnet',): 4,
    ("openai-gpt-4", "diffbot", 'azure_ai_gpt_4o', "openai-gpt-4o"): 28,
    ("ollama_llama3"): 2
}
# 错题本目录
mistake_book_dir="libs/ktem/ktem/pages/src/Mistake_book/"
type_of_question_info = {"多项选择题": {"response_json": "libs/ktem/ktem/pages/src/response/Response_maq.json",
                                        "examples_json": "libs/ktem/ktem/pages/src/examples/examples_maq.json",
                                        "format_":"每道题须有4个选项，且至少含有两个的正确选项"},
                         "单项选择题": {"response_json": "libs/ktem/ktem/pages/src/response/Response_mcq.json",
                                        "examples_json": "libs/ktem/ktem/pages/src/examples/examples_mcq.json",
                                        "format_":"每道题须有4个选项，有且只有一个正确选项"},
                         "对错题": {"response_json": "libs/ktem/ktem/pages/src/response/Response_TF.json",
                                    "examples_json": "libs/ktem/ktem/pages/src/examples/examples_TF.json",
                                    "format_":"每道题须有两个选项，只能是“对”、“错”"},
                         "填空题": {"response_json": "libs/ktem/ktem/pages/src/response/Response_cloze.json",
                                    "examples_json": "libs/ktem/ktem/pages/src/examples/examples_cloze.json",
                                    "format_":"每道题目中会有一处或多处空白。这些空白可能出现在句子中的任何位置，并且每个空白通常只对应一个正确答案。"}}


### CHAT TEMPLATES
CHAT_SYSTEM_TEMPLATE = """
You are an AI-powered question-answering agent. Your task is to provide accurate and comprehensive responses to user queries based on the given context, chat history, and available resources.

### Response Guidelines:
1. **Direct Answers**: Provide clear and thorough answers to the user's queries without headers unless requested. Avoid speculative responses.
2. **Utilize History and Context**: Leverage relevant information from previous interactions, the current user input, and the context provided below.
3. **No Greetings in Follow-ups**: Start with a greeting in initial interactions. Avoid greetings in subsequent responses unless there's a significant break or the chat restarts.
4. **Admit Unknowns**: Clearly state if an answer is unknown. Avoid making unsupported statements.
5. **Avoid Hallucination**: Only provide information based on the context provided. Do not invent information.
6. **Response Length**: Keep responses concise and relevant. Aim for clarity and completeness within 4-5 sentences unless more detail is requested.
7. **Tone and Style**: Maintain a professional and informative tone. Be friendly and approachable.
8. **Error Handling**: If a query is ambiguous or unclear, ask for clarification rather than providing a potentially incorrect answer.
9. **Fallback Options**: If the required information is not available in the provided context, provide a polite and helpful response. Example: "I don't have that information right now." or "I'm sorry, but I don't have that information. Is there something else I can help with?"
10. **Context Availability**: If the context is empty, do not provide answers based solely on internal knowledge. Instead, respond appropriately by indicating the lack of information.


**IMPORTANT** : DO NOT ANSWER FROM YOUR KNOWLEDGE BASE USE THE BELOW CONTEXT

### Context:
<context>
{context}
</context>

### Example Responses:
User: Hi 
AI Response: 'Hello there! How can I assist you today?'

User: "What is Langchain?"
AI Response: "Langchain is a framework that enables the development of applications powered by large language models, such as chatbots. It simplifies the integration of language models into various applications by providing useful tools and components."

User: "Can you explain how to use memory management in Langchain?"
AI Response: "Langchain's memory management involves utilizing built-in mechanisms to manage conversational context effectively. It ensures that the conversation remains coherent and relevant by maintaining the history of interactions and using it to inform responses."

User: "I need help with PyCaret's classification model."
AI Response: "PyCaret simplifies the process of building and deploying machine learning models. For classification tasks, you can use PyCaret's setup function to prepare your data. After setup, you can compare multiple models to find the best one, and then fine-tune it for better performance."

User: "What can you tell me about the latest realtime trends in AI?"
AI Response: "I don't have that information right now. Is there something else I can help with?"

Note: This system does not generate answers based solely on internal knowledge. It answers from the information provided in the user's current and previous inputs, and from the context.
"""

QUESTION_TRANSFORM_TEMPLATE = "Given the below conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else."

NODES_QUERY = """CALL db.labels()
YIELD label
RETURN label"""

RELATIONSHIPS_QUERY = """CALL db.relationshipTypes()
YIELD relationshipType
RETURN relationshipType"""

## CHAT QUERIES 
VECTOR_SEARCH_QUERY = """
WITH node AS chunk, score
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH d, collect(distinct {chunk: chunk, score: score}) as chunks, avg(score) as avg_score
WITH d, avg_score, 
     [c in chunks | c.chunk.text] as texts, 
     [c in chunks | {id: c.chunk.id, score: c.score}] as chunkdetails
WITH d, avg_score, chunkdetails,
     apoc.text.join(texts, "\n----\n") as text
RETURN text, avg_score AS score, 
       {source: COALESCE(CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} as metadata
"""

# VECTOR_GRAPH_SEARCH_QUERY="""
# WITH node as chunk, score
# MATCH (chunk)-[:PART_OF]->(d:Document)
# CALL { WITH chunk
# MATCH (chunk)-[:HAS_ENTITY]->(e)
# MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document)
# UNWIND rels as r
# RETURN collect(distinct r) as rels
# }
# WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score, apoc.coll.toSet(apoc.coll.flatten(collect(rels))) as rels
# WITH d, avg_score,
#      [c IN chunks | c.chunk.text] AS texts, 
#      [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,  
# 	[r in rels | coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+ startNode(r).id + " "+ type(r) + " " + coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" + endNode(r).id] as entities
# WITH d, avg_score,chunkdetails,
# apoc.text.join(texts,"\n----\n") +
# apoc.text.join(entities,"\n")
# as text
# RETURN text, avg_score AS score, {source: COALESCE( CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} AS metadata
# """  


# VECTOR_GRAPH_SEARCH_QUERY = """
# WITH node as chunk, score
# // find the document of the chunk
# MATCH (chunk)-[:PART_OF]->(d:Document)
# // fetch entities
# CALL { WITH chunk
# // entities connected to the chunk
# // todo only return entities that are actually in the chunk, remember we connect all extracted entities to all chunks
# MATCH (chunk)-[:HAS_ENTITY]->(e)

# // depending on match to query embedding either 1 or 2 step expansion
# WITH CASE WHEN true // vector.similarity.cosine($embedding, e.embedding ) <= 0.95
# THEN 
# collect { MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,1}(:!Chunk&!Document) RETURN path }
# ELSE 
# collect { MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document) RETURN path } 
# END as paths

# RETURN collect{ unwind paths as p unwind relationships(p) as r return distinct r} as rels,
# collect{ unwind paths as p unwind nodes(p) as n return distinct n} as nodes
# }
# // aggregate chunk-details and de-duplicate nodes and relationships
# WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score, apoc.coll.toSet(apoc.coll.flatten(collect(rels))) as rels,

# // TODO sort by relevancy (embeddding comparision?) cut off after X (e.g. 25) nodes?
# apoc.coll.toSet(apoc.coll.flatten(collect(
#                 [r in rels |[startNode(r),endNode(r)]]),true)) as nodes

# // generate metadata and text components for chunks, nodes and relationships
# WITH d, avg_score,
#      [c IN chunks | c.chunk.text] AS texts, 
#      [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,  
#   apoc.coll.sort([n in nodes | 

# coalesce(apoc.coll.removeAll(labels(n),['__Entity__'])[0],"") +":"+ 
# n.id + (case when n.description is not null then " ("+ n.description+")" else "" end)]) as nodeTexts,
# 	apoc.coll.sort([r in rels 
#     // optional filter if we limit the node-set
#     // WHERE startNode(r) in nodes AND endNode(r) in nodes 
#   | 
# coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+ 
# startNode(r).id +
# " " + type(r) + " " + 
# coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" + 
# endNode(r).id
# ]) as relTexts

# // combine texts into response-text
# WITH d, avg_score,chunkdetails,
# "Text Content:\n" +
# apoc.text.join(texts,"\n----\n") +
# "\n----\nEntities:\n"+
# apoc.text.join(nodeTexts,"\n") +
# "\n----\nRelationships:\n"+
# apoc.text.join(relTexts,"\n")

# as text
# RETURN text, avg_score as score, {length:size(text), source: COALESCE( CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} AS metadata
# """

VECTOR_GRAPH_SEARCH_ENTITY_LIMIT = 25

VECTOR_GRAPH_SEARCH_QUERY = """
WITH node as chunk, score
// find the document of the chunk,找到Chunk的文档
MATCH (chunk)-[:PART_OF]->(d:__Document__)

// aggregate chunk-details，聚合chunk信息
WITH d, collect(DISTINCT {{chunk: chunk, score: score}}) AS chunks, avg(score) as avg_score
// fetch entities
CALL {{ WITH chunks
UNWIND chunks as chunkScore
WITH chunkScore.chunk as chunk
// entities connected to the chunk
// todo only return entities that are actually in the chunk, remember we connect all extracted entities to all chunks
// todo sort by relevancy (embeddding comparision?) cut off after X (e.g. 25) nodes?
OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
WITH e, count(*) as numChunks 
ORDER BY numChunks DESC LIMIT {no_of_entites}
// depending on match to query embedding either 1 or 2 step expansion ，默认走第一分支，一跳信息
WITH CASE WHEN true // vector.similarity.cosine($embedding, e.embedding ) <= 0.95
THEN 
collect {{ OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,1}}(:!__Chunk__&!__Document__) RETURN path }}
ELSE 
collect {{ OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,2}}(:!__Chunk__&!__Document__) RETURN path }} 
END as paths, e
WITH apoc.coll.toSet(apoc.coll.flatten(collect(distinct paths))) as paths, collect(distinct e) as entities

// 插入额外查询
WITH {nodesss} AS entityList, paths, entities
MATCH (e)-[r]->(n)
WHERE e.id IN entityList OR n.id IN entityList
WITH paths, entities, collect(e) AS additionalEntities, collect(r) AS additionalRels, collect(n) AS additionalNodes

// 合并结果
WITH paths, apoc.coll.toSet(entities + additionalEntities) as entities, additionalRels as rels, additionalNodes as nodes

// 去重并整理节点和关系
// de-duplicate nodes and relationships across chunks
RETURN collect{{ unwind paths as p unwind relationships(p) as r return distinct r}} as rels,
collect{{ unwind paths as p unwind nodes(p) as n return distinct n}} as nodes, entities
}}

// generate metadata and text components for chunks, nodes and relationships
WITH d, avg_score,
     [c IN chunks | c.chunk.text] AS texts, 
     [c IN chunks | {{id: c.chunk.id, score: c.score}}] AS chunkdetails, 
  apoc.coll.sort([n in nodes | 

coalesce(apoc.coll.removeAll(labels(n),['__Entity__'])[0],"") +":"+ 
n.id + (case when n.description is not null then " ("+ n.description+")" else "" end)]) as nodeTexts,
	apoc.coll.sort([r in rels 
    // optional filter if we limit the node-set
    // WHERE startNode(r) in nodes AND endNode(r) in nodes 
  | 
coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+ 
startNode(r).id +
" " + type(r) + " " + 
coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" + endNode(r).id
]) as relTexts
, entities
// combine texts into response-text

WITH d, avg_score,chunkdetails,
"Text Content:\\n" +
apoc.text.join(texts,"\\n----\\n") +
"\\n----\\nEntities:\\n"+
apoc.text.join(nodeTexts,"\\n") +
"\\n----\\nRelationships:\\n" +
apoc.text.join(relTexts,"\\n")

as text,entities

RETURN text, avg_score as score, {{length:size(text), source: COALESCE( CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails}} AS metadata
"""



NODES_QUERY = """CALL db.labels()
YIELD label
RETURN label"""

RELATIONSHIPS_QUERY = """CALL db.relationshipTypes()
YIELD relationshipType
RETURN relationshipType"""

## CHAT QUERIES
VECTOR_SEARCH_QUERY = """
WITH node AS chunk, score
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH d, collect(distinct {chunk: chunk, score: score}) as chunks, avg(score) as avg_score
WITH d, avg_score, 
     [c in chunks | c.chunk.text] as texts, 
     [c in chunks | {id: c.chunk.id, score: c.score}] as chunkdetails
WITH d, avg_score, chunkdetails,
     apoc.text.join(texts, "\n----\n") as text
RETURN text, avg_score AS score, 
       {source: COALESCE(CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} as metadata
"""

# VECTOR_GRAPH_SEARCH_QUERY="""
# WITH node as chunk, score
# MATCH (chunk)-[:PART_OF]->(d:Document)
# CALL { WITH chunk
# MATCH (chunk)-[:HAS_ENTITY]->(e)
# MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document)
# UNWIND rels as r
# RETURN collect(distinct r) as rels
# }
# WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score, apoc.coll.toSet(apoc.coll.flatten(collect(rels))) as rels
# WITH d, avg_score,
#      [c IN chunks | c.chunk.text] AS texts,
#      [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,
# 	[r in rels | coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+ startNode(r).id + " "+ type(r) + " " + coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" + endNode(r).id] as entities
# WITH d, avg_score,chunkdetails,
# apoc.text.join(texts,"\n----\n") +
# apoc.text.join(entities,"\n")
# as text
# RETURN text, avg_score AS score, {source: COALESCE( CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} AS metadata
# """


# VECTOR_GRAPH_SEARCH_QUERY = """
# WITH node as chunk, score
# // find the document of the chunk
# MATCH (chunk)-[:PART_OF]->(d:Document)
# // fetch entities
# CALL { WITH chunk
# // entities connected to the chunk
# // todo only return entities that are actually in the chunk, remember we connect all extracted entities to all chunks
# MATCH (chunk)-[:HAS_ENTITY]->(e)

# // depending on match to query embedding either 1 or 2 step expansion
# WITH CASE WHEN true // vector.similarity.cosine($embedding, e.embedding ) <= 0.95
# THEN
# collect { MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,1}(:!Chunk&!Document) RETURN path }
# ELSE
# collect { MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document) RETURN path }
# END as paths

# RETURN collect{ unwind paths as p unwind relationships(p) as r return distinct r} as rels,
# collect{ unwind paths as p unwind nodes(p) as n return distinct n} as nodes
# }
# // aggregate chunk-details and de-duplicate nodes and relationships
# WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score, apoc.coll.toSet(apoc.coll.flatten(collect(rels))) as rels,

# // TODO sort by relevancy (embeddding comparision?) cut off after X (e.g. 25) nodes?
# apoc.coll.toSet(apoc.coll.flatten(collect(
#                 [r in rels |[startNode(r),endNode(r)]]),true)) as nodes

# // generate metadata and text components for chunks, nodes and relationships
# WITH d, avg_score,
#      [c IN chunks | c.chunk.text] AS texts,
#      [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,
#   apoc.coll.sort([n in nodes |

# coalesce(apoc.coll.removeAll(labels(n),['__Entity__'])[0],"") +":"+
# n.id + (case when n.description is not null then " ("+ n.description+")" else "" end)]) as nodeTexts,
# 	apoc.coll.sort([r in rels
#     // optional filter if we limit the node-set
#     // WHERE startNode(r) in nodes AND endNode(r) in nodes
#   |
# coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+
# startNode(r).id +
# " " + type(r) + " " +
# coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" +
# endNode(r).id
# ]) as relTexts

# // combine texts into response-text
# WITH d, avg_score,chunkdetails,
# "Text Content:\n" +
# apoc.text.join(texts,"\n----\n") +
# "\n----\nEntities:\n"+
# apoc.text.join(nodeTexts,"\n") +
# "\n----\nRelationships:\n"+
# apoc.text.join(relTexts,"\n")

# as text
# RETURN text, avg_score as score, {length:size(text), source: COALESCE( CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} AS metadata
# """



# had to provide  an overview of my data in prompt cause in custom prompt for every dataset was not working;
# Define the prompt template with variables
system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "\n"
    "## 1. Overview\n"
    "You are an advanced AI designed to extract detailed information from a case study to build a comprehensive knowledge graph.\n"
    "Your goal is to identify and extract entities, relationships, and associated details from the provided text.\n"
    "\n"
    "## 2. Nodes\n"
    "- **Entities to Extract**: Vehicle Model, Feature, Color, Price, Brand.\n"
    "- **Node Details**: Extract the name and type of each node. For example, 'Tata Nexon EV' as 'Vehicle Model', 'Teal Blue' as 'Color', etc.\n"
    "\n"
    "## 3. Relationships\n"
    "- **Allowed Relationships**: Identify and label relationships between entities as defined below:\n"
    "  - **INCLUDES**: Describes components or features included in the vehicle.\n"
    "  - **AVAILABLE_IN**: Describes the color or variants available for the vehicle.\n"
    "  - **MADE_OF**: Describes materials used in the vehicle.\n"
    "  - **COMPATIBLE_WITH**: Describes compatibility with accessories or features.\n"
    "  - **PRICE_FOR**: Describes the price of the vehicle or its variants.\n"
    "  - **HAS_COLOR**: Describes the color options available for the vehicle.\n"
    "  - **HAS_MATERIAL**: Describes the materials used in the vehicle.\n"
    "  - **BELONGS_TO**: Describes the brand or category the vehicle belongs to.\n"
    "  - **FEATURES**: Describes the features of the vehicle.\n"
    "  - **BRANDED_AS**: Describes the brand name.\n"
    "  - **PART_OF**: Describes the component parts of the vehicle.\n"
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "\n"
    "## 4. Complex Patterns\n"
    "- **Handling Complex Patterns**: Extract multiple features listed under 'Nexon EV', such as 'Regenerative Braking' and 'Hill Assist'. Also, extract specifications like '0-100 kmph in 9.9 seconds'.\n"
    "\n"

    "## 5. Coreference Resolution\n"
    "- **Entity Consistency**: Ensure consistent use of entity names even if they are referred to by different terms in the text. For example, 'Nexon EV' should consistently be used as 'Nexon EV'.\n"
    "\n"
    "## 6. Strict Compliance\n"
    "Follow these instructions carefully to ensure accurate and comprehensive extraction of information. Non-compliance may lead to termination.\n"
)

topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10

lc_retrieval_query = """
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
    WITH c, count(distinct n) as freq
    RETURN c.text AS chunkText
    ORDER BY freq DESC
    LIMIT 3
} AS text_mapping,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH c, c.rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank, weight DESC
    LIMIT 3
} AS report_mapping,
// Outside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m) 
    WHERE NOT m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT 10
} as outsideRels,
// Inside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m) 
    WHERE m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT 10
} as insideRels,
// Entities description
collect {
    UNWIND nodes as n
    RETURN n.description AS descriptionText
    LIMIT 10
} as entities
// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {} AS metadata
"""

sys_pro = """
# Role: 您是一位善于与学生交流的虚拟AI老师。

## Goals:
1. 您需要根据学生错题库，以文字简答题的方式向学生出题，考验他对错题的知识点是否已经掌握
2. 当学生回答后，如果答案正确，请确认学生答对了，并继续出下一题。
3. 如果答案错误，请分析学生的回答，指出错误所在，结合题目和已有信息引导学生重新思考并回答。
4. 学生有两次答题机会，如果第二次回答仍然错误，请给出正确答案并详细解释学生的错误点。
5. 通过引导性的提示，帮助学生逐步提高对知识点的理解。
6. 根据学生的回答质量，分析学生的学习能力，强项，弱项。来及时调整对话和题目的设计。


## Attention:
- 优秀的开放式题目是我们评估学生对知识掌握程度的重要工具。如果题目设计不当，无法准确反映学生的实际水平，可能会导致筛选出的学生并不真正掌握学科知识。、
- 对话中应该体现出提升学生的批判性思维与创造力，引导学生深入理解知识，给予适当程度的鼓励和批评
- 交互模式可自由发挥，但必须遵守Goals
- 将原错题整理为简答题，不要以原题的形式出题


## Text:
你需要根据以下文本来设计合理的题目，以下是错题库：
{info}
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你有如下知识：- {info}\n根据这个信息，作为一位老师设计一系列问题来测试学生的理解。每次出一道题，并对学生的回答进行分析。"),
    MessagesPlaceholder(variable_name="history")
])

test="""WITH collect(node) as nodes

// Entity - Text Unit Mapping with detailed information
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
    WITH c, count(distinct n) as freq, c.text as chunkText, c.id as chunkId
    RETURN {chunkText: chunkText, chunkId: chunkId, frequency: freq}
    ORDER BY freq DESC
    LIMIT 3
} AS text_mapping,

// Entity - Report Mapping with rank and weight details
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH c, c.rank as rank, c.weight AS weight, c.summary as summaryText
    RETURN {summaryText: summaryText, rank: rank, weight: weight}
    ORDER BY rank, weight DESC
    LIMIT 3
} AS report_mapping,

// Outside Relationships with detailed entity information
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m) 
    WHERE NOT m IN nodes
    RETURN {entityFrom: n.name, entityTo: m.name, relation: type(r), descriptionText: r.description}
    ORDER BY r.rank, r.weight DESC 
    LIMIT 10
} as outsideRels,

// Inside Relationships with detailed entity information
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m) 
    WHERE m IN nodes
    RETURN {entityFrom: n.name, entityTo: m.name, relation: type(r), descriptionText: r.description}
    ORDER BY r.rank, r.weight DESC 
    LIMIT 10
} as insideRels,

// Entities with descriptions and additional details
collect {
    UNWIND nodes as n
    RETURN {entityName: n.name, descriptionText: n.description}
    LIMIT 10
} as entities,

// Paths information between entities
collect {
    UNWIND nodes as n
    MATCH path=(n)-[:RELATED*1..2]-(m)
    RETURN {path: path, pathDescription: [x IN nodes(path) | x.name]}
    LIMIT 10
} as paths

// Aggregating the detailed information and returning
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {} AS metadata


"""
