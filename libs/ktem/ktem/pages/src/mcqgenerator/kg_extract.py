from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.docstore.document import Document


relationship_properties = ["comprehensiveDescriptionOfTheRelationship"]
node_properties = ["comprehensiveDescriptionOfTheEntity"]
allowedRelationship = [
    "RELATED_TO"
]
allowedNodes = [
    "Discipline",
    "SubDiscipline",
    "Concept",
    "Theory",
    "Method",
    "Technology",
    "Tool",
    "Person",
    "Event",
    "Institution",
    "Publication",
    "Application",
    "Experiment",
    "KnowledgePoint",
]