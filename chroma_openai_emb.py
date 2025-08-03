import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("Missing OPENAI_API_KEY in .env file")

# Initialize ChromaDB client and embedding function
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="./db/chroma_persist")

# Get or create the collection
collection = chroma_client.get_or_create_collection(
    name="my_story",
    embedding_function=embedding_fn
)

# Define documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {"id": "doc4", "text": "Microsoft is a technology company that develops software. It was founded by Bill Gates and Paul Allen in 1975."},
    {"id": "doc5", "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions."},
    {"id": "doc6", "text": "Machine Learning (ML) is a subset of AI that focuses on the development of algorithms that allow computers to learn from and make predictions based on data."},
    {"id": "doc7", "text": "Deep Learning is a subset of Machine Learning that uses neural networks with many layers to analyze various factors of data."},
    {"id": "doc8", "text": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and respond to human language."},
    {"id": "doc9", "text": "AI can be categorized into two types: Narrow AI, which is designed to perform a narrow task, and General AI, which can perform any intellectual task that a human can do."},
    {"id": "doc10", "text": "Computer Vision is a field of AI that enables computers to interpret and make decisions based on visual data from the world."},
    {"id": "doc11", "text": "Reinforcement Learning is an area of Machine Learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward."},
    {"id": "doc12", "text": "The Turing Test, proposed by Alan Turing, is a measure of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human."},
]

# Upsert documents
for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

# Perform semantic query
query_text = "find document related to Turing Test"
results = collection.query(query_texts=[query_text], n_results=3)

# Display results
print(f"\nQuery: \"{query_text}\"\n")
for idx, (doc_id, doc_text, distance) in enumerate(
    zip(results["ids"][0], results["documents"][0], results["distances"][0]), 1
):
    print(f"Result {idx}:\n  ID       : {doc_id}\n  Text     : {doc_text}\n  Distance : {distance:.4f}\n")
