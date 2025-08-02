import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Create or get collection
collection = client.get_or_create_collection(
    name="test_collection",
    embedding_function=embedding_fn
)

# Insert documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

# Query for similar documents
query = "Age of the Earth"
results = collection.query(query_texts=[query], n_results=2)

# Display results
print(f"\nQuery: {query}")
for idx, doc in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    score = results["distances"][0][idx]
    print(f"- Match: \"{doc}\" (ID: {doc_id}, Distance: {score:.4f})")
