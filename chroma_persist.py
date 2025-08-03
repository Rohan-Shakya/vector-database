import chromadb
from chromadb.utils import embedding_functions

# Initialize embedding function and persistent client
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./db/chroma_persist")

# Get or create the collection
collection = chroma_client.get_or_create_collection(
    name="my_story", embedding_function=embedding_fn
)

# Define text documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {
        "id": "doc4",
        "text": (
            "Microsoft is a technology company that develops software. "
            "It was founded by Bill Gates and Paul Allen in 1975."
        ),
    },
]

# Upsert documents into the collection
for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

# Define the query text
query_text = "find document related to technology company"

# Perform similarity search
results = collection.query(
    query_texts=[query_text],
    n_results=2,
)

# Display the results
for idx, (doc_id, doc_text, distance) in enumerate(
    zip(results["ids"][0], results["documents"][0], results["distances"][0])
):
    print(
        f"For the query: '{query_text}',\n"
        f"Found similar document [{idx + 1}]:\n"
        f"  ID       : {doc_id}\n"
        f"  Content  : {doc_text}\n"
        f"  Distance : {distance:.4f}\n"
    )
