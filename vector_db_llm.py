import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# === Load OpenAI Key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in .env file")

# === Initialize Clients ===
openai_client = OpenAI(api_key=api_key)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="./db/chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="article_embeddings", embedding_function=embedding_fn
)

# === Step 1: Load and Chunk Documents ===
def load_txt_documents(path: str):
    print("📁 Loading .txt files from:", path)
    docs = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                docs.append({"id": file, "text": f.read()})
    return docs

def chunk_text(text, size=1000, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# === Step 2: Embed and Store Chunks ===
def embed_and_store_documents(dir_path="./data/new_articles"):
    raw_docs = load_txt_documents(dir_path)
    all_chunks = []

    for doc in raw_docs:
        print(f"🧩 Chunking {doc['id']}...")
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk{idx+1}"
            print(f"🔗 Embedding {chunk_id}")
            embedding = get_embedding(chunk)
            collection.upsert(ids=[chunk_id], documents=[chunk], embeddings=[embedding])

def get_embedding(text):
    res = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return res.data[0].embedding

# === Step 3: Query Relevant Chunks ===
def retrieve_relevant_chunks(query, top_k=3):
    print(f"🔍 Searching for: {query}")
    results = collection.query(query_texts=[query], n_results=top_k)
    return [doc for doc_group in results["documents"] for doc in doc_group]

# === Step 4: Generate Final Answer ===
def answer_question(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a concise Q&A assistant. Answer based on the following context. "
        "If you don't know the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}"
    )

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
    )
    return response.choices[0].message.content

# === Run Flow ===
if __name__ == "__main__":
    # Uncomment below line on first run to ingest
    # embed_and_store_documents()

    user_question = "give me a brief overview of the articles. Be concise, please."
    chunks = retrieve_relevant_chunks(user_question)
    final_answer = answer_question(user_question, chunks)

    print("\n🧠 Final Answer:")
    print(final_answer)
