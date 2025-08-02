from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Initialize embedding function
embedding_fn = DefaultEmbeddingFunction()

# Input text
text = "John"

# Generate embedding (input must be a list of strings)
embedding = embedding_fn([text])

# Output the result
print(f"Embedding for '{text}':\n{embedding[0]}")
