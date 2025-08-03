import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")

openai_client = OpenAI(api_key=api_key)

# Define input text and embedding model
text_input = "Your text string goes here"
model_name = "text-embedding-3-small"

# Generate embedding
embedding_response = openai_client.embeddings.create(
    input=text_input,
    model=model_name
)

# Print embedding result
print(embedding_response)
