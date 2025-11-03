import os
from groq import Groq

# Get API key from environment, NOT hardcoded
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not set in environment")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

# List available models
models = client.models.list()

for model in models.data:
    print(f"Model: {model.id}")
