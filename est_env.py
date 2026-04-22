import os
from dotenv import load_dotenv

load_dotenv()

# Check if keys are loaded
print("GROQ_API_KEY_1:", os.getenv("GROQ_API_KEY_1")[:10] + "..." if os.getenv("GROQ_API_KEY_1") else "NOT FOUND")
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY")[:10] + "..." if os.getenv("GROQ_API_KEY") else "NOT FOUND")

# List all env vars with GROQ
for key, value in os.environ.items():
    if "GROQ" in key:
        print(f"{key}: {value[:10]}...")