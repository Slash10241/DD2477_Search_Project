from google import genai
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from current directory

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
print(os.environ.get("GEMINI_API_KEY"))

prompt = "Return JSON: {\"test\": \"ok\"}"

response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": {
            "type": "object",
            "properties": {
                "test": {"type": "string"}
            },
            "required": ["test"]
        },
    },
)

print(response.text)