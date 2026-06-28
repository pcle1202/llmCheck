import os
import time
from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def query_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> dict:
    start = time.time()
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        latency = time.time() - start
        return {
            "text": response.choices[0].message.content,
            "latency": round(latency, 3),
            "error": None
        }
    except Exception as e:
        return {"text": None, "latency": None, "error": str(e)}

def query_gemini(prompt: str, model: str = "gemini-2.0-flash") -> dict:
    start = time.time()
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt
        )
        latency = time.time() - start
        return {
            "text": response.text,
            "latency": round(latency, 3),
            "error": None
        }
    except Exception as e:
        return {"text": None, "latency": None, "error": str(e)}

MODELS = {
    "groq": query_groq,
    "gemini": query_gemini
}