from google import genai
from google.genai import types
from config import GEMINI_MODEL


def get_client(api_key: str) -> genai.Client:
    """Create and return a Gemini client."""
    return genai.Client(api_key=api_key)


def call_gemini(client, prompt: str, max_tokens: int = 1000) -> str:
    """Call Gemini using the google-genai SDK."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
    )
    return response.text.strip()
