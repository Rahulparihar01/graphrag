import re
import json
from gemini_client import call_gemini


def extract_entities_and_relations(chunks: list[dict], client) -> list[dict]:
    """Extract named entities and relationships from each chunk using Gemini."""
    print(f"🔍 Extracting entities from {len(chunks)} chunks...")
    all_extractions = []

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...", end="\r")

        prompt = f"""Extract entities and relationships from the text below.

Text: {chunk['text']}

Reply with ONLY valid JSON, no markdown, no extra text:
{{"entities":[{{"name":"...","type":"PERSON|ORG|CONCEPT|PLACE|EVENT","description":"..."}}],"relationships":[{{"source":"...","relation":"...","target":"..."}}]}}

Keep descriptions under 10 words. Extract max 8 entities and 6 relationships."""

        try:
            raw = call_gemini(client, prompt, max_tokens=2048)
            clean = re.sub(r"```json|```", "", raw).strip()

            # Attempt to salvage truncated JSON
            if not clean.endswith("}"):
                open_braces   = clean.count("{") - clean.count("}")
                open_brackets = clean.count("[") - clean.count("]")
                if clean.count('"') % 2 != 0:
                    clean += '"'
                clean += "}" * max(0, open_braces - open_brackets)
                clean += "]" * max(0, open_brackets)
                clean += "}" * max(0, open_braces - max(0, open_braces - open_brackets))

            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")

            data = json.loads(match.group())
            data.setdefault("entities", [])
            data.setdefault("relationships", [])
            data["chunk_id"]   = chunk["id"]
            data["chunk_text"] = chunk["text"]

            print(f"  ✅ Chunk {i+1}: {len(data['entities'])} entities found", end="\r")
            all_extractions.append(data)

        except Exception as e:
            print(f"\n  ⚠️  Chunk {i+1} failed: {e}")
            all_extractions.append({
                "chunk_id":      chunk["id"],
                "chunk_text":    chunk["text"],
                "entities":      [],
                "relationships": [],
            })

    print(f"\n✅ Entity extraction complete")
    return all_extractions
