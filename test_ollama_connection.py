"""Quick local model check before running the four-agent workflow."""
import os
import json
from dotenv import load_dotenv
from litellm import completion

load_dotenv()
model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
url = os.getenv("OLLAMA_URL", "http://localhost:11434")

print(f"Testing Ollama model: {model}")
answer = completion(
    model=f"ollama_chat/{model}",
    api_base=url,
    messages=[{"role": "user", "content": "Return only valid JSON with exactly two keys: status as OK and amount as the number 100."}],
    response_format={"type": "json_object"},
    temperature=0,
)
content = answer.choices[0].message.content
print("Raw response:", content)
parsed = json.loads(content)
assert parsed["status"] == "OK" and float(parsed["amount"]) == 100
print("OLLAMA + LiteLLM JSON TEST: OK")
