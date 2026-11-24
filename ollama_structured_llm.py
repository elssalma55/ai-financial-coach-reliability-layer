"""Ollama adapter for structured responses inside the Google ADK workflow.

Google ADK still runs the original LlmAgent/SequentialAgent objects. This
adapter only controls the transport and enforces their Pydantic output schema.
"""
import logging
from typing import AsyncGenerator

import httpx
from pydantic import BaseModel
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

logger = logging.getLogger(__name__)


def _text_from_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "\n".join(
        part.text for part in (getattr(content, "parts", None) or [])
        if getattr(part, "text", None)
    )


class OllamaStructuredLlm(BaseLlm):
    base_url: str = "http://localhost:11434"
    schema_type: type[BaseModel]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # Use the schema already defined for this agent, not a new prompt format.
        schema = self.schema_type.model_json_schema()
        system_text = _text_from_content(
            getattr(llm_request.config, "system_instruction", None)
        )
        system_text += (
            "\nReturn only a JSON object matching the provided schema. "
            "Do not use Markdown or explanatory text outside the JSON. "
            "Preserve numeric values as JSON numbers."
        )
        messages = [{"role": "system", "content": system_text}]
        for content in llm_request.contents:
            body = _text_from_content(content)
            if body:
                role = "assistant" if content.role == "model" else "user"
                messages.append({"role": role, "content": body})

        # Native Ollama schema enforcement is more reliable than asking the
        # small local model to output JSON with prompt instructions alone.
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_ctx": 4096},
        }
        url = self.base_url.rstrip("/") + "/api/chat"
        async with httpx.AsyncClient(timeout=240.0) as client:
            for attempt in range(2):
                response = await client.post(url, json=payload)
                response.raise_for_status()
                content = response.json()["message"]["content"]
                try:
                    parsed = self.schema_type.model_validate_json(content)
                    text = parsed.model_dump_json()
                    logger.info("Validated %s response with local Ollama", self.schema_type.__name__)
                    yield LlmResponse(content=types.Content(
                        role="model", parts=[types.Part(text=text)]
                    ))
                    return
                except ValueError:
                    logger.warning("Malformed %s response, retrying", self.schema_type.__name__)
                    if attempt == 1:
                        raise
                    payload["messages"].append({
                        "role": "user",
                        "content": "Your previous response did not match the JSON schema. Return valid schema JSON only.",
                    })
