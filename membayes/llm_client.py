from __future__ import annotations

"""
LLM Client for DeepInfra (Chat) + OpenAI (Embeddings)
=======================================================

Chat (DeepInfra / Qwen3-14B):
    POST https://api.deepinfra.com/v1/openai/chat/completions

Embeddings (OpenAI text-embedding-3-large):
    POST https://api.openai.com/v1/embeddings

Required env vars:
    DEEPINFRA_API_KEY
    OPENAI_API_KEY
"""

import os
import re
import json
import time
import logging
from typing import Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# Endpoints
DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

# Default models
DEFAULT_MODEL = "Qwen/Qwen3-14B"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


class LLMClient:
    """Client for DeepInfra chat (Qwen) and OpenAI embeddings."""

    def __init__(self, api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 model: str = DEFAULT_MODEL,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 max_retries: int = 3,
                 timeout: int = 60):
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self.timeout = timeout
        self.total_calls = 0
        self.total_tokens = 0
        self.total_embedding_calls = 0

        if not self.api_key:
            raise ValueError(
                "No DeepInfra API key found. Set DEEPINFRA_API_KEY environment variable "
                "or pass api_key to LLMClient()."
            )
        if not self.openai_api_key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key to LLMClient()."
            )
        if not HAS_REQUESTS:
            raise ImportError(
                "The 'requests' library is required. Install with: pip install requests"
            )

        logger.info("LLMClient initialized: chat=%s, embeddings=%s", model, embedding_model)

    def chat(self, user_message: str, system_message: str = "",
             temperature: float = 0.1, max_tokens: int = 512) -> str:
        """Send a chat completion request to DeepInfra."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        logger.debug("Chat request: model=%s, msg_len=%d, temp=%.1f",
                      self.model, len(user_message), temperature)

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    DEEPINFRA_CHAT_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code != 200:
                    body = resp.text[:500]
                    logger.error("Chat HTTP %d: %s | URL: %s | Model: %s",
                                 resp.status_code, body, DEEPINFRA_CHAT_URL, self.model)
                resp.raise_for_status()
                data = resp.json()

                self.total_calls += 1
                usage = data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)

                content = data["choices"][0]["message"]["content"].strip()
                logger.debug("Chat response: %d tokens, %d chars",
                             usage.get("total_tokens", 0), len(content))
                return content

            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited, waiting %ds (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                raise
            except requests.exceptions.Timeout:
                logger.warning("Chat timeout (attempt %d/%d)", attempt + 1, self.max_retries)
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
            except Exception as e:
                logger.error("Chat call failed: %s", e)
                raise

        return ""

    def chat_json(self, user_message: str, system_message: str = "",
                  temperature: float = 0.0, max_tokens: int = 512) -> dict:
        """Send a chat request expecting JSON output. Returns parsed dict."""
        json_instruction = (
            "You must respond with valid JSON only. No markdown, no explanation, "
            "no code fences, no thinking. Just the JSON object."
        )
        full_system = f"{system_message}\n\n{json_instruction}" if system_message else json_instruction

        raw = self.chat(user_message, system_message=full_system,
                        temperature=temperature, max_tokens=max_tokens)

        # Strip <think>...</think> blocks (Qwen3 reasoning mode fallback)
        cleaned = raw.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            result = json.loads(cleaned)
            logger.debug("Parsed JSON: %s", list(result.keys()) if isinstance(result, dict) else type(result))
            return result
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    result = json.loads(cleaned[start:end])
                    logger.debug("Parsed JSON (extracted): %s", list(result.keys()))
                    return result
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse JSON from LLM response: %s", raw[:300])
            return {}

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Compute embeddings via OpenAI text-embedding-3-large."""
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_model,
            "input": texts,
            "encoding_format": "float",
        }

        logger.debug("Embedding request: %d texts, model=%s", len(texts), self.embedding_model)

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    OPENAI_EMBEDDINGS_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code != 200:
                    body = resp.text[:500]
                    logger.error("Embed HTTP %d: %s", resp.status_code, body)
                resp.raise_for_status()
                data = resp.json()

                self.total_embedding_calls += 1

                # Sort by index to preserve input order
                items = sorted(data["data"], key=lambda x: x["index"])
                dim = len(items[0]["embedding"]) if items else 0
                logger.debug("Embedding response: %d vectors, dim=%d", len(items), dim)
                return [item["embedding"] for item in items]

            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited (embed), waiting %ds", wait)
                    time.sleep(wait)
                    continue
                raise
            except requests.exceptions.Timeout:
                logger.warning("Embed timeout (attempt %d/%d)", attempt + 1, self.max_retries)
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
            except Exception as e:
                logger.error("Embed call failed: %s", e)
                raise

        return [[] for _ in texts]

    def embed_single(self, text: str) -> list[float]:
        """Compute embedding for a single text string."""
        return self.embed(text)[0]

    def get_usage(self) -> dict:
        """Return usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_embedding_calls": self.total_embedding_calls,
        }
