"""
Gemini LLM integration - free alternative to Anthropic.

Use as fallback when Anthropic credits are exhausted or for cost optimization.
"""

import structlog
import json
from typing import List, Dict, Optional, Any

log = structlog.get_logger()


class GeminiClient:
    """Wrapper around Gemini API for text generation."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini client.

        Args:
            api_key: Google Generative AI API key
            model: Model name (default: gemini-1.5-flash for free tier)
        """
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
                log.info("gemini_client_initialized", model=self.model)
            except ImportError:
                log.error("gemini_import_failed", detail="Install google-generativeai")
                return None
            except Exception as e:
                log.error("gemini_init_failed", error=str(e))
                return None
        return self._client

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate text using Gemini API.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            max_tokens: Max tokens in response

        Returns:
            Generated text
        """
        genai = self._get_client()
        if genai is None:
            log.error("gemini_unavailable")
            return ""

        try:
            model = genai.GenerativeModel(self.model)
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                ),
            )

            result = response.text.strip()
            log.info("gemini_generation_success", tokens=len(result.split()))
            return result
        except Exception as e:
            log.error("gemini_generation_failed", error=str(e))
            return ""

    async def generate_json_response(
        self,
        prompt: str,
        max_tokens: int = 1500,
    ) -> List[Dict[str, Any]]:
        """
        Generate JSON response using Gemini (for Task B CoT reranking).

        Args:
            prompt: Prompt requesting JSON array response
            max_tokens: Max tokens

        Returns:
            Parsed JSON list or empty list on error
        """
        genai = self._get_client()
        if genai is None:
            log.error("gemini_unavailable")
            return []

        try:
            model = genai.GenerativeModel(self.model)
            
            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nRespond with ONLY valid JSON array, no markdown fences."

            response = model.generate_content(
                json_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,  # Lower temp for structured output
                ),
            )

            raw = response.text.strip()

            # Strip markdown if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw.strip())
            log.info("gemini_json_generation_success", items=len(result))
            return result
        except json.JSONDecodeError as e:
            log.error("gemini_json_decode_failed", error=str(e))
            return []
        except Exception as e:
            log.error("gemini_json_generation_failed", error=str(e))
            return []


async def get_gemini_client(api_key: str) -> Optional[GeminiClient]:
    """
    Factory function to create and validate Gemini client.

    Args:
        api_key: Google Generative AI API key

    Returns:
        GeminiClient or None if initialization failed
    """
    if not api_key:
        log.warning("gemini_api_key_missing")
        return None

    try:
        client = GeminiClient(api_key=api_key)
        # Test connection
        test = await client.generate_text("Test")
        if test:
            log.info("gemini_client_validated")
            return client
        else:
            log.warning("gemini_client_test_failed")
            return None
    except Exception as e:
        log.error("gemini_client_creation_failed", error=str(e))
        return None
