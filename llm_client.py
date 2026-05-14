"""
LLM client abstraction supporting multiple providers (Vertex AI, OpenAI).

This module provides a unified interface for calling LLMs:
- Vertex AI (Google Cloud): Using GenerativeModel with Application Default Credentials
- OpenAI: Using API keys

The client auto-detects the Google Cloud project from environment variables or
gcloud configuration, simplifying credential management.
"""

import os
from typing import Dict, List

try:
    import google.auth
except Exception:
    google_auth = None
else:
    google_auth = google.auth

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
except Exception:
    vertexai = None
    GenerativeModel = None
    GenerationConfig = None


class LLMClient:
    """Unified LLM client for Vertex AI and OpenAI providers.
    
    Attributes:
        provider: "vertex_ai" or "openai"
        model_name: Model identifier (e.g., "gemini-2.5-flash")
        project: GCP project ID (Vertex AI only)
        location: GCP region (Vertex AI only)
    """

    def __init__(
        self,
        provider: str = "vertex_ai",
        model: str | None = None,
        project: str | None = None,
        location: str | None = None,
    ):
        """Initialize LLM client with provider and credentials.
        
        For Vertex AI:
        - Reads GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables
        - Falls back to gcloud default project if env vars not set
        - Supports Application Default Credentials (gcloud auth application-default login)
        
        Args:
            provider: "vertex_ai" (default) or "openai"
            model: Model ID (defaults to env VERTEX_MODEL or "gemini-2.5-flash")
            project: GCP project ID (auto-detected if not provided)
            location: GCP region (defaults to "us-central1")
            
        Raises:
            EnvironmentError: If required packages/credentials are missing
            ValueError: If provider is unknown
        """
        self.provider = provider
        self.model_name = model or os.getenv("VERTEX_MODEL", "gemini-2.5-flash")
        
        if provider == "openai":
            # OpenAI provider setup
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise EnvironmentError("OPENAI_API_KEY not set")
            if OpenAI is None:
                raise EnvironmentError("openai package not installed")
            self.openai_client = OpenAI(api_key=key)
            
        elif provider == "vertex_ai":
            # Vertex AI provider setup with automatic project detection
            if vertexai is None or GenerativeModel is None:
                raise EnvironmentError("vertexai package not installed. Install google-cloud-aiplatform.")
            
            # Detect GCP project from environment or gcloud config
            self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
            if not self.project and google_auth is not None:
                _, detected_project = google_auth.default()
                self.project = detected_project
            if not self.project:
                raise EnvironmentError(
                    "Google Cloud project not found. Set GOOGLE_CLOUD_PROJECT or run "
                    "`gcloud config set project YOUR_PROJECT_ID`."
                )
            
            self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            vertexai.init(project=self.project, location=self.location)
            self.model = GenerativeModel(self.model_name)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def chat(self, messages: List[Dict[str, str]], temperature=0.0, max_tokens=512) -> str:
        """Send messages to LLM and return response.
        
        Args:
            messages: List of message dicts with "role" ("system"/"user"/"assistant") and "content"
            temperature: Model temperature (0.0 = deterministic, higher = more creative)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response text (stripped of leading/trailing whitespace)
        """
        if self.provider == "openai":
            # OpenAI API call
            resp = self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()

        # Vertex AI API call
        prompt = self._messages_to_vertex_prompt(messages)
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()

    @staticmethod
    def _messages_to_vertex_prompt(messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Vertex AI prompt format.
        
        Vertex AI's GenerativeModel doesn't support message history directly,
        so we concatenate system + user messages into a single prompt.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Combined prompt string
        """
        blocks = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            blocks.append(f"{role}:\n{content}")
        return "\n\n".join(blocks)
