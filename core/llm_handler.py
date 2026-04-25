"""
LLM handler for optional Jarvis-style responses.
"""

import re
import time
from typing import Any, Dict, List, Optional

import requests

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("[LLMHandler] llama-cpp-python not installed. GGUF models unavailable.")


class LLMHandler:
    """Handles optional LLM-backed scene descriptions and answers."""

    def __init__(self, llm_config: dict):
        self.config = llm_config
        self.provider = llm_config.get("provider", "ollama")
        self.api_key = llm_config.get("api_key")
        self.api_url = llm_config.get("api_url", "http://localhost:11434")
        self.model = llm_config.get("model", "smollm2")
        self.model_path = llm_config.get("model_path", "")
        self.temperature = llm_config.get("temperature", 0.2)
        self.max_tokens = llm_config.get("max_tokens", 80)
        self.timeout = llm_config.get("timeout", 15)
        self._gguf_model = None
        self.response_cache = {}
        self.cache_ttl = 5.0

        print(f"[LLMHandler] Initialized with model: {self.provider} / {self.model_path or self.model}")

    def check_connection(self) -> bool:
        if self.provider == "gguf":
            return LLAMA_CPP_AVAILABLE and bool(self.model_path)
        if self.provider == "gemini":
            return self.api_key is not None

        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"[LLMHandler] Ollama returned status {response.status_code}")
                return False

            model_names = [m["name"] for m in response.json().get("models", [])]
            if self.model in model_names:
                print(f"[LLMHandler] Connected to Ollama, model '{self.model}' available")
                return True

            print(f"[LLMHandler] Model '{self.model}' not found. Available: {model_names}")
            return False
        except requests.exceptions.RequestException as exc:
            print(f"[LLMHandler] Cannot connect to Ollama: {exc}")
            return False

    def _generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        if self.provider == "gguf":
            return self._generate_gguf(prompt, system_prompt)
        if self.provider == "gemini":
            return self._generate_gemini(prompt, system_prompt)
        return self._generate_ollama(prompt, system_prompt)

    def _sanitize_response(self, text: Optional[str]) -> Optional[str]:
        """Strip echoed prompt metadata before speaking the result."""
        if not text:
            return None

        cleaned_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith(("DETECTED OBJECTS:", "HISTORY:", "USER:", "NAVSENSE:", "ASSISTANT:")):
                continue
            if line.lower().startswith(("user:", "assistant:")):
                continue
            cleaned_lines.append(line)

        cleaned = " ".join(cleaned_lines).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            return None

        meta_tokens = ["detected objects", "history:", "navsense:", "<|im_start|>", "<|im_end|>"]
        if any(token in cleaned.lower() for token in meta_tokens):
            return None

        return cleaned[:220].strip()

    def _generate_gguf(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        if not LLAMA_CPP_AVAILABLE:
            print("[LLMHandler] llama-cpp-python not available")
            return None

        try:
            if self._gguf_model is None:
                import os

                path = self.model_path
                if not os.path.isabs(path):
                    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)

                print(f"[LLMHandler] Loading GGUF model from {path}...")
                self._gguf_model = Llama(
                    model_path=path,
                    n_ctx=512,
                    n_threads=4,
                    verbose=False,
                )
                print("[LLMHandler] GGUF model loaded")

            sys_prompt = system_prompt or (
                "You are JARVIS, a navigation AI. Give concise responses under 15 words."
            )
            chat_prompt = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            start = time.time()
            output = self._gguf_model(
                chat_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )
            text = self._sanitize_response(output["choices"][0]["text"].strip())
            print(f"[LLMHandler] GGUF replied in {time.time() - start:.1f}s")
            return text
        except Exception as exc:
            print(f"[LLMHandler] GGUF error: {exc}")
            return None

    def _generate_gemini(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        if not self.api_key:
            print("[LLMHandler] Gemini API key missing")
            return None

        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent?key={self.api_key}"
            )
            full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            payload = {
                "contents": [{"parts": [{"text": full_text}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                },
            }
            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code != 200:
                print(f"[LLMHandler] Gemini API error: {response.status_code} - {response.text}")
                return None

            parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return self._sanitize_response(parts[0].get("text", "").strip() if parts else None)
        except Exception as exc:
            print(f"[LLMHandler] Gemini exception: {exc}")
            return None

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            cache_key = full_prompt[:100]
            cached = self.response_cache.get(cache_key)
            if cached and (time.time() - cached[1] < self.cache_ttl):
                return cached[0]

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            response = requests.post(f"{self.api_url}/api/generate", json=payload, timeout=self.timeout)
            if response.status_code != 200:
                return None

            generated_text = self._sanitize_response(response.json().get("response", "").strip())
            self.response_cache[cache_key] = (generated_text, time.time())
            return generated_text
        except Exception as exc:
            print(f"[LLMHandler] Ollama error: {exc}")
            return None

    def generate_scene_description(self, detections: List[Dict[str, Any]]) -> str:
        if not detections:
            return "I don't see any objects in the current view."

        object_summary = []
        for det in detections:
            obj_class = det.get("class_name", det.get("class", "object"))
            distance = det.get("distance", 0)
            zone = det.get("zone", "center")
            angle = det.get("angle_degrees", 0)
            object_summary.append(f"{obj_class} at {distance:.1f}m, {zone} ({angle:.0f} degrees)")

        prompt = "Target Environment Scan:\n" + "\n".join(f"- {obj}" for obj in object_summary)
        system_prompt = (
            "You are JARVIS. The user is blind. Describe the scene in one concise sentence."
        )
        response = self._generate(prompt, system_prompt)
        if response:
            return response

        closest = min(detections, key=lambda x: x.get("distance", 999))
        return (
            f"I see {len(detections)} objects. The closest is a "
            f"{closest.get('class_name', 'object')} at {closest.get('distance', 0):.1f} meters."
        )

    def answer_query(
        self,
        question: str,
        detections: List[Dict[str, Any]],
        conversation_history: Optional[List[str]] = None,
    ) -> str:
        if detections:
            objects_text = ", ".join(
                f"{d.get('class_name', 'object')} ({d.get('zone', 'nearby')})" for d in detections
            )
        else:
            objects_text = "nothing detected"

        context = f"DETECTED OBJECTS: {objects_text}\n"
        if conversation_history:
            context += "HISTORY:\n" + "\n".join(conversation_history[-1:]) + "\n"

        prompt = f"{context}USER: {question}\nNAVSENSE:"
        system_prompt = (
            "You are JARVIS. Answer concisely. Use detected objects for surroundings questions, "
            "and answer directly for general knowledge questions."
        )
        response = self._generate(prompt, system_prompt)
        return response or "I am having trouble processing that request right now."

    def process_location_query(self, object_name: str, detections: List[Dict[str, Any]]) -> str:
        matches = [
            d
            for d in detections
            if object_name.lower() in d.get("class_name", d.get("class", "")).lower()
        ]
        if not matches:
            visible = ", ".join(d.get("class_name", "unknown") for d in detections[:5])
            if visible:
                return f"I do not see {object_name}. I currently see {visible}."
            return f"I do not see {object_name} in your current view."

        match = min(matches, key=lambda item: item.get("distance", 999))
        return (
            f"The {match.get('class_name', object_name)} is {match.get('distance', 0):.1f} meters "
            f"{match.get('direction_text', 'nearby')}."
        )
