"""
LLM handler for optional Jarvis-style responses.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import requests


class LLMHandler:
    """Handles optional LLM-backed scene descriptions and answers."""

    def __init__(self, llm_config: dict):
        self.config = llm_config
        self.provider = llm_config.get("provider", "gemini")
        self.api_key = llm_config.get("api_key") or os.getenv(llm_config.get("api_key_env", "GEMINI_API_KEY"))
        self.api_url = llm_config.get("api_url", "https://generativelanguage.googleapis.com/v1beta")
        self.model = llm_config.get("model", "gemini-2.5-flash")
        self.temperature = llm_config.get("temperature", 0.2)
        self.max_tokens = llm_config.get("max_tokens", 80)
        self.timeout = llm_config.get("timeout", 15)
        self.response_cache = {}
        self.cache_ttl = 5.0

        print(f"[LLMHandler] Initialized with model: {self.provider} / {self.model}")

    def check_connection(self) -> bool:
        if self.provider != "gemini":
            print(f"[LLMHandler] Unsupported provider configured: {self.provider}")
            return False
        if not self.api_key:
            print("[LLMHandler] Gemini API key missing. Set GEMINI_API_KEY.")
            return False
        return True

    def _generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        if self.provider != "gemini":
            return None
        return self._generate_gemini(prompt, system_prompt)

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

        words = cleaned.lower().split()
        if len(words) >= 8:
            for size in (3, 4, 5):
                chunks = [" ".join(words[i:i + size]) for i in range(0, len(words) - size + 1, size)]
                if len(chunks) >= 2 and len(set(chunks)) == 1:
                    return None

        if cleaned.count(".") >= 2:
            sentences = [part.strip().lower() for part in cleaned.split(".") if part.strip()]
            if len(sentences) >= 2 and len(set(sentences)) == 1:
                return None

        return cleaned[:220].strip()

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
