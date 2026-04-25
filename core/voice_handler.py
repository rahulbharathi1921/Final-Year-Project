"""
Voice Handler Module
Manages speech recognition (Whisper) and text-to-speech (gTTS/pyttsx3).
"""
import re
import speech_recognition as sr
import threading
import queue
import time
import os
import tempfile
import pygame
from gtts import gTTS
import pyttsx3
import audioop
from typing import Optional, Callable, List
from faster_whisper import WhisperModel
class VoiceHandler:
    """Handles high-accuracy offline voice recognition and robust TTS."""
    def __init__(self, voice_config: dict, tts_config: dict):
        self.voice_config = voice_config
        self.tts_config = tts_config
        # Speech recognition (Whisper)
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.listen_thread = None
        self.whisper_model = None
        self.whisper_model_size = voice_config.get('model_size', 'base.en')
        # TTS
        self.tts_engine_type = tts_config.get('engine', 'gtts').lower()
        self.tts_engine = None # pyttsx3 engine
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.is_speaking = False
        self._tts_lock = threading.Lock()
        # Callbacks & History
        self.command_callback = None
        self.history = []
        # Settings
        self.language = voice_config.get('language', 'en-US')
        self.phrase_limit = voice_config.get('phrase_time_limit', 5)
        self.tts_rate = tts_config.get('rate', 150)
        print(f"[VoiceHandler] Whisper model selected: {self.whisper_model_size}")
    def initialize(self) -> bool:
        """Initialize all voice components."""
        try:
            print("[VoiceHandler] Initializing components...")
            # 1. Microphone
            self.microphone = sr.Microphone()
            # 2. Whisper (Offline STT) - Optimized for Sub-Second Speeds
            print(f"[VoiceHandler] Loading Whisper '{self.whisper_model_size}' (Offline)...")
            self.whisper_model = WhisperModel(self.whisper_model_size, device="cpu", compute_type="int8")
            # 3. pyttsx3 (Offline TTS) - Will be initialized in its own thread to avoid hangs
            pass
            # Adjust for ambient noise and sensitivity
            if self.microphone:
                print("[VoiceHandler] Calibrating for ambient noise... (2.0 seconds)")
                with self.microphone as source:
                    # Accuracy Settings: Reverted to 1.2s for stable speech capture
                    self.recognizer.pause_threshold = 1.2 
                    self.recognizer.non_speaking_duration = 0.5
                    self.recognizer.energy_threshold = 800 # Stable threshold for most mics
                    self.recognizer.dynamic_energy_threshold = False
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                print(f"[VoiceHandler] Initial energy threshold set to: {self.recognizer.energy_threshold}")
            # 4. Pygame for gTTS playback
            if self.tts_engine_type == 'gtts':
                pygame.mixer.init()
            # 5. Start TTS loop
            self.tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
            self.tts_thread.start()
            print("[VoiceHandler] System initialized successfully!")
            return True
        except Exception as e:
            print(f"[VoiceHandler] Initialization failed: {e}")
            return False
    def speak(self, text: str, priority: bool = False):
        """Queue text for speaking."""
        if not text: return
        if priority:
            while not self.tts_queue.empty():
                try: self.tts_queue.get_nowait()
                except queue.Empty: break
        self.tts_queue.put(text)
        print(f"[TTS] Ready: '{text[:50]}...'")
    def _tts_loop(self):
        """Continuous TTS consumer."""
        # 🧵 ON WINDOWS: SAPI5 Must be initialized in the worker thread!
        try:
            if self.tts_engine_type == 'pyttsx3' or not self.tts_engine:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.tts_rate)
                voices = self.tts_engine.getProperty('voices')
                v_idx = self.tts_config.get('voice_index', 1) # Default to Zira (Female)
                if v_idx < len(voices):
                    self.tts_engine.setProperty('voice', voices[v_idx].id)
                print(f"[VoiceHandler] TTS Thread active with engine: {self.tts_engine_type}")
        except Exception as e:
            print(f"[TTS] Thread Init error: {e}")
        while True:
            try:
                text = self.tts_queue.get(timeout=1)
                self.is_speaking = True
                # Accuracy Revert: Use the engine specified in settings.yaml (gTTS for quality)
                clean_text = text.replace("%", " percent ").replace("#", " number ").strip()
                success = False
                if self.tts_engine_type == 'gtts':
                    success = self._speak_gtts(clean_text)
                if not success: # Fallback to offline
                    self._speak_pyttsx3(clean_text)
                self.is_speaking = False
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS] Loop error: {e}")
                self.is_speaking = False
    def _speak_gtts(self, text: str) -> bool:
        try:
            tts = gTTS(text=text, lang=self.language.split('-')[0])
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
                tmp = tf.name
                tts.save(tmp)
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            os.remove(tmp)
            return True
        except Exception as e:
            print(f"[TTS] gTTS failed: {e}")
            return False
    def _speak_pyttsx3(self, text: str):
        try:
            if not self.tts_engine:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.tts_rate)
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS] pyttsx3 failed: {e}")
    def start_listening(self, callback: Callable[[str], None]):
        self.command_callback = callback
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
    def _listen_loop(self):
        print("[VoiceHandler] Listening thread active.")
        
        # Whitelist of recognized command keywords (permissive - only filter obvious garbage)
        COMMAND_KEYWORDS = [
            "indoor", "outdoor", "jarvis", "inside", "outside", "switch", "change",
            "mode", "activate", "enter", "where", "what", "who", "how", "find",
            "help", "hello", "hi", "hey", "thank", "thanks", "safe", "clear",
            "walk", "repeat", "again", "scan", "look", "see", "person", "chair",
            "door", "table", "car", "stop", "go", "exit", "quit", "bye"
        ]
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    if self.is_speaking:
                        time.sleep(0.2)
                        continue
                    
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=self.phrase_limit)
                    wav_data = audio.get_wav_data()
                    
                    # Lower threshold to capture softer speech
                    rms = audioop.rms(wav_data, 2)
                    if rms < 500:
                        print(f"[VoiceHandler] Audio too quiet (rms={rms}), listening again...")
                        continue
                    
                    print(f"[VoiceHandler] Transcribing... (Audio Strength: {rms})")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
                        tmp_wav = tf.name
                        tf.write(wav_data)
                    
                    try:
                        segments, _ = self.whisper_model.transcribe(tmp_wav, beam_size=1, language="en")
                        text = " ".join([seg.text for seg in segments]).strip()
                        
                        # Clean text and check for keywords
                        text_clean = text.replace(".", "").replace(",", "").replace("?", "").strip().lower()
                        
                        if text_clean and len(text_clean) > 1:
                            has_keyword = any(kw in text_clean for kw in COMMAND_KEYWORDS)
                            if has_keyword or len(text_clean) <= 15 or 3 <= len(text_clean) <= 30:
                                print(f"[VoiceHandler] Heard: '{text}' -> '{text_clean}'")
                                if self.command_callback:
                                    self.command_callback(text_clean)
                            else:
                                print(f"[VoiceHandler] Filtered (no keyword): '{text_clean}'")
                        else:
                            print(f"[VoiceHandler] Empty transcription")
                    finally:
                        if os.path.exists(tmp_wav):
                            os.remove(tmp_wav)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[VoiceHandler] Listen error: {e}")
                time.sleep(0.5)

    def is_currently_speaking(self): return self.is_speaking
    def wait_until_done_speaking(self, timeout=10):
        start = time.time()
        while self.is_speaking and (time.time() - start) < timeout:
            time.sleep(0.1)
    def shutdown(self):
        self.is_listening = False
        print("[VoiceHandler] Cleaned up.")
    def add_history(self, text: str):
        self.history.append(text)
        if len(self.history) > 10: self.history.pop(0)
    def get_history(self, limit=5): return self.history[-limit:]