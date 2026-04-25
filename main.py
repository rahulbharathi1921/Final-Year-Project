"""
NavSense - Main Application (Phase 1)
Integrated system for object detection, voice interaction, and danger alerts.
"""

import sys
import os

# Fix for OMP: Error #15 (Multiple OpenMP runtimes)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import yaml
import json
import time
import threading
from pathlib import Path
from collections import deque
import difflib

# Import core modules
from core.camera_handler import CameraHandler
from core.detector import ObjectDetector
from core.distance_direction import DistanceDirectionCalculator
from core.voice_handler import VoiceHandler
from core.alert_system import AlertSystem

# Phase 2: LLM and tracking
from core.llm_handler import LLMHandler
from core.object_tracker import SimpleTracker
from core.session_logger import SessionLogger


class NavSense:
    """Main NavSense application."""
    
    def __init__(self):
        """Initialize NavSense application."""
        self.config = None
        self.object_data = None
        
        # Core components
        self.camera = None
        self.detector = None
        self.calculator = None
        self.voice = None
        self.alert_system = None
        
        # Phase 2 components
        self.llm = None
        self.tracker = None
        self.session_logger = None
        self.last_detections = []
        self._detections_lock = threading.Lock()
        self._empty_frame_count = 0
        self._max_empty_frames_before_clear = 15
        self._llm_busy = False
        self.speech_priority = False
        self.frame_index = 0
        # State
        self.current_mode = "indoor"
        self.is_running = False
        self.is_initialized = False
        
        # Assistant Memory & Context
        self.last_voice_report = ""
        self.assistant_history = []
        
        # Load Voice Vocabulary (Massive Dictionary)
        self.voice_vocab = {}
        try:
            vocab_path = Path("config/voice_vocab.json")
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    self.voice_vocab = json.load(f)
                print(f"[NavSense] Loaded {sum(len(v) for v in self.voice_vocab.values())} keywords.")
        except Exception as e:
            print(f"[NavSense] Error loading voice_vocab.json: {e}")
        
        # Personalized User Identity (Default)
        self.user_name = "Joe"
        self.object_aliases = {}
        
        # Display
        self.display_thread = None
        
    def load_config(self) -> bool:
        """Load configuration files."""
        try:
            print("[NavSense] Loading configuration...")
            
            with open('config/settings.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
            
            with open('config/object_data.json', 'r') as f:
                self.object_data = json.load(f)
            
            self.speech_priority = self.config.get('tts', {}).get('speech_priority', False)
            
            self.user_name = self.config.get('tts', {}).get('user_name', 'Joe')
            self.session_logger = SessionLogger(self.config.get('logging', {}))
            self.object_aliases = self._build_object_aliases()
            
            print("[NavSense] Configuration loaded successfully")
            if self.session_logger:
                self.session_logger.log_event("system", message="Configuration loaded")
            return True
            
        except Exception as e:
            print(f"[NavSense] ERROR: Failed to load config: {e}")
            return False
    
    def initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            print("\n" + "="*60)
            print("NavSense - Navigation Assistant for Visually Impaired")
            print("="*60 + "\n")
            
            # [1/5] Initializing camera...
            print("\n[1/5] Initializing camera...")
            self.camera = CameraHandler(self.config['camera'])
            if not self.camera.initialize():
                print("[NavSense] ❌ CRITICAL ERROR: Camera initialization failed.")
                return False
            # Reduce OpenCV thread contention for smoother audio
            cv2.setNumThreads(1)
            
            # Initialize detector
            print("\n[2/5] Loading object detector...")
            self.detector = ObjectDetector(self.config['detection'], self.object_data)
            if not self.detector.initialize():
                return False
            
            # Initialize calculator
            print("\n[3/5] Initializing distance & direction calculator...")
            self.calculator = DistanceDirectionCalculator(
                self.config['camera'],
                self.config['distance'],
                self.object_data
            )
            print("[3/5] Calculator initialized")
            
            # Initialize alert system
            print("\n[4/5] Initializing alert system...")
            self.alert_system = AlertSystem(self.config['alerts'], self.object_data)
            print("[4/5] Alert system initialized")
            
            # Initialize voice handler
            print("\n[5/7] Initializing voice interface...")
            self.voice = VoiceHandler(self.config['voice'], self.config['tts'])
            if not self.voice.initialize():
                return False
            
            # Initialize LLM handler (Phase 2)
            print("\n[6/7] Initializing LLM (Ollama phi3)...")
            self.llm = LLMHandler(self.config['llm'])
            if self.llm.check_connection():
                print("[6/7] LLM initialized successfully")
            else:
                print("[6/7] ⚠️ LLM not available (continuing without)")
            
            # Initialize tracker (Phase 2)
            print("\n[7/7] Initializing object tracker...")
            self.tracker = SimpleTracker(max_disappeared=5, max_distance=120.0)
            print("[7/7] Tracker initialized")
            
            print("\n" + "="*60)
            print("All components initialized successfully!")
            print("="*60 + "\n")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"[NavSense] ERROR: Initialization failed: {e}")
            return False
    
    def start(self):
        """Start NavSense application."""
        if not self.is_initialized:
            print("[NavSense] ERROR: Components not initialized!")
            return
        
        print("[NavSense] Starting NavSense...")
        self.is_running = True
        
        # Priority startup greeting
        self._speak_and_remember(f"Hello {self.user_name}! Navsense is active and 100 percent offline.", priority=True)
        self._speak_and_remember(f"Hello {self.user_name}! NavSense is active. Which mode do you want to activate now?: indoor, outdoor, or jarvis?", priority=True)
        
        # Optional: Short wait for the very first greeting, but much shorter than 10s
        self.voice.wait_until_done_speaking(timeout=2.0)
        time.sleep(0.1)
        
        # Start voice listening
        print("[NavSense] Starting voice recognition...")
        self.voice.start_listening(self._on_voice_command)
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        # Main detection loop
        self._main_loop()
    
    def _get_last_detections_snapshot(self):
        """Thread-safe copy of current detections."""
        with self._detections_lock:
            return list(self.last_detections)
    
    def _speak_and_remember(self, text: str, priority: bool = False):
        """Speak text and store it in memory for the 'Repeat' command."""
        if not text:
            return
        self.last_voice_report = text
        if self.session_logger:
            self.session_logger.log_response(text, mode=self.current_mode, source="navsense", priority=priority)
        self.voice.speak(text, priority=priority)

    def _fuzzy_match(self, text: str, keywords: list) -> bool:
        """
        Check if text fuzzy matches any keywords.
        Handles speech recognition variations and context.
        """
        text_lower = text.lower()
        
        # Common mishearings / variations
        global_variations = {
            'indoor': ['indore', 'in door', 'inside', 'internal', 'in-door', 'indoors'],
            'outdoor': ['out door', 'outside', 'external', 'out-door', 'outdoors'],
            'jarvis': ['jarvis', 'jarvish', 'jarvas', 'assistant', 'ai mode'],
            'where': ['where', 'find', 'locate', 'location', 'what is', 'whers', 'wheres'],
            'help': ['help', 'manual', 'what can i say', 'commands', 'how to use'],
            'safety': ['safe', 'safety', 'can i walk', 'is it clear', 'path clear'],
            'repeat': ['repeat', 'pardon', 'again', 'what was that', 'one more time']
        }
        
        for keyword in keywords:
            # 1. Check global variations first
            if keyword in global_variations:
                if any(var in text_lower for var in global_variations[keyword]):
                    return True
            
            # 2. Direct substring match
            if keyword in text_lower:
                return True
            
            # 3. Leading character match (robustness for cut-off audio)
            if len(keyword) >= 4 and len(text_lower) >= 4:
                if text_lower[:4] == keyword[:4]:
                    return True
                    
        return False

    def _build_object_aliases(self):
        """Build a normalized alias map for location and Jarvis queries."""
        aliases = {
            'person': ['person', 'people', 'human', 'man', 'woman', 'child', 'anyone', 'someone'],
            'cell phone': ['cell phone', 'phone', 'mobile', 'smartphone', 'iphone'],
            'laptop': ['laptop', 'computer', 'pc', 'macbook'],
            'tv': ['tv', 'television', 'screen', 'monitor', 'display'],
            'chair': ['chair', 'seat', 'stool'],
            'couch': ['couch', 'sofa'],
            'dining table': ['table', 'desk', 'counter'],
            'door': ['door', 'doorway', 'entrance', 'exit', 'gate'],
            'car': ['car', 'vehicle', 'automobile'],
            'bicycle': ['bicycle', 'bike', 'cycle'],
            'motorcycle': ['motorcycle', 'motorbike'],
            'bottle': ['bottle', 'water bottle'],
            'cup': ['cup', 'mug', 'glass'],
            'book': ['book', 'notebook'],
            'backpack': ['backpack', 'bag'],
            'refrigerator': ['refrigerator', 'fridge'],
            'traffic light': ['traffic light', 'signal'],
            'stop sign': ['stop sign'],
        }

        for key, values in self.voice_vocab.items():
            if key in {'indoor', 'outdoor', 'jarvis'}:
                continue
            if isinstance(values, list) and values:
                aliases.setdefault(key, [])
                aliases[key].extend(str(value) for value in values if value)

        for class_name in self.object_data.get('coco_classes', []):
            aliases.setdefault(class_name, [class_name])

        return {key: sorted(set(values), key=len, reverse=True) for key, values in aliases.items()}

    def _extract_target_object(self, text: str):
        """Resolve the user's requested object to a class name."""
        normalized = f" {text.lower().strip()} "
        for class_name, aliases in self.object_aliases.items():
            for alias in aliases:
                if f" {alias} " in normalized:
                    return class_name

        stopwords = {
            'where', 'is', 'the', 'a', 'an', 'my', 'find', 'me', 'show', 'locate',
            'please', 'tell', 'what', 'about', 'nearest'
        }
        words = [word for word in normalized.split() if word not in stopwords]
        if not words:
            return None
        candidate = " ".join(words)
        return candidate.strip()

    def _find_detection_matches(self, target_name: str, detections):
        """Find detections matching a resolved class name or free-text target."""
        target_name = (target_name or '').lower().strip()
        if not target_name:
            return []

        aliases = set(self.object_aliases.get(target_name, [target_name]))
        matches = []
        for det in detections:
            class_name = det.get('class_name', '').lower()
            raw_name = det.get('raw_class_name', class_name).lower()
            if class_name == target_name or raw_name == target_name:
                matches.append(det)
                continue
            if any(alias == class_name or alias == raw_name or alias in class_name for alias in aliases):
                matches.append(det)
        return matches

    def _describe_detection_location(self, detection):
        distance = detection.get('distance', 0.0)
        zone = detection.get('zone', 'center')

        if zone == 'left':
            direction = "on your left side"
        elif zone == 'right':
            direction = "on your right side"
        else:
            direction = "in the center"

        return f"{distance:.1f} meters {direction}"

    def _build_not_found_response(self, target_name: str, detections):
        visible = []
        for det in sorted(detections, key=lambda item: item.get('distance', 999))[:4]:
            visible.append(det.get('class_name', 'object'))

        if visible:
            visible_summary = ", ".join(visible)
            return f"I do not see any {target_name}. I currently see {visible_summary}."
        return f"I do not see any {target_name} right now."

    def _on_voice_command(self, text: str):
        """
        Handle voice commands with keyword + fuzzy matching.
        Mode switching ONLY on short, explicit mode-name phrases.
        """
        text = text.strip().lower()
        if not text or len(text) < 2:
            return

        if self.session_logger:
            self.session_logger.log_voice_command(text, mode=self.current_mode)

        text_words = text.split()
        # Strip punctuation from each word so "anyone?" matches "anyone"
        import string
        text_words_clean = [w.strip(string.punctuation) for w in text_words]
        words_set = set(text_words_clean)


        # ── 1. Mode Switching ─────────────────────────────────────────────────
        # Uses BOTH exact phonetic phrases AND word-level fuzzy matching
        # so Whisper mishearings like "in the mode" still trigger indoor.
        is_short = len(text_words) <= 5

        # Exact phonetic phrases (all known Whisper mishearings)
        MODE_PHRASES = {
            'indoor':  ['indoor', 'in door', 'indore', 'in normal', 'in mode',
                        'in the mode', 'in the model', 'in the remote',
                        'in the middle', 'ender mode', 'enter mode', 'indoor mode'],
            'outdoor': ['outdoor', 'out door', 'outside', 'out mode',
                        'outdoor mode', 'out normal'],
            'jarvis':  ['jarvis', 'javis', 'jarva', 'jarvis mode',
                        'java', 'garvis', 'charvis'],
        }

        if is_short:
            for mode, phrases in MODE_PHRASES.items():
                if any(p in text for p in phrases):
                    self._set_mode(mode)
                    return

        # Word-level fuzzy: any single word that sounds like a mode name
        for word in text_words:
            # Prefix check: 'ind...' words in short phrases → indoor
            if is_short and word.startswith('ind'):
                self._set_mode('indoor')
                return
            if is_short and word.startswith('out') and len(word) >= 5:
                self._set_mode('outdoor')
                return
            for mode in ['indoor', 'outdoor', 'jarvis']:
                score = difflib.SequenceMatcher(None, word, mode).ratio()
                if score >= 0.60 and is_short:
                    self._set_mode(mode)
                    return

        # "switch/change/go to <mode>"
        if any(w in words_set for w in ['switch', 'change', 'go', 'activate', 'enter', 'mode']):
            for mode, phrases in MODE_PHRASES.items():
                if any(p in text for p in phrases) or mode in words_set:
                    self._set_mode(mode)
                    return

        # ── 2. Shutdown ───────────────────────────────────────────────────────
        if any(w in words_set for w in ['shutdown', 'shut', 'exit', 'quit', 'bye']):
            self._speak_and_remember("Shutting down NavSense. Goodbye!", priority=True)
            time.sleep(1.5)
            self.is_running = False
            return

        # ── 3. Social ─────────────────────────────────────────────────────────
        if any(t in text for t in ['hello', 'hi ', 'hey', 'how are you', 'thank you', 'thanks', 'who are you', 'good morning', 'good afternoon', 'good evening']):
            self._handle_social_command(text)
            return

        # ── 4. Help ───────────────────────────────────────────────────────────
        if any(w in words_set for w in ['help', 'commands', 'manual']):
            self._handle_help_command()
            return

        # ── 5. Repeat ─────────────────────────────────────────────────────────
        if any(w in words_set for w in ['repeat', 'again', 'pardon']):
            self._handle_repeat_command()
            return

        # ── 6. Location / Distance / Safety ──────────────────────────────────
        if any(w in words_set for w in ['where', 'find', 'locate', 'location']):
            self._handle_location_query(text)
            return
        if any(w in words_set for w in ['distance', 'far', 'near', 'meters']):
            self._handle_distance_query(text)
            return
        if any(w in words_set for w in ['safe', 'clear', 'obstacle', 'path', 'walk']):
            self._handle_safety_command()
            return

        # ── 7. Scene Scan ─────────────────────────────────────────────────────
        if any(p in text for p in ['what do you see', 'look around', 'describe', 'around me', 'what can you see', 'see anything']) or 'scan' in words_set:
            self._handle_scan_command()
            return


        # ── 8. Object Queries ─────────────────────────────────────────────────
        if any(w in words_set for w in ['person', 'people', 'human', 'man', 'woman', 'child', 'anyone', 'someone', 'crowd']):
            self._handle_person_command()
            return
        if any(w in words_set for w in ['chair', 'seat', 'bench', 'stool', 'sofa', 'couch']):
            self._handle_chair_command()
            return
        if any(w in words_set for w in ['door', 'doorway', 'entrance', 'exit', 'gate']):
            self._handle_door_command()
            return
        if any(w in words_set for w in ['wall', 'barrier', 'obstacle']):
            self._handle_wall_command()
            return
        if any(w in words_set for w in ['laptop', 'tv', 'screen', 'monitor', 'computer', 'television']):
            self._handle_tv_laptop_command()
            return

        # ── 9. General Question / Jarvis mode → LLM ──────────────────────────
        QUESTION_STARTERS = ['what', 'who', 'how', 'tell', 'can you', 'do you', 'is there', 'are there', 'show']
        is_question = any(text.startswith(q) for q in QUESTION_STARTERS)
        if self.current_mode == 'jarvis' or is_question:
            self._handle_jarvis_query(text)
            return

        print(f"[Command] ❔ Unrecognized: '{text}'")
    
    def _handle_social_command(self, text: str):
        """Instant social responses (Enterprise Grade, No LLM delay)."""
        text_lower = text.lower()
        if any(w in text_lower for w in ['hello', 'hi', 'hey', 'greetings']):
            self._speak_and_remember(f"At your service, {self.user_name}. How may I help you?")
        elif 'how are you' in text_lower:
            self._speak_and_remember(f"All systems are nominal, {self.user_name}. Ready for navigation.")
        elif any(w in text_lower for w in ['thank', 'thanks']):
            self._speak_and_remember(f"You are most welcome, Sir.")
        elif 'who are you' in text_lower:
            self._speak_and_remember("I am NavSense, your elite enterprise navigation assistant.")
        else:
            self._handle_jarvis_query(text)

    def _set_mode(self, mode: str):
        """Set operation mode."""
        self.current_mode = mode
        self.alert_system.clear_cooldown()  # Reset cooldowns on mode change
        if self.session_logger:
            self.session_logger.log_event("mode_change", mode=mode, message=f"Mode changed to {mode}")
        
        print(f"\n{'='*60}")
        print(f"  🎯 MODE SET TO: {mode.upper()}")
        print(f"{'='*60}\n")
        
        self._speak_and_remember(f"{mode.capitalize()} mode activated.", priority=True)
        if mode == 'indoor':
            self._speak_and_remember("I will help you navigate indoor spaces. Looking for furniture, doors, and obstacles.")
        elif mode == 'outdoor':
            self._speak_and_remember("I will help you navigate outdoor spaces. Looking for vehicles, people, and traffic.")
        elif mode == 'jarvis':
            self._speak_and_remember("Jarvis mode active. You can ask me questions about your surroundings.")

    def _handle_jarvis_query(self, text: str):
        """
        Rule-based query handler. No LLM needed.
        Scans live detections and replies instantly.
        """
        detections = self._get_last_detections_snapshot()
        asked_class = self._extract_target_object(text)

        if asked_class:
            matches = self._find_detection_matches(asked_class, detections)
            if matches:
                closest = min(matches, key=lambda x: x.get('distance', 999))
                if len(matches) == 1:
                    response = f"I found the {closest['class_name']} {self._describe_detection_location(closest)}."
                else:
                    response = (
                        f"I found {len(matches)} {closest['class_name']} objects. "
                        f"The nearest is {self._describe_detection_location(closest)}."
                    )
                self._speak_and_remember(response, priority=True)
            else:
                self._speak_and_remember(self._build_not_found_response(asked_class, detections), priority=True)
            return

        if not detections:
            self._speak_and_remember("I do not see anything right now.", priority=True)
            return

        counts = {}
        for d in detections:
            cls = d.get('class_name', d.get('class', 'object'))
            counts[cls] = counts.get(cls, 0) + 1
        parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]
        self._speak_and_remember(f"I can see {', '.join(parts)}.", priority=True)


    
    def _fast_distance_fallback(self, detections: list) -> str:
        """Quick rule-based distance answer when LLM is slow/unavailable."""
        if not detections:
            return "I don't see any objects to measure."
        closest = min(detections, key=lambda x: x.get('distance', 999))
        name = closest.get('class_name', closest.get('class', 'object'))
        dist = closest.get('distance', 0)
        return f"The closest object is a {name}, about {dist:.1f} meters away."
    
    def _fast_safety_fallback(self, detections: list) -> str:
        """Quick safety summary when LLM is slow/unavailable."""
        if not detections:
            return "I don't see any obstacles. Proceed with caution."
        close = [d for d in detections if d.get('distance', 999) < 1.5]
        if close:
            names = ", ".join(d.get('class_name', d.get('class', 'object')) for d in close[:3])
            return f"Caution: {names} nearby. Move slowly."
        return "No immediate obstacles. Path looks clear."
    
    def _handle_scene_query(self):
        """Handle 'what do you see' queries."""
        detections = self._get_last_detections_snapshot()
        
        # Check mode - Only use LLM for Jarvis mode
        if self.current_mode != 'jarvis':
            # Fast Path: Simple list of objects
            if not detections:
                self._speak_and_remember("I don't see anything right now.", priority=True)
            else:
                # Group by class
                counts = {}
                for d in detections:
                    cls = d.get('class_name', d.get('class', 'object'))
                    counts[cls] = counts.get(cls, 0) + 1
                
                summary = ", ".join(f"{count} {name}{'s' if count > 1 else ''}" for name, count in counts.items())
                self._speak_and_remember(f"I see {summary}.", priority=True)
            return

        self._speak_and_remember("Let me look around...", priority=True)
        self._llm_busy = True
        try:
            description = self.llm.generate_scene_description(detections)
        finally:
            self._llm_busy = False
        self._speak_and_remember(description or "I don't see anything specific.", priority=True)
    
    def _handle_location_query(self, text: str = ""):
        """Handle "where is X" queries with voice response."""
        enhanced = self._get_last_detections_snapshot()
        object_name = self._extract_target_object(text)

        if not object_name:
            if not enhanced:
                self._speak_and_remember("I do not see anything around you.", priority=True)
                return
            match = min(enhanced, key=lambda x: x.get('distance', 999))
        else:
            matches = self._find_detection_matches(object_name, enhanced)
            if not matches:
                self._speak_and_remember(self._build_not_found_response(object_name, enhanced), priority=True)
                return
            match = min(matches, key=lambda x: x.get('distance', 999))

        response = f"The {match['class_name']} is {self._describe_detection_location(match)}."
        self._speak_and_remember(response, priority=True)

    def _handle_distance_query(self, text: str):
        """Handle 'how far' queries using cached detections for fast, consistent response."""
        enhanced_detections = self._get_last_detections_snapshot()
        self._llm_busy = True
        try:
            if self.llm and self.llm.check_connection():
                response = self.llm.answer_query(text, enhanced_detections)
            else:
                response = self._fast_distance_fallback(enhanced_detections)
        finally:
            self._llm_busy = False
        self._speak_and_remember(response or "I don't have distance information right now.", priority=True)
    
    def _handle_chair_command(self):
        """Intelligent chair reporting."""
        detections = [d for d in self._get_last_detections_snapshot() if 'chair' in d['class_name'] or 'couch' in d['class_name']]
        if not detections:
            self._speak_and_remember("I don't see any chairs or places to sit.")
            return

        count = len(detections)
        if count <= 2:
            # Tell exactly where
            responses = []
            for d in detections:
                responses.append(f"one {d['distance']:.1f} meters to your {d['zone']}")
            self._speak_and_remember(f"Sir, I have located {' and '.join(responses)}.")
        else:
            # Summarize by zone
            zones = {'left': 0, 'center': 0, 'right': 0}
            for d in detections:
                zones[d['zone']] += 1
            zone_msgs = [f"{c} on your {z}" for z, c in zones.items() if c > 0]
            self._speak_and_remember(f"Sir, I count {count} available seats. {' and '.join(zone_msgs)}.")

    def _handle_person_command(self):
        """Detailed person reports grouped by zone."""
        detections = [d for d in self._get_last_detections_snapshot() if 'person' in d['class_name']]
        if not detections:
            self._speak_and_remember("No individuals detected in your immediate vicinity.")
            return

        count = len(detections)
        zones = {'left': 0, 'center': 0, 'right': 0}
        for d in detections:
            zones[d['zone']] += 1
        
        if count == 1:
            zone = list(zones.keys())[0] if len(zones) == 1 else [z for z, c in zones.items() if c > 0][0]
            self._speak_and_remember(f"Sir, I detect one person directly in your {zone} path.")
        else:
            self._speak_and_remember(f"I currently track {count} individuals nearby.")

    def _handle_tv_laptop_command(self):
        """Report tech objects."""
        detections = [d for d in self._get_last_detections_snapshot() if any(x in d['class_name'] for x in ['tv', 'laptop', 'monitor'])]
        if not detections:
            self._speak_and_remember("I don't see any TV or laptop.")
            return
        
        d = detections[0]
        self._speak_and_remember(f"I see a {d['class_name']} {d['distance']:.1f} meters away, {d['direction_text']}.")

    def _handle_door_command(self):
        """Report door and estimate status."""
        detections = [d for d in self._get_last_detections_snapshot() if 'door' in d['class_name']]
        if not detections:
            self._speak_and_remember("I don't see any doors nearby.")
            return
        
        d = detections[0]
        status = "open" if (d['bbox'][2]-d['bbox'][0]) > 200 else "closed"
        self._speak_and_remember(f"Door on your {d['zone']} at {d['distance']:.1f} meters. It appears {status}.")

    def _handle_wall_command(self):
        """Identify closest large obstacle as a wall."""
        enhanced = self._get_last_detections_snapshot()
        close_obstacles = [d for d in enhanced if d['distance'] < 1.5]
        
        if not close_obstacles:
            self._speak_and_remember("No walls or immediate obstacles detected.")
        else:
            d = min(close_obstacles, key=lambda x: x['distance'])
            alert = "High" if d['distance'] < 0.8 else "Medium"
            self._speak_and_remember(f"Wall-like obstacle {d['distance']:.1f} meters {d['direction_text']}. Alert level: {alert}.")

    def _handle_scan_command(self):
        """Comprehensive room scan."""
        detections = self._get_last_detections_snapshot()
        if not detections:
            self._speak_and_remember("I don't see anything to scan right now.")
            return
        
        self._speak_and_remember("Scanning environment...", priority=True)
        # Summarize by zone
        report = []
        for zone in ['left', 'center', 'right']:
            zone_dets = [d for d in detections if d['zone'] == zone]
            if zone_dets:
                names = list(set([d['class_name'] for d in zone_dets]))
                report.append(f"On your {zone}, I see {' and '.join(names[:2])}")
        
        if report:
            self._speak_and_remember(". ".join(report) + ".")
        else:
            self._speak_and_remember("Room looks mostly clear.")

    def _handle_safety_command(self):
        """Direct safety path analysis."""
        detections = self._get_last_detections_snapshot()
        front_obstacles = [d for d in detections if d['zone'] == 'center' and d['distance'] < 2.0]
        
        if not front_obstacles:
            self._speak_and_remember("The path ahead looks mostly clear. Proceed with caution.", priority=True)
        else:
            closest = min(front_obstacles, key=lambda x: x['distance'])
            self._speak_and_remember(f"Caution. There is a {closest['class_name']} only {closest['distance']:.1f} meters ahead.", priority=True)

    def _handle_repeat_command(self):
        """Repeat the last spoken report."""
        if self.last_voice_report:
            print(f"[Command] 🔄 Repeating: {self.last_voice_report}")
            self._speak_and_remember(f"I said: {self.last_voice_report}", priority=True)
        else:
            self._speak_and_remember("I haven't said anything yet.")

    def _handle_help_command(self):
        """List major command categories."""
        guide = "You can ask: Where is something? What do you see? Scan room. Is it safe to walk? Repeat that. Or switch to indoor, outdoor, or jarvis mode."
        self._speak_and_remember(guide, priority=True)
    
    def _main_loop(self):
        """Main detection and processing loop."""
        print("[NavSense] Main loop started. Press 'q' in display window to quit.\n")
        
        last_alert_time = 0
        frame_count = 0
        
        while self.is_running:
            try:
                # Get frame
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process detections
                if self.current_mode:
                    if self._llm_busy:
                        time.sleep(0.02)
                        frame_count += 1
                        continue
                    
                    # Refinement: Process more frames even when speaking
                    skip_rate = 2 if self.voice.is_currently_speaking() else 1
                    if frame_count % skip_rate != 0:
                        frame_count += 1
                        continue

                    detections = self.detector.detect(frame)
                    
                    
                    # Filter by mode
                    modes_config = self.config.get('modes', {})
                    detections = self.detector.filter_by_mode(
                        detections,
                        self.current_mode,
                        modes_config
                    )
                    
                    
                    # Calculate distance and direction
                    enhanced_detections = []
                    for det in detections:
                        enhanced = self.calculator.calculate_all(det)
                        enhanced_detections.append(enhanced)
                    
                    # Update tracker and last_detections with temporal persistence
                    if self.tracker:
                        tracked_objects = self.tracker.update(enhanced_detections)
                    else:
                        tracked_objects = enhanced_detections
                    
                    if tracked_objects:
                        self._empty_frame_count = 0
                        with self._detections_lock:
                            self.last_detections = tracked_objects
                    else:
                        self._empty_frame_count += 1
                        if self._empty_frame_count > self._max_empty_frames_before_clear:
                            with self._detections_lock:
                                self.last_detections = []

                    self.frame_index += 1
                    if self.session_logger:
                        self.session_logger.log_detection_frame(self.frame_index, self.current_mode, tracked_objects)
                    
                    # Evaluate for alerts
                    alerts = self.alert_system.evaluate_detections(tracked_objects)
                    
                    # Speak high-priority alerts
                    current_time = time.time()
                    if current_time - last_alert_time > 1.2:  # Slightly longer throttle for combined speech
                        speakable_alerts = [a for a in alerts[:3] if self.alert_system.should_speak_alert(a)]
                        
                        if speakable_alerts:
                            if self.session_logger:
                                for alert in speakable_alerts:
                                    self.session_logger.log_alert(alert, mode=self.current_mode)
                            if len(speakable_alerts) == 1:
                                # Standard single alert
                                self.voice.speak(speakable_alerts[0]['message'])
                            else:
                                # Combined multi-object alert
                                # Remove redundant "Warning:" etc. prefixes for a cleaner sentence
                                items = []
                                for a in speakable_alerts:
                                    msg = a['message'].split(': ')[-1] if ': ' in a['message'] else a['message']
                                    items.append(msg)
                                
                                # Highest priority prefix
                                top_priority = speakable_alerts[0]['priority'].capitalize()
                                combined_msg = f"{top_priority}: {items[0]} and {items[1]}"
                                if len(items) > 2:
                                    combined_msg = f"{top_priority}: {items[0]}, {items[1]}, and {items[2]}"
                                
                                self._speak_and_remember(combined_msg)
                            
                            last_alert_time = current_time
                
                frame_count += 1
                
                # Small delay to prevent overwhelming CPU
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[NavSense] ERROR in main loop: {e}")
                time.sleep(0.1)
        
        print("\n[NavSense] Main loop terminated")
    
    def _display_loop(self):
        """Display loop for camera feed with overlays."""
        print("[Display] Display window started")
        
        while self.is_running:
            try:
                # Never freeze the display, even if speaking
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                display_frame = frame.copy()
                
                # Only show detections if mode is selected
                # Only show detections if mode is selected
                if self.current_mode:
                    detections = self._get_last_detections_snapshot()
                    
                    # Filter just in case config changed, but usually main loop handles it
                    # (Actually main loop already filtered 'last_detections' by mode, 
                    # so we just draw them directly if they are robust enough)
                    
                    # Draw detections
                    for det in detections:
                        # det is already 'enhanced' in main loop
                        if 'distance' not in det:
                            # In case of race condition where mains loop put raw det
                            det = self.calculator.calculate_all(det)
                        self._draw_detection(display_frame, det)
                
                # Add mode indicator
                mode_text = f"Mode: {self.current_mode or 'Not Selected'}"
                cv2.putText(display_frame, mode_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add FPS
                fps_text = f"FPS: {self.camera.get_fps():.1f}"
                cv2.putText(display_frame, fps_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('NavSense - Phase 1', display_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break
                
            except Exception as e:
                print(f"[Display] ERROR: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        print("[Display] Display window closed")
    
    def _draw_detection(self, frame, detection: dict):
        """Draw detection on frame with labels that don't get cut off."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        distance = detection.get('distance', 0)
        direction_text = detection.get('direction_text', '')
        
        # Determine color based on distance
        if distance < 1.0:
            color = (0, 0, 255)  # Red - close
        elif distance < 2.5:
            color = (0, 165, 255)  # Orange - medium
        else:
            color = (255, 0, 0)  # Blue - far
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 1. Top Label (Class & Confidence)
        label = f"{class_name} {confidence:.2f}"
        (l_w, l_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background (top) - SOLID BLACK for 100% contrast
        label_y = y1 if y1 > l_h + 10 else y1 + l_h + 10
        cv2.rectangle(frame, (x1, label_y - l_h - 5), (x1 + l_w + 5, label_y), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 2. Bottom Label (Distance & Direction)
        info = f"{distance:.1f}m {direction_text}"
        (i_w, i_h), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        # info_y logic: If too close to bottom, draw it INSIDE the box at the bottom
        if y2 + i_h + 10 > h:
            info_y = y2 - 5
        else:
            info_y = y2 + i_h + 5
            
        # Draw info background
        cv2.rectangle(frame, (x1, info_y - i_h - 2), (x1 + i_w + 5, info_y + 2), (0, 0, 0), -1)
        cv2.putText(frame, info, (x1 + 2, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def shutdown(self):
        """Shutdown NavSense application."""
        print("\n[NavSense] Shutting down...")
        self.is_running = False
        
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        
        if self.voice:
            self.voice.shutdown()
        
        if self.camera:
            self.camera.release()
        
        if self.session_logger:
            self.session_logger.log_event("system", message="NavSense shutdown")
            self.session_logger.close()
        
        cv2.destroyAllWindows()
        print("[NavSense] Shutdown complete\n")


def main():
    """Main entry point."""
    navsense = NavSense()
    
    try:
        # Load configuration
        if not navsense.load_config():
            print("Failed to load configuration. Exiting.")
            return 1
        
        # Initialize components
        if not navsense.initialize_components():
            print("Failed to initialize components. Exiting.")
            return 1
        
        # Start application
        navsense.start()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        navsense.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
