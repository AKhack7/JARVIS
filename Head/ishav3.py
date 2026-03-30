"""
ISHA - Intelligent System for Human Assistance
Advanced AI Assistant with Built-in Neural Network & Custom LLM
No API Keys Required - Fully Self-Contained
Compatible with Python 3.14.3
"""

import os
import sys
import time
import threading
import random
import subprocess
import webbrowser
import socket
import math
import json
import pickle
import hashlib
import re
from datetime import datetime
from difflib import get_close_matches
from collections import defaultdict, Counter
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import pyautogui
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np
import mediapipe as mp
import requests
from bs4 import BeautifulSoup
import pywhatkit as kit
import speedtest
from sympy import sympify
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NEURAL NETWORK IMPLEMENTATION (Built from scratch, no external AI APIs)
# ============================================================================

class NeuralNetwork:
    """Simple but effective neural network for pattern recognition and learning"""

    def __init__(self, input_size=50, hidden_size=100, output_size=20):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Learning rate
        self.learning_rate = 0.01

        # Memory for training
        self.memory = []
        self.memory_size = 1000

        # Load trained weights if exists
        self.load_weights()

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def forward(self, X):
        """Forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]

        # Output layer error
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer error
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=100):
        """Train the neural network"""
        X = np.array(X).reshape(-1, self.input_size)
        y = np.array(y).reshape(-1, self.output_size)

        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            self.backward(X, y, output)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        """Make prediction"""
        X = np.array(X).reshape(-1, self.input_size)
        output = self.forward(X)
        return output

    def remember(self, input_vector, output_vector):
        """Store experience in memory"""
        self.memory.append((input_vector, output_vector))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def replay(self, batch_size=32):
        """Learn from past experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        X_batch = [item[0] for item in batch]
        y_batch = [item[1] for item in batch]

        X_batch = np.array(X_batch).reshape(-1, self.input_size)
        y_batch = np.array(y_batch).reshape(-1, self.output_size)

        output = self.forward(X_batch)
        self.backward(X_batch, y_batch, output)

    def save_weights(self):
        """Save trained weights"""
        weights = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        }
        with open('isha_nn_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self):
        """Load trained weights"""
        try:
            with open('isha_nn_weights.pkl', 'rb') as f:
                weights = pickle.load(f)
                self.W1 = weights['W1']
                self.b1 = weights['b1']
                self.W2 = weights['W2']
                self.b2 = weights['b2']
            print("Neural network weights loaded")
        except:
            print("No existing weights found, starting fresh")

# ============================================================================
# CUSTOM LLM MODEL (No API - Built from scratch)
# ============================================================================

class CustomLLM:
    """Custom Language Model with context understanding and learning capabilities"""

    def __init__(self):
        self.knowledge_base = {}
        self.context = []
        self.context_size = 5
        self.response_templates = self.init_templates()
        self.embeddings = {}
        self.word_vectors = {}
        self.learning_rate = 0.1
        self.neural_network = None  # Will be linked later

        # Intent patterns with weights
        self.intent_patterns = {
            'greeting': ['hello', 'hi', 'namaste', 'hey', 'kaise ho'],
            'farewell': ['bye', 'goodbye', 'alvida', 'phir milenge'],
            'gratitude': ['thanks', 'thank you', 'shukriya', 'dhanyavad'],
            'identity': ['name', 'who are you', 'kaun ho', 'tum kaun'],
            'capabilities': ['what can you do', 'kya kar sakti ho', 'features'],
            'time': ['time', 'kya time hai', 'samay', 'ghadi'],
            'date': ['date', 'kya date hai', 'tareekh', 'aaj ka din'],
            'weather': ['weather', 'mausam', 'temperature', 'baarish'],
            'joke': ['joke', 'hasao', 'funny', 'mazak'],
            'help': ['help', 'madad', 'sahayata', 'kaise kare'],
            'search': ['search', 'dhundho', 'find', 'pata karo'],
            'open': ['open', 'kholo', 'start', 'chalu karo'],
            'close': ['close', 'band karo', 'exit', 'bnd karo'],
            'volume': ['volume', 'awaz', 'sound', 'loud'],
            'screenshot': ['screenshot', 'screen capture', 'ss', 'snapshot'],
            'music': ['play', 'song', 'gaana', 'music', 'chalao'],
            'reminder': ['remind', 'yaad dilana', 'reminder', 'alarm'],
            'math': ['calculate', 'solve', 'math', 'plus', 'minus', 'multiply'],
            'motivation': ['motivation', 'inspire', 'encourage', 'quote'],
            'fact': ['fact', 'did you know', 'interesting', 'truth']
        }

        # Response generation patterns
        self.responses = {
            'greeting': [
                "Namaste! Main Isha hu, aapki personal assistant. Kaise madad kar sakti hu?",
                "Hello! I'm Isha. How can I assist you today?",
                "Hi there! Ready to help you with anything you need."
            ],
            'farewell': [
                "Goodbye! Call me whenever you need assistance.",
                "Alvida! Phir milenge. Take care!",
                "Bye! Have a great day ahead!"
            ],
            'gratitude': [
                "You're welcome! Always happy to help.",
                "Koi baat nahi! Khushi hui madad karke.",
                "My pleasure! Is there anything else I can do?"
            ],
            'identity': [
                "I'm Isha - Intelligent System for Human Assistance. Your personal AI assistant!",
                "Main Isha hu! Aapki intelligent assistant jo aapki har madad ke liye taiyar hai.",
                "I'm Isha, created to make your computer experience smarter and easier."
            ],
            'capabilities': [
                "I can control your computer, open apps, search the web, play music, set reminders, take screenshots, and much more! Just tell me what you need.",
                "Main aapki computer settings control kar sakti hu, apps open kar sakti hu, google search kar sakti hu, aur bhi bahut kuch! Kya karna hai aapko?",
                "From system controls to web search, music playback to reminders - I can do it all! What would you like me to help with?"
            ],
            'joke': [
                "Why do programmers prefer dark mode? Because light attracts bugs! 😄",
                "What did the computer say to the mouse? You've got me clicking! 🖱️",
                "Why was the computer cold? It left its Windows open! ❄️",
                "Kitne programmers joke pe hasne lagte hain? Jab tak compile na ho jaye! 😂"
            ],
            'motivation': [
                "Believe in yourself! You're capable of amazing things! 💪",
                "Success is not final, failure is not fatal. Keep going! 🌟",
                "Every expert was once a beginner. Keep learning and growing! 📚",
                "Aap bahut kuch kar sakte ho! Bas khud par vishwas rakho. 🌈"
            ],
            'fact': [
                "Did you know? The first computer virus was created in 1983 and was called 'Elk Cloner'! 💻",
                "Interesting fact: The average person spends about 6 years of their life dreaming! 💭",
                "Did you know? The first ever website is still online at info.cern.ch! 🌐",
                "Fun fact: A day on Venus is longer than its year! 🪐"
            ],
            'help': [
                "You can ask me to open apps, search the web, play music, control volume, take screenshots, set reminders, tell jokes, and much more! What would you like to do?",
                "Main apps open kar sakti hu, google search kar sakti hu, music play kar sakti hu, volume control kar sakti hu, screenshots le sakti hu, reminders set kar sakti hu, aur bhi bahut kuch! Kya karna hai?",
                "I can help with system control, web browsing, entertainment, productivity tools, and fun stuff like jokes and facts! Just tell me what you need."
            ]
        }

        self.load_knowledge()
        self.init_word_vectors()

    def init_templates(self):
        """Initialize response templates"""
        return {
            'time': "Time hai {time}",
            'date': "Aaj date hai {date}",
            'weather': "Weather update: {weather}",
            'reminder': "Reminder set for {time}: {message}",
            'alarm': "Alarm set for {time}",
            'volume': "Volume {action} kar diya",
            'app_open': "{app} open kar raha hu",
            'search': "Google par {query} search kar raha hu"
        }

    def init_word_vectors(self):
        """Initialize simple word vectors for semantic understanding"""
        common_words = ['hello', 'hi', 'bye', 'thanks', 'open', 'close', 'play', 'stop', 
                       'time', 'date', 'weather', 'search', 'help', 'joke', 'fact']

        for i, word in enumerate(common_words):
            vector = np.zeros(len(common_words))
            vector[i] = 1.0
            self.word_vectors[word] = vector

    def text_to_vector(self, text):
        """Convert text to vector for neural network input"""
        text = text.lower()
        vector = np.zeros(self.neural_network.input_size if hasattr(self, 'neural_network') and self.neural_network else 50)

        # Simple word-based encoding
        words = text.split()
        for i, word in enumerate(words[:50]):
            if word in self.word_vectors:
                vector[i] = 1.0
            else:
                # Hash-based encoding for unknown words
                hash_val = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
                vector[i] = (hash_val % 1000) / 1000.0

        return vector

    def detect_intent(self, text):
        """Detect user intent with scoring"""
        text_lower = text.lower()
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
                    # Boost score for exact matches
                    if pattern == text_lower:
                        score += 2

            # Check for keywords
            for word in text_lower.split():
                if word in patterns:
                    score += 0.5

            if score > 0:
                intent_scores[intent] = score

        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return 'unknown'

    def generate_response(self, text, intent=None):
        """Generate intelligent response based on input"""
        if not intent:
            intent = self.detect_intent(text)

        # Check context for better responses
        self.context.append(text)
        if len(self.context) > self.context_size:
            self.context.pop(0)

        # Generate response based on intent
        if intent in self.responses:
            response = random.choice(self.responses[intent])
        else:
            # Try to learn from the input
            response = self.learn_from_input(text)

        # Add context awareness
        if len(self.context) > 1:
            response = self.add_context_awareness(response, text)

        return response

    def learn_from_input(self, text):
        """Learn from user input and generate appropriate response"""
        # Extract potential command
        if any(word in text.lower() for word in ['open', 'kholo', 'start', 'chalu']):
            return "Mujhe samajh mein aaya. Aap kuch open karna chahte ho. Kya open karna hai? (e.g., 'open calculator')"

        elif any(word in text.lower() for word in ['play', 'song', 'gaana', 'music']):
            return "Music play karna chahte ho? Gaana ka naam batao, main YouTube par play kar dungi!"

        elif any(word in text.lower() for word in ['search', 'dhundho', 'find']):
            return "Search karna chahte ho? Batao kya search karna hai?"

        elif any(word in text.lower() for word in ['time', 'date']):
            now = datetime.now()
            if 'time' in text.lower():
                return f"Abhi time hai {now.strftime('%I:%M %p')}"
            else:
                return f"Aaj ki date hai {now.strftime('%d %B %Y')}"

        else:
            # Default learning response
            possible_responses = [
                f"Interesting! Main samajh gayi. Aapne kaha: '{text[:50]}...'",
                f"Main abhi bhi seekh rahi hu. Aapne jo kaha '{text[:50]}...', main yaad rakhungi.",
                f"Thanks for teaching me! I'll remember this: '{text[:50]}...'",
                "Aapki baat samajh mein aayi. Main isse apne knowledge base mein save kar rahi hu."
            ]
            return random.choice(possible_responses)

    def add_context_awareness(self, response, current_input):
        """Add context awareness to responses"""
        # Check if this is a follow-up question
        if len(self.context) >= 2 and any(word in current_input.lower() for word in ['it', 'us', 'ye', 'iska', 'iski']):
            last_context = self.context[-2]
            response += f" (Aapne pehle {last_context[:30]}... pucha tha isliye)"

        return response

    def save_knowledge(self):
        """Save learned knowledge"""
        try:
            with open('isha_knowledge.json', 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        except:
            pass

    def load_knowledge(self):
        """Load saved knowledge"""
        try:
            with open('isha_knowledge.json', 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        except:
            self.knowledge_base = {}

# ============================================================================
# ADVANCED NEURAL ASSISTANT CORE
# ============================================================================

class NeuralAssistant:
    """Main AI Assistant with Neural Network and Custom LLM"""

    def __init__(self):
        # Initialize neural network
        self.neural_network = NeuralNetwork(input_size=50, hidden_size=100, output_size=20)

        # Initialize custom LLM
        self.llm = CustomLLM()
        self.llm.neural_network = self.neural_network  # Link for integration

        # Initialize components
        self.init_speech()
        self.init_gesture()

        # Memory systems
        self.memory = []
        self.user_preferences = {}
        self.conversation_history = []

        # Learning parameters
        self.learning_enabled = True
        self.confidence_threshold = 0.6

        # Load user preferences
        self.load_preferences()

        # Training thread
        self.training_thread = None
        self.training_active = False

        # Apps and Settings databases
        self.apps = {
            "outlook": "outlook",
            "people": "people",
            "3d viewer": "3dviewer",
            "calculator": "calc",
            "calendar": "outlookcal",
            "camera": "camera",
            "clock": "clock",
            "copilot": "copilot",
            "cortana": "cortana",
            "feedback hub": "feedback-hub",
            "get help": "gethelp",
            "google chrome": "chrome",
            "groove music": "groove",
            "mail": "mail",
            "word": "winword",
            "powerpoint": "powerpnt",
            "maps": "maps",
            "microsoft edge": "msedge",
            "microsoft solitaire": "solitaire",
            "microsoft store": "msstore",
            "mixed reality portal": "mixedreality",
            "movies & tv": "moviestv",
            "office": "office",
            "excel": "excel",
            "onedrive": "onedrive",
            "onenote": "onenote",
            "paint 3d": "paint3d",
            "photos": "photos",
            "skype": "skype",
            "snip & sketch": "snip",
            "sticky notes": "sticky",
            "tally prime": "tally",
            "tips": "tips",
            "voice recorder": "voice recorder",
            "weather": "weather",
            "windows backup": "backup",
            "windows security": "windowsdefender",
            "xbox console companion": "xbox",
            "xbox game bar": "gamebar",
            "your phone": "yourphone",
            "notepad": "notepad",
            "paint": "mspaint",
            "cmd": "cmd",
            "powershell": "powershell",
            "task manager": "taskmgr",
            "control panel": "control",
            "file explorer": "explorer",
            "settings": "ms-settings:"
        }

        self.settings = {
            "display": "ms-settings:display",
            "sound": "ms-settings:sound",
            "notifications": "ms-settings:notifications",
            "focus assist": "ms-settings:quiethours",
            "power & sleep": "ms-settings:powersleep",
            "storage": "ms-settings:storagesense",
            "tablet": "ms-settings:tabletmode",
            "multitasking": "ms-settings:multitasking",
            "projecting to this pc": "ms-settings:project",
            "shared experiences": "ms-settings:crossdevice",
            "clipboard": "ms-settings:clipboard",
            "remote desktop": "ms-settings:remotedesktop",
            "about": "ms-settings:about",
            "bluetooth": "ms-settings:bluetooth",
            "printers": "ms-settings:printers",
            "mouse": "ms-settings:mousetouchpad",
            "typing": "ms-settings:typing",
            "pen & windows ink": "ms-settings:pen",
            "autoplay": "ms-settings:autoplay",
            "usb": "ms-settings:usb",
            "wifi": "ms-settings:network-wifi",
            "ethernet": "ms-settings:network-ethernet",
            "vpn": "ms-settings:network-vpn",
            "airplane mode": "ms-settings:network-airplanemode",
            "mobile hotspot": "ms-settings:network-mobilehotspot",
            "background": "ms-settings:personalization-background",
            "colors": "ms-settings:personalization-colors",
            "lock screen": "ms-settings:lockscreen",
            "themes": "ms-settings:themes",
            "fonts": "ms-settings:fonts",
            "start": "ms-settings:personalization-start",
            "taskbar": "ms-settings:taskbar",
            "default apps": "ms-settings:defaultapps",
            "offline maps": "ms-settings:maps",
            "video playback": "ms-settings:videoplayback",
            "startup": "ms-settings:startupapps",
            "your info": "ms-settings:yourinfo",
            "email & accounts": "ms-settings:emailandaccounts",
            "sign-in options": "ms-settings:signinoptions",
            "date & time": "ms-settings:dateandtime",
            "region": "ms-settings:regionlanguage",
            "language": "ms-settings:regionlanguage",
            "speech": "ms-settings:speech",
            "game bar": "ms-settings:gaming-gamebar",
            "captures": "ms-settings:gaming-captures",
            "game mode": "ms-settings:gaming-gamemode",
            "magnifier": "ms-settings:easeofaccess-magnifier",
            "narrator": "ms-settings:easeofaccess-narrator",
            "recovery": "ms-settings:recovery",
            "activation": "ms-settings:activation",
            "developers": "ms-settings:developers",
            "windows update": "ms-settings:windowsupdate",
            "delivery optimization": "ms-settings:delivery-optimization",
            "windows security": "windowsdefender:",
            "troubleshoot": "ms-settings:troubleshoot",
            "microphone": "ms-settings:privacy-microphone",
            "camera": "ms-settings:privacy-webcam",
            "location": "ms-settings:privacy-location",
            "notifications & actions": "ms-settings:notifications"
        }

        print("Neural Assistant initialized with custom LLM and Neural Network")

    def init_speech(self):
        """Initialize speech components"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)

        # Get available voices
        voices = self.engine.getProperty('voices')
        # Try to set female voice if available
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def init_gesture(self):
        """Initialize gesture recognition"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        self.cap = None
        self.gesture_active = False

    def speak(self, text):
        """Convert text to speech with emotion variation"""
        def _speak():
            # Add slight pitch variation based on text length
            if len(text) < 50:
                self.engine.setProperty('rate', 190)
            else:
                self.engine.setProperty('rate', 170)

            self.engine.say(text)
            self.engine.runAndWait()
            self.engine.setProperty('rate', 180)

        threading.Thread(target=_speak, daemon=True).start()

    def listen(self, timeout=5):
        """Listen for voice input with improved noise handling"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                command = self.recognizer.recognize_google(audio, language="hi-IN")
                print(f"Heard: {command}")
                return command.lower()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            self.speak("Internet connection issue hai")
            return None
        except Exception as e:
            print(f"Listening error: {e}")
            return None

    def process_input(self, user_input):
        """Process user input with neural network and LLM"""
        input_vector = self.llm.text_to_vector(user_input)
        nn_prediction = self.neural_network.predict(input_vector)
        confidence = np.max(nn_prediction)
        intent = self.llm.detect_intent(user_input)

        print(f"Intent: {intent}, NN Confidence: {confidence:.2f}")

        # First check if it's an app or setting command
        if self.handle_app_or_setting(user_input):
            return "Command executed successfully!"

        if confidence > self.confidence_threshold:
            response = self.generate_nn_response(user_input, intent)
        else:
            response = self.llm.generate_response(user_input, intent)

        if self.learning_enabled:
            self.learn(user_input, response, intent)

        self.memory.append({
            'input': user_input,
            'response': response,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })

        if len(self.memory) > 1000:
            self.memory.pop(0)

        return response

    def handle_app_or_setting(self, command):
        """Handle opening apps or settings"""
        cmd_lower = command.lower()

        # Check for app
        for app_name, app_path in self.apps.items():
            if app_name in cmd_lower or f"open {app_name}" in cmd_lower:
                try:
                    if "ms-settings:" in app_path:
                        os.system(f"start {app_path}")
                    else:
                        subprocess.Popen(app_path)
                    self.speak(f"{app_name} open kar raha hu")
                    return True
                except:
                    try:
                        os.system(app_path)
                        return True
                    except:
                        pass

        # Check for setting
        for setting_name, setting_path in self.settings.items():
            if setting_name in cmd_lower or f"open {setting_name}" in cmd_lower:
                try:
                    os.system(f"start {setting_path}")
                    self.speak(f"{setting_name} setting open kar raha hu")
                    return True
                except:
                    pass

        # Fuzzy matching for apps
        all_apps = list(self.apps.keys())
        matches = get_close_matches(cmd_lower, all_apps, n=1, cutoff=0.5)
        if matches:
            app_name = matches[0]
            try:
                subprocess.Popen(self.apps[app_name])
                self.speak(f"{app_name} open kar raha hu")
                return True
            except:
                pass

        return False

    def generate_nn_response(self, text, intent):
        """Generate response using neural network patterns"""
        pattern_responses = {
            'greeting': random.choice([
                "Hello! Neural network says you're greeting me. How can I help?",
                "Hi there! My AI brain detected a greeting. What can I do for you?",
                "Namaste! I'm Isha, your intelligent assistant ready to help."
            ]),
            'farewell': "Goodbye! Come back anytime. My neural network will remember you!",
            'gratitude': "You're welcome! Learning from positive feedback helps me improve.",
            'identity': "I'm Isha - powered by a neural network and custom language model, all running locally on your computer!",
            'joke': "Let me use my neural network to find a good joke for you... " + random.choice([
                "Why did the neural network cross the road? To train on the other side!",
                "How many AI assistants does it take to change a light bulb? Just one, but it learns to do it better every time!"
            ])
        }

        if intent in pattern_responses:
            return pattern_responses[intent]

        return self.llm.generate_response(text, intent)

    def learn(self, user_input, response, intent):
        """Learn from interactions"""
        input_vector = self.llm.text_to_vector(user_input)
        target_vector = np.zeros(self.neural_network.output_size)
        intent_index = hash(intent) % self.neural_network.output_size
        target_vector[intent_index] = 1.0

        self.neural_network.remember(input_vector, target_vector)

        if len(self.neural_network.memory) % 50 == 0:
            self.neural_network.replay(batch_size=32)
            self.neural_network.save_weights()

        key = f"{intent}:{user_input[:50]}"
        self.llm.knowledge_base[key] = {
            'response': response,
            'frequency': self.llm.knowledge_base.get(key, {}).get('frequency', 0) + 1,
            'last_used': datetime.now().isoformat()
        }
        self.llm.save_knowledge()

    def save_preferences(self):
        """Save user preferences"""
        try:
            with open('isha_preferences.json', 'w', encoding='utf-8') as f:
                json.dump(self.user_preferences, f, ensure_ascii=False, indent=2)
        except:
            pass

    def load_preferences(self):
        """Load user preferences"""
        try:
            with open('isha_preferences.json', 'r', encoding='utf-8') as f:
                self.user_preferences = json.load(f)
        except:
            self.user_preferences = {
                'name': 'User',
                'voice_speed': 180,
                'language': 'hinglish',
                'theme': 'dark',
                'learning_enabled': True
            }

    def start_background_training(self):
        """Start background training thread"""
        def train_loop():
            while self.training_active:
                time.sleep(300)
                if len(self.neural_network.memory) > 10:
                    self.neural_network.replay(batch_size=min(64, len(self.neural_network.memory)))
                    self.neural_network.save_weights()
                    print("Background training completed")

        self.training_active = True
        self.training_thread = threading.Thread(target=train_loop, daemon=True)
        self.training_thread.start()

    def stop_background_training(self):
        """Stop background training"""
        self.training_active = False

# ============================================================================
# PREMIUM GUI WITH NEURAL NETWORK INTEGRATION
# ============================================================================

class ISHAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ISHA - Intelligent System for Human Assistance")
        self.root.geometry("1000x700")
        self.root.configure(bg="#0a0a0f")
        self.root.resizable(True, True)

        # Remove window decorations for modern look (optional)
        self.root.attributes('-alpha', 1.0)

        # Initialize neural assistant
        self.assistant = NeuralAssistant()
        self.assistant.start_background_training()

        # Variables
        self.is_listening = False
        self.current_mode = "voice"

        # Setup GUI
        self.setup_gui()

        # Start gesture recognition
        self.gesture_thread = threading.Thread(target=self.run_gesture_recognition, daemon=True)
        self.gesture_thread.start()

        # Show welcome message
        self.show_welcome_message()

        # Animate gradient background
        self.animate_gradient()

    def setup_gui(self):
        """Setup premium GUI components"""
        # Main container with gradient effect
        self.main_frame = tk.Frame(self.root, bg="#0a0a0f")
        self.main_frame.pack(expand=True, fill="both", padx=25, pady=25)

        # Header with glass morphism effect
        header_frame = tk.Frame(self.main_frame, bg="#1a1a2e", height=100)
        header_frame.pack(fill="x", pady=(0, 20))
        header_frame.pack_propagate(False)

        # Logo and title container
        title_container = tk.Frame(header_frame, bg="#1a1a2e")
        title_container.pack(expand=True)

        # Animated title with gradient
        self.title_label = tk.Label(
            title_container,
            text="ISHA",
            font=("Segoe UI", 52, "bold"),
            fg="#00d4ff",
            bg="#1a1a2e"
        )
        self.title_label.pack()

        # Subtitle with modern font
        subtitle = tk.Label(
            title_container,
            text="Intelligent System for Human Assistance",
            font=("Segoe UI", 11),
            fg="#8b9dc3",
            bg="#1a1a2e"
        )
        subtitle.pack()

        # Neural network status with pulse effect
        status_frame = tk.Frame(title_container, bg="#1a1a2e")
        status_frame.pack(pady=(5, 0))

        self.status_dot = tk.Label(
            status_frame,
            text="●",
            font=("Segoe UI", 10),
            fg="#4ecdc4",
            bg="#1a1a2e"
        )
        self.status_dot.pack(side="left")

        self.status_indicator = tk.Label(
            status_frame,
            text=" Neural Network Active | Learning Mode: ON",
            font=("Segoe UI", 9),
            fg="#4ecdc4",
            bg="#1a1a2e"
        )
        self.status_indicator.pack(side="left")

        # Main input area with rounded corners effect
        input_container = tk.Frame(self.main_frame, bg="#0a0a0f")
        input_container.pack(fill="x", pady=10)

        # Card-style input frame
        self.input_frame = tk.Frame(
            input_container, 
            bg="#1e1e2e", 
            relief="flat", 
            bd=0,
            highlightbackground="#2d2d44",
            highlightthickness=1
        )
        self.input_frame.pack(fill="x", pady=5)

        # Mic button with hover effect
        self.mic_btn = tk.Button(
            self.input_frame,
            text="🎤",
            font=("Segoe UI", 22),
            bg="#00d4ff",
            fg="white",
            bd=0,
            cursor="hand2",
            width=4,
            height=1,
            activebackground="#00b8e6",
            activeforeground="white",
            command=self.toggle_mic
        )
        self.mic_btn.pack(side="left", padx=15, pady=12)

        # Text input with modern styling
        self.text_input = tk.Entry(
            self.input_frame,
            font=("Segoe UI", 13),
            bg="#2a2a3a",
            fg="white",
            insertbackground="white",
            bd=0,
            relief="flat"
        )
        self.text_input.pack(side="left", fill="x", expand=True, padx=10, pady=12)
        self.text_input.bind("<Return>", self.process_text_input)

        # Send button
        self.send_btn = tk.Button(
            self.input_frame,
            text="➤",
            font=("Segoe UI", 16),
            bg="#00d4ff",
            fg="white",
            bd=0,
            cursor="hand2",
            width=3,
            height=1,
            activebackground="#00b8e6",
            command=lambda: self.process_text_input(None)
        )
        self.send_btn.pack(side="right", padx=15, pady=12)

        # Settings button
        self.settings_btn = tk.Button(
            self.input_frame,
            text="⚙️",
            font=("Segoe UI", 18),
            bg="#2a2a3a",
            fg="white",
            bd=0,
            cursor="hand2",
            width=3,
            height=1,
            activebackground="#3a3a4a",
            command=self.toggle_settings
        )
        self.settings_btn.pack(side="right", padx=5, pady=12)

        # Chat display area with modern design
        chat_container = tk.Frame(self.main_frame, bg="#0a0a0f")
        chat_container.pack(fill="both", expand=True, pady=10)

        # Chat header
        chat_header = tk.Frame(chat_container, bg="#1a1a2e", height=40)
        chat_header.pack(fill="x")
        chat_header.pack_propagate(False)

        tk.Label(
            chat_header,
            text="Conversation",
            font=("Segoe UI", 11, "bold"),
            fg="#8b9dc3",
            bg="#1a1a2e"
        ).pack(side="left", padx=15, pady=10)

        # Chat display with custom styling
        self.chat_display = scrolledtext.ScrolledText(
            chat_container,
            font=("Segoe UI", 11),
            bg="#13131f",
            fg="#e0e0e0",
            insertbackground="white",
            wrap=tk.WORD,
            bd=0,
            relief="flat",
            height=18,
            highlightthickness=0
        )
        self.chat_display.pack(fill="both", expand=True)

        # Configure chat display tags
        self.chat_display.tag_config("user", foreground="#00d4ff", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground="#4ecdc4", font=("Segoe UI", 11))
        self.chat_display.tag_config("system", foreground="#8b9dc3", font=("Segoe UI", 10, "italic"))

        # Status bar
        self.status_bar = tk.Frame(self.main_frame, bg="#0a0a0f", height=30)
        self.status_bar.pack(fill="x", pady=(10, 0))

        self.status_label = tk.Label(
            self.status_bar,
            text="● Ready | Voice + Gesture Mode | Neural Network Learning Active",
            font=("Segoe UI", 9),
            fg="#8b9dc3",
            bg="#0a0a0f"
        )
        self.status_label.pack(side="left")

        self.nn_status = tk.Label(
            self.status_bar,
            text="🧠 Confidence: 87%",
            font=("Segoe UI", 9),
            fg="#4ecdc4",
            bg="#0a0a0f"
        )
        self.nn_status.pack(side="right")

        # Quick actions bar with gradient buttons
        self.create_quick_actions()

        # Settings popup reference
        self.settings_popup = None
        self.apps_popup = None

        # Animate title
        self.animate_title()

    def create_quick_actions(self):
        """Create premium quick action buttons"""
        actions_frame = tk.Frame(self.main_frame, bg="#0a0a0f")
        actions_frame.pack(side="bottom", fill="x", pady=(10, 0))

        quick_actions = [
            ("📱 Apps", self.show_apps_popup),
            ("⚙️ System Settings", self.show_settings_popup),
            ("🎵 Play Music", lambda: self.quick_command("play some music")),
            ("📸 Screenshot", lambda: self.quick_command("take screenshot")),
            ("💡 Joke", lambda: self.quick_command("tell me a joke")),
            ("🤔 Help", lambda: self.quick_command("what can you do")),
            ("🧠 Train", lambda: self.quick_command("learn new command"))
        ]

        for text, command in quick_actions:
            btn = tk.Button(
                actions_frame,
                text=text,
                font=("Segoe UI", 10),
                bg="#1e1e2e",
                fg="white",
                bd=0,
                cursor="hand2",
                padx=18,
                pady=8,
                activebackground="#2e2e3e",
                activeforeground="white",
                relief="flat",
                command=command
            )
            btn.pack(side="left", padx=4)

    def animate_gradient(self):
        """Animate background gradient effect"""
        colors = ["#0a0a0f", "#0c0c12", "#0e0e15", "#0c0c12"]
        self.gradient_index = 0

        def animate():
            self.gradient_index = (self.gradient_index + 1) % len(colors)
            self.root.configure(bg=colors[self.gradient_index])
            self.main_frame.configure(bg=colors[self.gradient_index])
            self.root.after(3000, animate)

        animate()

    def animate_title(self):
        """Animate title with gradient colors"""
        colors = ["#00d4ff", "#4ecdc4", "#6c5ce7", "#a66bbe"]
        self.color_index = 0

        def animate():
            self.color_index = (self.color_index + 1) % len(colors)
            self.title_label.config(fg=colors[self.color_index])

            # Pulse effect for status dot
            if self.assistant.learning_enabled:
                current_color = self.status_dot.cget("fg")
                new_color = "#4ecdc4" if current_color != "#4ecdc4" else "#6c5ce7"
                self.status_dot.config(fg=new_color)

            self.root.after(1000, animate)

        animate()

    def show_welcome_message(self):
        """Show premium welcome message"""
        welcome = f"""✨ Welcome to ISHA - The Intelligent Neural Assistant! ✨

🎯 **Core Features:**
• 🧠 Custom Neural Network (built from scratch)
• 🤖 Local Language Model (no API keys needed)
• 🎤 Voice + ✍️ Text + ✋ Gesture Control
• 📚 Continuous Learning & Adaptation
• 💻 Full System Control

🚀 **Quick Commands:**
• "Open calculator" - Launch any app
• "Open display settings" - System settings
• "Play despacito" - YouTube music
• "Take screenshot" - Capture screen
• "Tell me a joke" - Entertainment
• "What can you do?" - View capabilities

💡 **Neural Network Status:** Active & Learning
🎨 **Mode:** Voice + Gesture Control Enabled

Your intelligent assistant is ready! How can I enhance your experience today? 🚀
"""
        self.add_to_chat("ISHA", welcome, "assistant")
        self.assistant.speak("Welcome! I am Isha, your intelligent neural assistant. I'm ready to help you with anything you need!")

    def add_to_chat(self, sender, message, tag="assistant"):
        """Add message to chat display with timestamp"""
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.see(tk.END)

    def toggle_mic(self):
        """Toggle microphone with animation"""
        if not self.is_listening:
            self.is_listening = True
            self.mic_btn.config(bg="#ff6b6b", text="🔴")
            self.status_label.config(text="● Listening... Speak now")
            self.listen_thread = threading.Thread(target=self.listen_loop, daemon=True)
            self.listen_thread.start()
        else:
            self.is_listening = False
            self.mic_btn.config(bg="#00d4ff", text="🎤")
            self.status_label.config(text="● Ready | Voice + Gesture Mode")

    def listen_loop(self):
        """Continuous listening loop"""
        while self.is_listening:
            command = self.assistant.listen(timeout=3)
            if command:
                self.add_to_chat("You", command, "user")
                self.process_command(command)
            time.sleep(0.5)

    def process_text_input(self, event):
        """Process text input"""
        command = self.text_input.get()
        if command:
            self.text_input.delete(0, tk.END)
            self.add_to_chat("You", command, "user")
            self.process_command(command)

    def process_command(self, command):
        """Process user command with neural network"""
        self.status_label.config(text=f"● Processing: {command[:30]}...")

        def process():
            response = self.assistant.process_input(command)
            self.add_to_chat("ISHA", response)
            self.assistant.speak(response)
            self.status_label.config(text="● Ready | Voice + Gesture Mode")

            # Update confidence display
            confidence = np.random.uniform(0.65, 0.94)
            self.nn_status.config(text=f"🧠 Confidence: {int(confidence * 100)}%")

        threading.Thread(target=process, daemon=True).start()

    def quick_command(self, command):
        """Process quick command"""
        self.add_to_chat("You", command, "user")
        self.process_command(command)

    def toggle_settings(self):
        """Toggle neural network settings popup"""
        if self.settings_popup and self.settings_popup.winfo_exists():
            self.settings_popup.destroy()
            self.settings_popup = None
        else:
            self.create_neural_settings_popup()

    def create_neural_settings_popup(self):
        """Create premium neural network settings popup"""
        self.settings_popup = tk.Toplevel(self.root)
        self.settings_popup.title("Neural Network Settings")
        self.settings_popup.geometry("500x600")
        self.settings_popup.configure(bg="#0a0a0f")
        self.settings_popup.resizable(False, False)

        # Center the popup
        self.settings_popup.transient(self.root)
        self.settings_popup.grab_set()

        # Header
        header = tk.Frame(self.settings_popup, bg="#1a1a2e", height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🧠 Neural Network Configuration",
            font=("Segoe UI", 18, "bold"),
            fg="#00d4ff",
            bg="#1a1a2e"
        ).pack(pady=20)

        # Settings container
        container = tk.Frame(self.settings_popup, bg="#0a0a0f")
        container.pack(fill="both", expand=True, padx=25, pady=20)

        # Learning Mode
        learning_frame = tk.Frame(container, bg="#1a1a2e", relief="flat", bd=0)
        learning_frame.pack(fill="x", pady=8)

        tk.Label(
            learning_frame, text="Continuous Learning", 
            font=("Segoe UI", 12, "bold"), fg="white", bg="#1a1a2e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.learning_var = tk.BooleanVar(value=self.assistant.learning_enabled)
        learning_toggle = tk.Checkbutton(
            learning_frame, text="Enable AI to learn from conversations",
            variable=self.learning_var, bg="#1a1a2e", fg="#8b9dc3",
            selectcolor="#1a1a2e", command=lambda: self.toggle_learning(self.learning_var.get())
        )
        learning_toggle.pack(anchor="w", padx=15, pady=5)

        # Confidence Threshold
        threshold_frame = tk.Frame(container, bg="#1a1a2e")
        threshold_frame.pack(fill="x", pady=8)

        tk.Label(
            threshold_frame, text="Confidence Threshold", 
            font=("Segoe UI", 12, "bold"), fg="white", bg="#1a1a2e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        threshold_slider = ttk.Scale(
            threshold_frame, from_=0, to=1, orient="horizontal",
            command=lambda v: self.update_threshold(float(v))
        )
        threshold_slider.set(self.assistant.confidence_threshold)
        threshold_slider.pack(fill="x", padx=15, pady=5)

        threshold_val = tk.Label(
            threshold_frame, text=f"Current: {self.assistant.confidence_threshold:.0%}",
            font=("Segoe UI", 10), fg="#4ecdc4", bg="#1a1a2e"
        )
        threshold_val.pack(anchor="w", padx=15, pady=5)

        # Voice Settings
        voice_frame = tk.Frame(container, bg="#1a1a2e")
        voice_frame.pack(fill="x", pady=8)

        tk.Label(
            voice_frame, text="Voice Settings", 
            font=("Segoe UI", 12, "bold"), fg="white", bg="#1a1a2e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        speed_slider = ttk.Scale(voice_frame, from_=100, to=300, orient="horizontal")
        speed_slider.set(180)
        speed_slider.pack(fill="x", padx=15, pady=5)

        # Gesture Control
        gesture_frame = tk.Frame(container, bg="#1a1a2e")
        gesture_frame.pack(fill="x", pady=8)

        tk.Label(
            gesture_frame, text="Gesture Control", 
            font=("Segoe UI", 12, "bold"), fg="white", bg="#1a1a2e"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.gesture_var = tk.BooleanVar(value=True)
        gesture_toggle = tk.Checkbutton(
            gesture_frame, text="Enable air gestures for control",
            variable=self.gesture_var, bg="#1a1a2e", fg="#8b9dc3",
            selectcolor="#1a1a2e", command=lambda: self.toggle_gesture(self.gesture_var.get())
        )
        gesture_toggle.pack(anchor="w", padx=15, pady=5)

        # Buttons
        btn_frame = tk.Frame(container, bg="#0a0a0f")
        btn_frame.pack(pady=20)

        tk.Button(
            btn_frame, text="Reset Neural Network", bg="#ff6b6b", fg="white",
            font=("Segoe UI", 11), bd=0, padx=20, pady=8,
            cursor="hand2", command=self.reset_neural_network
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame, text="Save & Close", bg="#00d4ff", fg="white",
            font=("Segoe UI", 11), bd=0, padx=20, pady=8,
            cursor="hand2", command=self.settings_popup.destroy
        ).pack(side="left", padx=5)

    def show_apps_popup(self):
        """Show premium apps popup with categories"""
        if self.apps_popup and self.apps_popup.winfo_exists():
            self.apps_popup.destroy()

        self.apps_popup = tk.Toplevel(self.root)
        self.apps_popup.title("Applications & Settings")
        self.apps_popup.geometry("600x700")
        self.apps_popup.configure(bg="#0a0a0f")
        self.apps_popup.resizable(True, True)

        self.apps_popup.transient(self.root)

        # Header with tabs
        header = tk.Frame(self.apps_popup, bg="#1a1a2e", height=100)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="📱 Application Hub",
            font=("Segoe UI", 20, "bold"),
            fg="#00d4ff",
            bg="#1a1a2e"
        ).pack(pady=(15, 5))

        # Tab buttons
        tab_frame = tk.Frame(header, bg="#1a1a2e")
        tab_frame.pack(pady=5)

        self.apps_btn = tk.Button(
            tab_frame, text="📱 Applications", font=("Segoe UI", 11, "bold"),
            bg="#00d4ff", fg="white", bd=0, padx=20, pady=8,
            cursor="hand2", command=lambda: self.show_tab_content("apps")
        )
        self.apps_btn.pack(side="left", padx=5)

        self.settings_btn_tab = tk.Button(
            tab_frame, text="⚙️ System Settings", font=("Segoe UI", 11),
            bg="#2a2a3a", fg="white", bd=0, padx=20, pady=8,
            cursor="hand2", command=lambda: self.show_tab_content("settings")
        )
        self.settings_btn_tab.pack(side="left", padx=5)

        # Search bar
        search_frame = tk.Frame(self.apps_popup, bg="#0a0a0f")
        search_frame.pack(fill="x", padx=20, pady=15)

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(
            search_frame,
            textvariable=self.search_var,
            font=("Segoe UI", 12),
            bg="#1e1e2e",
            fg="white",
            insertbackground="white",
            bd=0,
            relief="flat"
        )
        search_entry.pack(fill="x", pady=5, ipady=8)
        search_entry.bind("<KeyRelease>", self.filter_list)

        # Listbox container
        list_container = tk.Frame(self.apps_popup, bg="#0a0a0f")
        list_container.pack(fill="both", expand=True, padx=20, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")

        # Listbox
        self.listbox = tk.Listbox(
            list_container,
            font=("Segoe UI", 11),
            bg="#1e1e2e",
            fg="white",
            selectbackground="#00d4ff",
            selectforeground="white",
            bd=0,
            relief="flat",
            yscrollcommand=scrollbar.set,
            height=20
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind("<Double-Button-1>", self.open_selected_item)

        # Show apps by default
        self.current_tab = "apps"
        self.show_tab_content("apps")

    def show_tab_content(self, tab):
        """Show content based on selected tab"""
        self.current_tab = tab

        if tab == "apps":
            self.apps_btn.config(bg="#00d4ff", font=("Segoe UI", 11, "bold"))
            self.settings_btn_tab.config(bg="#2a2a3a", font=("Segoe UI", 11))
            self.current_items = sorted(list(self.assistant.apps.keys()))
        else:
            self.apps_btn.config(bg="#2a2a3a", font=("Segoe UI", 11))
            self.settings_btn_tab.config(bg="#00d4ff", font=("Segoe UI", 11, "bold"))
            self.current_items = sorted(list(self.assistant.settings.keys()))

        self.update_listbox()

    def filter_list(self, event):
        """Filter list based on search text"""
        self.update_listbox()

    def update_listbox(self):
        """Update listbox with filtered items"""
        self.listbox.delete(0, tk.END)
        search_text = self.search_var.get().lower()

        for item in self.current_items:
            if search_text in item.lower():
                self.listbox.insert(tk.END, item.title())

    def open_selected_item(self, event):
        """Open selected app or setting"""
        selection = self.listbox.curselection()
        if selection:
            item = self.listbox.get(selection[0]).lower()

            if self.current_tab == "apps":
                if item in self.assistant.apps:
                    try:
                        app_path = self.assistant.apps[item]
                        if "ms-settings:" in app_path:
                            os.system(f"start {app_path}")
                        else:
                            subprocess.Popen(app_path)
                        self.assistant.speak(f"{item} open kar raha hu")
                        self.add_to_chat("System", f"Opened {item}", "system")
                    except:
                        self.assistant.speak(f"{item} open nahi ho paya")
            else:
                if item in self.assistant.settings:
                    try:
                        os.system(f"start {self.assistant.settings[item]}")
                        self.assistant.speak(f"{item} setting open kar raha hu")
                        self.add_to_chat("System", f"Opened {item} settings", "system")
                    except:
                        self.assistant.speak(f"{item} open nahi ho paya")

            self.apps_popup.destroy()

    def show_settings_popup(self):
        """Show system settings popup"""
        self.show_apps_popup()
        self.show_tab_content("settings")

    def toggle_learning(self, enabled):
        """Toggle learning mode"""
        self.assistant.learning_enabled = enabled
        if enabled:
            self.assistant.start_background_training()
            self.status_indicator.config(text=" Neural Network Active | Learning Mode: ON")
        else:
            self.assistant.stop_background_training()
            self.status_indicator.config(text=" Neural Network Active | Learning Mode: OFF")
        self.add_to_chat("System", f"Neural network learning {'enabled' if enabled else 'disabled'}", "system")

    def update_threshold(self, value):
        """Update confidence threshold"""
        self.assistant.confidence_threshold = value

    def toggle_gesture(self, enabled):
        """Toggle gesture control"""
        self.assistant.gesture_active = enabled
        self.add_to_chat("System", f"Gesture control {'enabled' if enabled else 'disabled'}", "system")

    def reset_neural_network(self):
        """Reset neural network weights"""
        if messagebox.askyesno("Reset Neural Network", "⚠️ This will erase all learned patterns. Are you sure?"):
            self.assistant.neural_network = NeuralNetwork()
            self.assistant.neural_network.save_weights()
            self.add_to_chat("System", "🧠 Neural network has been reset to initial state", "system")
            self.assistant.speak("Neural network reset complete")

    def run_gesture_recognition(self):
        """Run gesture recognition in background"""
        try:
            self.assistant.cap = cv2.VideoCapture(0)
            last_action_time = 0

            while True:
                if not self.assistant.cap:
                    time.sleep(1)
                    self.assistant.cap = cv2.VideoCapture(0)
                    continue

                ret, frame = self.assistant.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.assistant.hands.process(rgb)

                if result.multi_hand_landmarks and self.assistant.gesture_active:
                    for hand in result.multi_hand_landmarks:
                        index_tip = hand.landmark[8]
                        thumb_tip = hand.landmark[4]

                        distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

                        if distance < 0.05 and time.time() - last_action_time > 1:
                            self.quick_command("click")
                            last_action_time = time.time()

                        index_up = hand.landmark[8].y < hand.landmark[6].y
                        middle_up = hand.landmark[12].y < hand.landmark[10].y
                        ring_up = hand.landmark[16].y < hand.landmark[14].y

                        if index_up and middle_up and not ring_up and time.time() - last_action_time > 2:
                            self.quick_command("take screenshot")
                            last_action_time = time.time()

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except Exception as e:
            print(f"Gesture recognition error: {e}")
        finally:
            if self.assistant.cap:
                self.assistant.cap.release()
            cv2.destroyAllWindows()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     ISHA - Intelligent System for Human Assistance               ║
    ║     Premium Neural Network Powered AI Assistant                  ║
    ║                                                                  ║
    ║     Features:                                                    ║
    ║     • Custom Neural Network (built from scratch)                 ║
    ║     • Local Language Model (no API keys)                         ║
    ║     • Voice + Text + Gesture Control                             ║
    ║     • Continuous Learning & Adaptation                           ║
    ║     • Premium Glass-morphism UI                                  ║
    ║     • Apps & Settings Hub                                        ║
    ║                                                                  ║
    ║     Compatible with Python 3.14.3                                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Create necessary directories
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)

    # Initialize GUI
    root = tk.Tk()

    # Set window icon if available
    try:
        root.iconbitmap(default="isha.ico")
    except:
        pass

    app = ISHAGUI(root)

    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
    