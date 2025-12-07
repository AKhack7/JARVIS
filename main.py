import os
import sys
import time
import threading
import webbrowser
import logging
import queue
import socket
import datetime
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Optional / external libraries
try:
    import pyttsx3
except Exception as e:
    print("Missing pyttsx3. Install with: pip install pyttsx3")
    raise

try:
    import speech_recognition as sr
except Exception as e:
    print("Missing SpeechRecognition. Install with: pip install SpeechRecognition")
    raise

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except Exception:
    print("Missing NLTK. Install with: pip install nltk")
    raise

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    print("Missing scikit-learn. Install with: pip install scikit-learn")
    raise

# Optional: keyboard listener (may require root on Linux)
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except Exception:
    KEYBOARD_AVAILABLE = False

# Basic logging
logging.basicConfig(level=logging.INFO, filename="isha_assistant.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Download necessary NLTK data (safe to call repeatedly)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Default dataset path (you can change it at runtime)
DEFAULT_QNA_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "JARVIS", "Data", "brain_data", "qna_dat.txt")

class QnAModel:
    """Loads QnA dataset once and prepares TF-IDF model for simple matching."""
    def __init__(self, path=None):
        self.path = path or DEFAULT_QNA_PATH
        self.dataset = []
        self.vectorizer = None
        self.X = None
        self.lock = threading.Lock()
        self._load_and_train()

    def _load_and_train(self):
        try:
            if not os.path.isfile(self.path):
                logging.warning("QnA dataset not found at %s. Continuing without QnA.", self.path)
                return
            with open(self.path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            pairs = []
            for line in lines:
                if ':' in line:
                    q, a = line.split(':', 1)
                    pairs.append({'question': q.strip(), 'answer': a.strip()})
            if not pairs:
                logging.warning("QnA file found but no valid 'question:answer' lines detected.")
                return
            self.dataset = pairs
            corpus = [self._preprocess_text(qa['question']) for qa in self.dataset]
            self.vectorizer = TfidfVectorizer()
            self.X = self.vectorizer.fit_transform(corpus)
            logging.info("QnA model trained on %d pairs.", len(self.dataset))
        except Exception as e:
            logging.exception("Failed to load/train QnA model: %s", e)

    def _preprocess_text(self, text):
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        tokens = word_tokenize(text.lower())
        tokens = [ps.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
        return " ".join(tokens)

    def get_answer(self, question):
        with self.lock:
            if not self.dataset or self.vectorizer is None or self.X is None:
                return None
            proc = self._preprocess_text(question)
            qv = self.vectorizer.transform([proc])
            sims = cosine_similarity(qv, self.X)
            best = sims.argmax()
            score = sims[0, best]
            if score < 0.15:
                return None
            return self.dataset[best]['answer']

class IshaAssistant:
    def __init__(self, host='localhost', port=8000):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.last_internet_check = 0
        self.internet_status = False
        self.internet_check_interval = 10
        self.input_queue = queue.Queue()
        self.pending = None
        self.qna = QnAModel()
        self.host = host
        self.port = port

        # Safe console show/hide helpers (Windows-only)
        self._is_windows = sys.platform.startswith("win")

        # Try initialize microphone (graceful fallback)
        try:
            # Don't throw if no PyAudio; we will prompt later
            mic_names = sr.Microphone.list_microphone_names()
            if mic_names:
                try:
                    self.microphone = sr.Microphone()
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    logging.info("Default microphone initialized.")
                except Exception as e:
                    logging.warning("Could not initialize default microphone: %s", e)
                    self.microphone = None
            else:
                self.microphone = None
        except Exception as e:
            logging.warning("Microphone list error (likely PyAudio missing): %s", e)
            self.microphone = None

        # Choose female-ish voice if available
        self.set_female_voice()

        # Start optional hotkey listener
        if KEYBOARD_AVAILABLE:
            threading.Thread(target=self._listen_ctrl_m_hotkey, daemon=True).start()

        # Start web server & wish
        self.wish_me()
        self._create_dashboard_file()
        self._start_server_thread()

    # -------------------- TTS --------------------
    def set_female_voice(self):
        try:
            voices = self.engine.getProperty('voices')
            chosen = None
            for v in voices:
                name = getattr(v, 'name', '') or ''
                if 'zira' in name.lower() or 'female' in name.lower():
                    chosen = v
                    break
            if not chosen and voices:
                chosen = voices[0]
            if chosen:
                self.engine.setProperty('voice', chosen.id)
                logging.info("TTS voice set: %s", getattr(chosen, 'name', 'unknown'))
            else:
                logging.warning("No TTS voice available.")
        except Exception as e:
            logging.exception("Failed to set voice: %s", e)

    def speak(self, text):
        """Non-blocking TTS (spawns a daemon thread)."""
        def _run(t):
            try:
                # small stop/reinit guard
                try:
                    self.engine.stop()
                except Exception:
                    pass
                self.engine.say(t)
                self.engine.runAndWait()
                logging.info("Spoke: %s", t)
            except Exception as e:
                logging.exception("TTS failed: %s", e)
        threading.Thread(target=_run, args=(text,), daemon=True).start()
        time.sleep(0.25)

    # -------------------- Utilities --------------------
    def check_internet(self):
        now = time.time()
        if now - self.last_internet_check < self.internet_check_interval:
            return self.internet_status
        self.last_internet_check = now
        for host, port in [("8.8.8.8", 53), ("1.1.1.1", 53)]:
            try:
                socket.create_connection((host, port), timeout=2)
                self.internet_status = True
                return True
            except Exception:
                continue
        self.internet_status = False
        return False

    def wish_me(self):
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            g = "Good morning"
        elif 12 <= hour < 17:
            g = "Good afternoon"
        elif 17 <= hour < 21:
            g = "Good evening"
        else:
            g = "Hello"
        self.speak(g + ". I am Isha — your assistant.")
        time.sleep(0.8)

    # -------------------- Microphone / Voice --------------------
    def select_microphone(self):
        names = []
        try:
            names = sr.Microphone.list_microphone_names()
        except Exception:
            pass
        if not names:
            self.speak("No microphones detected. Please install PyAudio or check device permissions.")
            return None
        # If running in terminal, ask user to choose
        print("Available microphones:")
        for i, n in enumerate(names):
            print(f"{i}: {n}")
        try:
            idx = int(input("Enter index of microphone to use: ").strip())
            if 0 <= idx < len(names):
                self.microphone = sr.Microphone(device_index=idx)
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                self.speak(f"Selected microphone: {names[idx]}")
                return self.microphone
        except Exception as e:
            logging.warning("Microphone selection failed: %s", e)
            self.speak("Invalid selection.")
        return None

    def toggle_voice(self):
        if self.microphone is None:
            self.select_microphone()
            if self.microphone is None:
                return "no-mic"
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.speak("Microphone is on.")
            threading.Thread(target=self._listen_loop, daemon=True).start()
        else:
            self.speak("Microphone is off.")
        return "ok"

    def _listen_loop(self):
        while self.is_listening:
            cmd = self.listen_once()
            if cmd:
                self.process_command(cmd)
            time.sleep(0.5)

    def listen_once(self):
        if self.microphone is None:
            return None
        try:
            with self.microphone as src:
                self.recognizer.adjust_for_ambient_noise(src, duration=0.7)
                audio = self.recognizer.listen(src, timeout=8, phrase_time_limit=8)
            try:
                q = self.recognizer.recognize_google(audio).lower()
                logging.info("Heard: %s", q)
                return q
            except sr.UnknownValueError:
                logging.info("Could not understand audio")
                return None
            except sr.RequestError as e:
                logging.error("Speech service error: %s", e)
                return None
        except Exception as e:
            logging.exception("Listening failed: %s", e)
            return None

    # -------------------- Command processing --------------------
    def process_command(self, command):
        if not command:
            resp = "No command received."
            self.speak(resp)
            return resp

        cmd = command.lower().strip()
        logging.info("Processing command: %s", cmd)

        # simple open commands
        if cmd.startswith("open "):
            target = cmd[5:].strip()
            # try open as URL, else try known apps
            if target.startswith("http"):
                try:
                    webbrowser.open(target)
                    resp = f"Opening {target}"
                    self.speak(resp)
                    return resp
                except Exception as e:
                    resp = f"Failed to open URL: {e}"
                    self.speak(resp)
                    return resp
            # fallback: try os.startfile on windows or xdg-open on linux
            try:
                if sys.platform.startswith("win"):
                    os.startfile(target)
                else:
                    # try open with xdg-open where applicable
                    subprocess = __import__('subprocess')
                    subprocess.Popen(["xdg-open", target])
                resp = f"Attempting to open {target}"
                self.speak(resp)
                return resp
            except Exception as e:
                resp = f"Could not open {target}: {e}"
                self.speak(resp)
                return resp

        # time
        if "time" in cmd and len(cmd.split()) <= 3:
            t = datetime.datetime.now().strftime("%I:%M %p")
            resp = f"The time is {t}"
            self.speak(resp)
            return resp

        # date
        if "date" in cmd and len(cmd.split()) <= 3:
            d = datetime.datetime.now().strftime("%B %d, %Y")
            resp = f"Today's date is {d}"
            self.speak(resp)
            return resp

        # who are you
        if "who are you" in cmd or "your name" in cmd:
            resp = "मैं इशा हूँ, तुम्हारा पर्सनल असिस्टेंट!"
            self.speak(resp)
            return resp

        # QnA model
        ans = self.qna.get_answer(cmd)
        if ans:
            self.speak(ans)
            return ans

        # fallback echo
        resp = "I didn't find a specific answer. You said: " + command
        self.speak(resp)
        return resp

    # -------------------- Dashboard (static minimal) --------------------
    def _create_dashboard_file(self):
        html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Isha Assistant</title>
<style>
body{{font-family: Arial, Helvetica, sans-serif; background:#071027;color:#eaf6ff; padding:20px}}
.container{{max-width:900px;margin:0 auto;background: rgba(255,255,255,0.02); padding:18px;border-radius:12px}}
input[type=text]{{width:80%;padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,0.06)}}
button{{padding:10px 12px;border-radius:8px;border:none;cursor:pointer;background:#07a; color:white}}
.app-list{{margin-top:12px}}
.item{{padding:6px 0;cursor:pointer}}
.small{{font-size:13px;color:#cbefff;opacity:0.9}}
</style>
</head>
<body>
<div class="container">
  <h2>Isha Assistant</h2>
  <div class="small">Type a command and press Enter. Use "open example.com" or "time" or "who are you".</div>
  <div style="margin-top:12px">
    <input id="cmd" type="text" placeholder="Type command..." />
    <button id="send">Send</button>
    <button id="mic">Toggle Mic (Alt+M)</button>
  </div>
  <div id="output" style="margin-top:12px;padding:10px;background:rgba(255,255,255,0.02);border-radius:8px;min-height:80px"></div>

  <div class="app-list">
    <h4>Quick open</h4>
    <div class="item" onclick="send('open https://www.google.com')">• Open Google</div>
    <div class="item" onclick="send('open https://www.youtube.com')">• Open YouTube</div>
    <div class="item" onclick="send('time')">• Time</div>
    <div class="item" onclick="send('date')">• Date</div>
    <div class="item" onclick="send('who are you')">• Who are you</div>
  </div>
</div>

<script>
const out = document.getElementById('output');
function append(msg){ const d=document.createElement('div'); d.textContent=msg; out.prepend(d); }
function send(cmd){
  fetch('/command?cmd='+encodeURIComponent(cmd)).then(r=>r.text()).then(t=>append(t)).catch(e=>append('Error: '+e));
}
document.getElementById('send').onclick=function(){ const v=document.getElementById('cmd').value.trim(); if(!v) return; send(v); document.getElementById('cmd').value='';}
document.getElementById('cmd').addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ document.getElementById('send').click(); }});
document.getElementById('mic').onclick=function(){ fetch('/voice').then(r=>r.text()).then(t=>append(t));}
document.addEventListener('keydown', (e)=>{ if(e.altKey && e.key.toLowerCase()==='m'){ document.getElementById('mic').click(); }});
</script>
</body>
</html>
"""
        try:
            fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(html)
            logging.info("Dashboard written to %s", fn)
        except Exception as e:
            logging.exception("Failed to write dashboard.html: %s", e)

    def _start_server_thread(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(inner_self):
                parsed = urlparse(inner_self.path)
                if parsed.path == '/':
                    # serve static file
                    try:
                        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html"), 'rb') as fh:
                            content = fh.read()
                        inner_self.send_response(200)
                        inner_self.send_header('Content-type', 'text/html; charset=utf-8')
                        inner_self.end_headers()
                        inner_self.wfile.write(content)
                    except Exception as e:
                        inner_self.send_response(500)
                        inner_self.end_headers()
                        inner_self.wfile.write(b"Error serving dashboard.")
                elif parsed.path == '/command':
                    qs = parse_qs(parsed.query)
                    cmd = qs.get('cmd', [None])[0]
                    if not cmd:
                        inner_self.send_response(400)
                        inner_self.end_headers()
                        inner_self.wfile.write(b"Missing cmd param.")
                        return
                    # When assistant is pending input, queue it; else process immediately
                    if self.pending:
                        self.input_queue.put(cmd)
                        inner_self.send_response(200)
                        inner_self.end_headers()
                        inner_self.wfile.write(b"Queued input")
                    else:
                        try:
                            resp = self.process_command(cmd)
                            inner_self.send_response(200)
                            inner_self.send_header('Content-type', 'text/plain; charset=utf-8')
                            inner_self.end_headers()
                            inner_self.wfile.write(resp.encode('utf-8'))
                        except Exception as e:
                            inner_self.send_response(500)
                            inner_self.end_headers()
                            inner_self.wfile.write(b"Processing error.")
                elif parsed.path == '/voice':
                    res = self.toggle_voice()
                    inner_self.send_response(200)
                    inner_self.end_headers()
                    inner_self.wfile.write(res.encode() if isinstance(res, str) else b"ok")
                else:
                    inner_self.send_response(404)
                    inner_self.end_headers()

            def log_message(self, format, *args):
                # suppress default logging to stdout
                return

        def serve():
            try:
                server = HTTPServer((self.host, self.port), Handler)
                logging.info("Starting server on %s:%s", self.host, self.port)
                # open in browser
                url = f"http://{self.host}:{self.port}/"
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
                server.serve_forever()
            except Exception as e:
                logging.exception("Failed to start server: %s", e)
                print("Failed to start server:", e)

        t = threading.Thread(target=serve, daemon=True)
        t.start()

    # Optional hotkey monitor
    def _listen_ctrl_m_hotkey(self):
        try:
            while True:
                if keyboard.is_pressed('alt+m'):
                    self.toggle_voice()
                    time.sleep(0.6)
                time.sleep(0.1)
        except Exception as e:
            logging.exception("Hotkey thread error: %s", e)

if __name__ == "__main__":
    try:
        assistant = IshaAssistant()
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        logging.exception("App crashed: %s", e)
        print("Application failed to start:", e)
