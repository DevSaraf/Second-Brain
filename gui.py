#frontend 
import sys
import os
import json
import math
import time
import random
import platform
import threading
import subprocess
 
from PyQt6.QtCore import (
    Qt, QThread, QObject, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, pyqtProperty,
)
from PyQt6.QtGui import (
    QColor, QPainter, QFont, QBrush, QPen, QRadialGradient, QPainterPath, QFontMetrics,
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QListWidget, QListWidgetItem, QScrollArea, QLineEdit,
    QSizePolicy, QGraphicsDropShadowEffect,
)
 
# --------------------------------------------------------------------------- #
# Configuration (mirrors your backend script)
# --------------------------------------------------------------------------- #
RATE = 16000
WAKE_CHUNK = 1280
VOSK_CHUNK = 4096
WAKE_THRESHOLD = 0.92
COOLDOWN_SECONDS = 5
SILENCE_TIMEOUT = 3
MAX_HISTORY = 20
 
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
PIPER_VOICE_PATH = "voices/en_US-amy-medium.onnx"
 
# --------------------------------------------------------------------------- #
# Optional heavy dependencies — degrade gracefully if missing
# --------------------------------------------------------------------------- #
try:
    import requests
except Exception:
    requests = None
 
try:
    import numpy as np
except Exception:
    np = None
 
try:
    import pyaudio
    from vosk import Model as VoskModel, KaldiRecognizer
    from openwakeword.model import Model as WakeWordModel
    from piper import PiperVoice
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    AUDIO_STACK = True
except Exception:
    AUDIO_STACK = False
 
import wave  # stdlib, always available
 
 
# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
class C:
    BG = "#0d0f14"
    BG2 = "#12151c"
    CARD = "#171b24"
    CARD2 = "#1e2330"
    LINE = "#262c3a"
    TEXT = "#e8ebf2"
    MUTED = "#8b93a7"
    ACCENT = "#3b82f6"
    ACCENT_HI = "#5b9bff"
    USER = "#2b3650"
    BOT = "#1e2330"
 
# State -> (color, label)
STATES = {
    "idle":      ("#5b6172", "Idle"),
    "loading":   ("#f59e0b", "Loading models…"),
    "waiting":   ("#3b82f6", "Waiting for wake word"),
    "listening": ("#22c55e", "Listening…"),
    "thinking":  ("#f59e0b", "Thinking…"),
    "speaking":  ("#a855f7", "Speaking…"),
    "error":     ("#ef4444", "Error"),
}
 
 
# --------------------------------------------------------------------------- #
# Assistant core: history + LLM + emotion + TTS (no Qt; thread-safe)
# --------------------------------------------------------------------------- #
def _simple_sentiment(text):
    pos = {"good", "great", "love", "happy", "awesome", "thanks", "thank",
           "nice", "excellent", "amazing", "wonderful", "glad", "yay", "perfect"}
    neg = {"bad", "sad", "hate", "angry", "terrible", "awful", "tired",
           "annoyed", "upset", "depressed", "worried", "sorry", "cant", "wrong"}
    words = text.lower().split()
    s = sum(w in pos for w in words) - sum(w in neg for w in words)
    return max(-1.0, min(1.0, s / 3.0))
 
 
DEMO_REPLIES = [
    "I'm running in demo mode right now, but the interface is fully live. "
    "Hook up the audio stack and llama-server and I'll respond for real.",
    "Got it. This is a simulated reply so you can see the chat flow and the orb states.",
    "Sounds good. Once your local model is connected, this is exactly where my answer "
    "would appear.",
    "Happy to help. In live mode I'd keep replies short since they're read aloud.",
]
 
 
class AssistantCore:
    """Owns conversation history, LLM calls, emotion detection and TTS."""
 
    def __init__(self):
        self.history = []
        self._hist_lock = threading.Lock()
        self._speak_lock = threading.Lock()
 
        self.vosk_model = None
        self.wake_model = None
        self.voice = None
        self.analyzer = None
        self.mode = "demo"          # "live" or "demo"
        self.loaded = False
 
    # ----- model loading (called from a worker thread) ----------------------
    def load_models(self, log):
        if self.loaded:
            return self.mode
 
        if not AUDIO_STACK:
            log("Audio/AI stack not installed — running in DEMO mode.")
            self.mode = "demo"
            self.loaded = True
            return self.mode
 
        try:
            if not os.path.isdir(VOSK_MODEL_PATH):
                raise FileNotFoundError(f"Vosk model not found: {VOSK_MODEL_PATH}")
            if not os.path.isfile(PIPER_VOICE_PATH):
                raise FileNotFoundError(f"Piper voice not found: {PIPER_VOICE_PATH}")
 
            log("Loading Vosk model…")
            self.vosk_model = VoskModel(VOSK_MODEL_PATH)
            log("Loading wake-word model…")
            self.wake_model = WakeWordModel()
            log("Loading sentiment analyzer…")
            self.analyzer = SentimentIntensityAnalyzer()
            log("Loading Piper voice…")
            self.voice = PiperVoice.load(PIPER_VOICE_PATH)
            log("All models loaded — LIVE mode.")
            self.mode = "live"
        except Exception as e:
            log(f"Model load failed ({e}). Falling back to DEMO mode.")
            self.mode = "demo"
 
        self.loaded = True
        return self.mode
 
    # ----- emotion ----------------------------------------------------------
    def detect_emotion(self, text):
        if self.analyzer is not None:
            compound = self.analyzer.polarity_scores(text)["compound"]
        else:
            compound = _simple_sentiment(text)
        if compound >= 0.5:
            return "happy"
        if compound <= -0.5:
            return "sad"
        return "neutral"
 
    def clear_history(self):
        with self._hist_lock:
            self.history.clear()
 
    # ----- LLM --------------------------------------------------------------
    def ask(self, prompt):
        """Return (emotion, answer, ok)."""
        emotion = self.detect_emotion(prompt)
 
        with self._hist_lock:
            self.history.append({"role": "user", "content": prompt})
            if len(self.history) > MAX_HISTORY:
                self.history[:] = self.history[-MAX_HISTORY:]
            history_snapshot = list(self.history)
 
        if self.mode == "demo" or requests is None:
            time.sleep(0.9 + random.random())
            answer = random.choice(DEMO_REPLIES)
            self._remember(answer)
            return emotion, answer, True
 
        system_prompt = {
            "role": "system",
            "content": (
                "You are Jarvis, a voice assistant.\n"
                f"Current user emotion: {emotion}\n\n"
                "Rules:\n"
                "- Keep replies short: 2 to 4 sentences. They are read aloud.\n"
                "- If emotion is happy, respond warmly and enthusiastically.\n"
                "- If emotion is sad, respond empathetically and supportively.\n"
                "- If emotion is neutral, respond normally.\n"
                "Keep responses natural and conversational."
            ),
        }
        messages = [system_prompt] + history_snapshot
 
        try:
            resp = requests.post(
                LLM_URL,
                json={"messages": messages, "max_tokens": 200, "temperature": 0.7},
                timeout=120,
            )
            answer = resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return emotion, "I can't reach the language model. Is llama-server running on port 8080?", False
        except requests.exceptions.Timeout:
            return emotion, "The model took too long to respond.", False
        except Exception as e:
            return emotion, f"Something went wrong talking to the model: {e}", False
 
        self._remember(answer)
        return emotion, answer, True
 
    def _remember(self, answer):
        with self._hist_lock:
            self.history.append({"role": "assistant", "content": answer})
            if len(self.history) > MAX_HISTORY:
                self.history[:] = self.history[-MAX_HISTORY:]
 
    # ----- TTS --------------------------------------------------------------
    def speak(self, text):
        if self.voice is None or not text:
            return
        with self._speak_lock:
            try:
                with wave.open("reply.wav", "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file)
                self._play("reply.wav")
            except Exception as e:
                print(f"TTS error: {e}")
 
    @staticmethod
    def _play(path):
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME)
            elif system == "Darwin":
                subprocess.run(["afplay", path], check=False)
            else:
                subprocess.run(["aplay", "-q", path], check=False)
        except Exception as e:
            print(f"Playback error: {e}")
 
 
# --------------------------------------------------------------------------- #
# Voice worker: the wake -> listen -> ask -> speak loop, in its own thread
# --------------------------------------------------------------------------- #
class VoiceWorker(QThread):
    state_changed = pyqtSignal(str)
    status = pyqtSignal(str)
    user_said = pyqtSignal(str)
    assistant_said = pyqtSignal(str)
    partial = pyqtSignal(str)
    emotion = pyqtSignal(str)
    level = pyqtSignal(float)
    log = pyqtSignal(str)
    mode_resolved = pyqtSignal(str)
 
    def __init__(self, core: AssistantCore):
        super().__init__()
        self.core = core
        self._running = False
        self._pa = None
        self._stream = None
 
    def stop(self):
        self._running = False
 
    # ----- thread entry -----------------------------------------------------
    def run(self):
        self._running = True
        self.state_changed.emit("loading")
        mode = self.core.load_models(self.log.emit)
        self.mode_resolved.emit(mode)
 
        try:
            if mode == "live":
                self._run_live()
            else:
                self._run_demo()
        except Exception as e:
            self.log.emit(f"Worker crashed: {e}")
            self.state_changed.emit("error")
        finally:
            self._close_stream()
            self.state_changed.emit("idle")
 
    # ----- live mode --------------------------------------------------------
    def _open_stream(self, chunk):
        self._stream = self._pa.open(
            format=pyaudio.paInt16, channels=1, rate=RATE,
            input=True, frames_per_buffer=chunk,
        )
        self._stream.start_stream()
 
    def _close_stream(self):
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        except Exception:
            pass
 
    def _run_live(self):
        self._pa = pyaudio.PyAudio()
        self._open_stream(WAKE_CHUNK)
        last_activation = 0.0
 
        while self._running:
            self.state_changed.emit("waiting")
            self.status.emit("Say “Hey Jarvis”")
 
            detected = self._wait_for_wake_word(last_activation)
            if not self._running:
                break
            if not detected:
                continue
            last_activation = time.time()
 
            self.state_changed.emit("listening")
            transcript = self._voice_session()
            if not self._running:
                break
 
            prompt = " ".join(t["text"] for t in transcript)
            prompt = prompt.replace("stop listening", "").strip()
            if not prompt:
                self.log.emit("No speech captured.")
                continue
 
            self.user_said.emit(prompt)
            self.state_changed.emit("thinking")
            emotion, answer, ok = self.core.ask(prompt)
            self.emotion.emit(emotion)
            self.assistant_said.emit(answer)
 
            if ok and answer:
                self.state_changed.emit("speaking")
                self.core.speak(answer)
 
            self._sleep(COOLDOWN_SECONDS)
 
        if self._pa is not None:
            self._pa.terminate()
 
    def _wait_for_wake_word(self, last_activation):
        self.core.wake_model.reset()
        # warm-up
        for _ in range(int(RATE / WAKE_CHUNK * 1.5)):
            if not self._running:
                return False
            self._stream.read(WAKE_CHUNK, exception_on_overflow=False)
 
        while self._running:
            audio = self._stream.read(WAKE_CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(audio, dtype=np.int16)
            self.level.emit(self._rms(audio_np))
            prediction = self.core.wake_model.predict(audio_np)
            for wakeword, score in prediction.items():
                if score > WAKE_THRESHOLD:
                    if time.time() - last_activation < COOLDOWN_SECONDS:
                        continue
                    self.log.emit(f"Wake word detected: {wakeword} ({score:.2f})")
                    return True
        return False
 
    def _voice_session(self):
        self._close_stream()
        self._open_stream(VOSK_CHUNK)
 
        recognizer = KaldiRecognizer(self.core.vosk_model, RATE)
        transcript = []
        session_start = time.time()
        last_speech = time.time()
        last_partial = ""
        self.status.emit("Listening — say “stop listening” to end")
 
        while self._running:
            data = self._stream.read(VOSK_CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16)
            self.level.emit(self._rms(audio_np))
 
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "")
                if not text:
                    continue
                last_speech = time.time()
                transcript.append({"timestamp": time.time() - session_start, "text": text})
                self.partial.emit(text)
                if "stop listening" in text.lower():
                    self.log.emit("Stop command detected.")
                    for _ in range(20):
                        self._stream.read(VOSK_CHUNK, exception_on_overflow=False)
                    break
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                if partial and partial != last_partial:
                    last_speech = time.time()
                    last_partial = partial
                    self.partial.emit(partial)
 
            if time.time() - last_speech > SILENCE_TIMEOUT:
                self.log.emit(f"{SILENCE_TIMEOUT}s of silence — ending session.")
                break
 
        # restore wake stream
        self._close_stream()
        self._open_stream(WAKE_CHUNK)
        return transcript
 
    @staticmethod
    def _rms(audio_np):
        if np is None or audio_np.size == 0:
            return 0.0
        val = float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))
        return min(1.0, val / 4000.0)
 
    # ----- demo mode --------------------------------------------------------
    def _run_demo(self):
        samples = [
            "hey jarvis what's the weather like today",
            "can you set a reminder for my meeting at three",
            "tell me a fun fact about space",
            "what time is it right now",
        ]
        idx = 0
        while self._running:
            self.state_changed.emit("waiting")
            self.status.emit("Demo mode — auto-triggering wake word")
            if not self._sleep(2.5 + random.random() * 2):
                break
 
            self.log.emit("Wake word detected (demo).")
            self.state_changed.emit("listening")
            self.status.emit("Listening… (demo)")
 
            phrase = samples[idx % len(samples)]
            idx += 1
            built = ""
            for word in phrase.split():
                if not self._running:
                    return
                built = (built + " " + word).strip()
                self.partial.emit(built)
                self.level.emit(0.3 + random.random() * 0.6)
                time.sleep(0.18)
 
            self.user_said.emit(built)
            self.state_changed.emit("thinking")
            emotion, answer, _ = self.core.ask(built)
            self.emotion.emit(emotion)
            self.assistant_said.emit(answer)
 
            self.state_changed.emit("speaking")
            self._sleep(min(5, 1.5 + len(answer) / 30))
            self._sleep(1.5)
 
    def _sleep(self, seconds):
        """Sleep in small slices so stop() is responsive. Returns False if stopped."""
        end = time.time() + seconds
        while time.time() < end:
            if not self._running:
                return False
            time.sleep(0.05)
        return self._running
 
 
# --------------------------------------------------------------------------- #
# One-shot worker for typed messages (independent of the voice loop)
# --------------------------------------------------------------------------- #
class TextWorker(QThread):
    assistant_said = pyqtSignal(str)
    emotion = pyqtSignal(str)
    finished_ok = pyqtSignal()
 
    def __init__(self, core: AssistantCore, prompt: str, speak: bool):
        super().__init__()
        self.core = core
        self.prompt = prompt
        self.speak = speak
 
    def run(self):
        emotion, answer, ok = self.core.ask(self.prompt)
        self.emotion.emit(emotion)
        self.assistant_said.emit(answer)
        if self.speak and ok:
            self.core.speak(answer)
        self.finished_ok.emit()
 
 
# --------------------------------------------------------------------------- #
# Animated microphone orb
# --------------------------------------------------------------------------- #
class MicOrb(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(230, 230)
        self._phase = 0.0
        self._state = "idle"
        self._level = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
 
    def set_state(self, state):
        self._state = state if state in STATES else "idle"
 
    def set_level(self, level):
        self._level = max(self._level, max(0.0, min(1.0, level)))
 
    def _tick(self):
        self._phase += 0.05
        self._level *= 0.90
        self.update()
 
    def _color(self):
        return QColor(STATES.get(self._state, STATES["idle"])[0])
 
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        base = self._color()
 
        active = self._state in ("listening", "speaking", "thinking", "waiting")
        pulse = (math.sin(self._phase) + 1) / 2          # 0..1
        reactive = self._level if self._state == "listening" else pulse * 0.6
 
        # Outer pulsing rings
        max_r = min(w, h) / 2
        for i in range(3):
            t = (self._phase * 0.5 + i / 3.0) % 1.0
            r = max_r * (0.55 + t * 0.45)
            alpha = int(90 * (1 - t) * (1.0 if active else 0.3))
            ring = QColor(base)
            ring.setAlpha(alpha)
            pen = QPen(ring)
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
 
        # Glow halo
        halo_r = max_r * (0.62 + reactive * 0.22)
        grad = QRadialGradient(cx, cy, halo_r)
        glow = QColor(base)
        glow.setAlpha(120)
        grad.setColorAt(0.0, glow)
        edge = QColor(base)
        edge.setAlpha(0)
        grad.setColorAt(1.0, edge)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(int(cx - halo_r), int(cy - halo_r), int(halo_r * 2), int(halo_r * 2))
 
        # Core disc
        core_r = max_r * 0.42
        core_grad = QRadialGradient(cx, cy - core_r * 0.3, core_r * 1.4)
        light = base.lighter(135)
        core_grad.setColorAt(0.0, light)
        core_grad.setColorAt(1.0, base.darker(115))
        p.setBrush(QBrush(core_grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(int(cx - core_r), int(cy - core_r), int(core_r * 2), int(core_r * 2))
 
        # Mic glyph
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ffffff"))
        bw, bh = core_r * 0.34, core_r * 0.62
        p.drawRoundedRect(int(cx - bw / 2), int(cy - core_r * 0.55),
                          int(bw), int(bh), int(bw / 2), int(bw / 2))
        arc_pen = QPen(QColor("#ffffff"))
        arc_pen.setWidth(max(3, int(core_r * 0.07)))
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        ar = core_r * 0.5
        p.drawArc(int(cx - ar), int(cy - ar * 0.6),
                  int(ar * 2), int(ar * 1.4), 200 * 16, 140 * 16)
        p.drawLine(int(cx), int(cy + core_r * 0.42), int(cx), int(cy + core_r * 0.62))
        p.end()
 
 
# --------------------------------------------------------------------------- #
# Chat bubbles
# --------------------------------------------------------------------------- #
class Bubble(QFrame):
    def __init__(self, text, sender):
        super().__init__()
        self.setObjectName("Bubble")
        bg = {"user": C.USER, "assistant": C.BOT, "system": "#20242f"}.get(sender, C.BOT)
        fg = C.TEXT if sender != "system" else C.MUTED
        self.setStyleSheet(
            f"#Bubble {{ background:{bg}; border-radius:16px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
 
        if sender != "system":
            who = QLabel("You" if sender == "user" else "Jarvis")
            who.setStyleSheet(
                f"color:{C.ACCENT_HI if sender=='user' else '#a855f7'};"
                "font-size:11px;font-weight:600;"
            )
            lay.addWidget(who)
 
        msg = QLabel(text)
        msg.setWordWrap(True)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg.setStyleSheet(f"color:{fg}; font-size:14px; background:transparent;")
        lay.addWidget(msg)
        self.setMaximumWidth(560)
 
 
class TypingBubble(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Bubble")
        self.setStyleSheet(f"#Bubble {{ background:{C.BOT}; border-radius:16px; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)
        self.dots = []
        for _ in range(3):
            d = QLabel("●")
            d.setStyleSheet("color:#a855f7; font-size:12px;")
            lay.addWidget(d)
            self.dots.append(d)
        self._i = 0
        self._t = QTimer(self)
        self._t.timeout.connect(self._tick)
        self._t.start(280)
 
    def _tick(self):
        for j, d in enumerate(self.dots):
            d.setStyleSheet(
                f"color:{'#d6b4ff' if j == self._i else '#5a4a70'}; font-size:12px;"
            )
        self._i = (self._i + 1) % 3
 
 
class ChatView(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{background:transparent;width:8px;margin:4px;}"
            "QScrollBar::handle:vertical{background:#333a4a;border-radius:4px;min-height:30px;}"
            "QScrollBar::add-line,QScrollBar::sub-line{height:0;}"
        )
        self._content = QWidget()
        self._content.setStyleSheet("background:transparent;")
        self._lay = QVBoxLayout(self._content)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self._lay.setSpacing(10)
        self._lay.addStretch()
        self.setWidget(self._content)
        self._typing = None
 
    def _add_row(self, widget, sender):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if sender == "user":
            row.addStretch()
            row.addWidget(widget)
        elif sender == "system":
            row.addStretch()
            row.addWidget(widget)
            row.addStretch()
        else:
            row.addWidget(widget)
            row.addStretch()
        wrap = QWidget()
        wrap.setLayout(row)
        self._lay.insertWidget(self._lay.count() - 1, wrap)
        self._scroll_down()
        return wrap
 
    def add_message(self, text, sender):
        self._add_row(Bubble(text, sender), sender)
 
    def show_typing(self):
        if self._typing is None:
            self._typing = self._add_row(TypingBubble(), "assistant")
 
    def hide_typing(self):
        if self._typing is not None:
            self._typing.setParent(None)
            self._typing.deleteLater()
            self._typing = None
 
    def clear(self):
        self.hide_typing()
        while self._lay.count() > 1:
            item = self._lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
 
    def _scroll_down(self):
        QTimer.singleShot(20, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()))
 
 
# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class JarvisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.core = AssistantCore()
        self.worker = None
        self.text_worker = None
        self._text_busy = False
 
        self.setWindowTitle("Jarvis")
        self.resize(1200, 820)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(self._qss())
 
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(16)
 
        main.addWidget(self._build_sidebar())
        main.addWidget(self._build_content(), 1)
 
        self._set_state("idle")
 
    # ----- styling ----------------------------------------------------------
    def _qss(self):
        return f"""
        #Root {{ background: {C.BG}; }}
        QWidget {{ color: {C.TEXT}; font-family: "Segoe UI", "Inter", sans-serif; }}
        QFrame#Card {{ background: {C.CARD}; border-radius: 20px; }}
        QFrame#Card2 {{ background: {C.CARD2}; border-radius: 16px; }}
        QListWidget {{
            background: transparent; border: none; outline: 0;
        }}
        QListWidget::item {{
            padding: 12px 14px; border-radius: 10px; margin: 3px 0; color: {C.MUTED};
        }}
        QListWidget::item:hover {{ background: {C.CARD2}; color: {C.TEXT}; }}
        QListWidget::item:selected {{ background: {C.ACCENT}; color: white; }}
        QPushButton#Primary {{
            background: {C.ACCENT}; border: none; border-radius: 12px;
            padding: 12px 18px; font-size: 14px; font-weight: 600; color: white;
        }}
        QPushButton#Primary:hover {{ background: {C.ACCENT_HI}; }}
        QPushButton#Primary:disabled {{ background: #2a3142; color: #6a7184; }}
        QPushButton#Ghost {{
            background: {C.CARD2}; border: 1px solid {C.LINE}; border-radius: 12px;
            padding: 12px 18px; font-size: 14px; font-weight: 600; color: {C.TEXT};
        }}
        QPushButton#Ghost:hover {{ background: #262c3a; }}
        QPushButton#Ghost:disabled {{ color: #5a6072; }}
        QLineEdit {{
            background: {C.CARD2}; border: 1px solid {C.LINE}; border-radius: 12px;
            padding: 12px 14px; font-size: 14px; color: {C.TEXT};
        }}
        QLineEdit:focus {{ border: 1px solid {C.ACCENT}; }}
        """
 
    # ----- sidebar ----------------------------------------------------------
    def _build_sidebar(self):
        bar = QFrame()
        bar.setObjectName("Card")
        bar.setFixedWidth(248)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(18, 22, 18, 18)
        lay.setSpacing(16)
 
        brand = QHBoxLayout()
        dot = QLabel("◈")
        dot.setStyleSheet(f"color:{C.ACCENT}; font-size:22px;")
        name = QLabel("JARVIS")
        name.setStyleSheet("font-size:22px; font-weight:800; letter-spacing:2px;")
        brand.addWidget(dot)
        brand.addWidget(name)
        brand.addStretch()
        lay.addLayout(brand)
 
        self.nav = QListWidget()
        for label in ["  Assistant", "  History", "  Settings"]:
            QListWidgetItem(label, self.nav)
        self.nav.setCurrentRow(0)
        lay.addWidget(self.nav)
 
        lay.addStretch()
 
        # Status card
        card = QFrame()
        card.setObjectName("Card2")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.setSpacing(8)
        cap = QLabel("BACKEND")
        cap.setStyleSheet(f"color:{C.MUTED}; font-size:10px; font-weight:700; letter-spacing:1px;")
        self.mode_label = QLabel("● Not started")
        self.mode_label.setStyleSheet(f"color:{C.MUTED}; font-size:13px; font-weight:600;")
        cl.addWidget(cap)
        cl.addWidget(self.mode_label)
        lay.addWidget(card)
 
        clear_btn = QPushButton("New Chat")
        clear_btn.setObjectName("Ghost")
        clear_btn.clicked.connect(self._new_chat)
        lay.addWidget(clear_btn)
        return bar
 
    # ----- main content -----------------------------------------------------
    def _build_content(self):
        card = QFrame()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(26, 22, 26, 22)
        lay.setSpacing(16)
 
        # Header
        header = QHBoxLayout()
        title = QLabel("Voice Assistant")
        title.setStyleSheet("font-size:22px; font-weight:700;")
        self.pill = QLabel("● Idle")
        self.pill.setStyleSheet(self._pill_style("idle"))
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.pill)
        lay.addLayout(header)
 
        # Orb + state
        orb_box = QVBoxLayout()
        orb_box.setSpacing(6)
        self.orb = MicOrb()
        orb_box.addSpacing(6)
        orb_box.addWidget(self.orb, alignment=Qt.AlignmentFlag.AlignCenter)
        self.state_label = QLabel("Press “Start Listening” to begin")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet(f"font-size:16px; color:{C.MUTED};")
        self.emotion_label = QLabel("")
        self.emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.emotion_label.setStyleSheet(f"font-size:12px; color:{C.MUTED};")
        orb_box.addWidget(self.state_label)
        orb_box.addWidget(self.emotion_label)
        lay.addLayout(orb_box)
 
        # Chat
        self.chat = ChatView()
        self.chat.add_message("Ready. Connect the audio stack and llama-server for live mode, "
                              "or just press Start to see the demo.", "system")
        lay.addWidget(self.chat, 1)
 
        # Text input row
        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a message to Jarvis…")
        self.input.returnPressed.connect(self._send_text)
        send = QPushButton("Send")
        send.setObjectName("Primary")
        send.clicked.connect(self._send_text)
        self.send_btn = send
        input_row.addWidget(self.input, 1)
        input_row.addWidget(send)
        lay.addLayout(input_row)
 
        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.start_btn = QPushButton("Start Listening")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("Ghost")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.start_btn, 1)
        controls.addWidget(self.stop_btn, 1)
        lay.addLayout(controls)
 
        return card
 
    # ----- pill / state helpers --------------------------------------------
    def _pill_style(self, state):
        color = STATES.get(state, STATES["idle"])[0]
        return (f"background:{QColor(color).darker(260).name()};"
                f"color:{color}; padding:7px 16px; border-radius:13px;"
                "font-size:13px; font-weight:600;")
 
    def _set_state(self, state):
        color, label = STATES.get(state, STATES["idle"])
        self.orb.set_state(state)
        self.pill.setText(f"● {label}")
        self.pill.setStyleSheet(self._pill_style(state))
        if state == "thinking":
            self.chat.show_typing()
        else:
            self.chat.hide_typing()
        self.state_label.setText(label)
 
    # ----- worker wiring ----------------------------------------------------
    def _start(self):
        if self.worker and self.worker.isRunning():
            return
        self.worker = VoiceWorker(self.core)
        self.worker.state_changed.connect(self._set_state)
        self.worker.status.connect(lambda s: self.state_label.setText(s))
        self.worker.user_said.connect(lambda t: self.chat.add_message(t, "user"))
        self.worker.assistant_said.connect(self._on_assistant)
        self.worker.partial.connect(self._on_partial)
        self.worker.emotion.connect(self._on_emotion)
        self.worker.level.connect(self.orb.set_level)
        self.worker.log.connect(lambda m: print("[worker]", m))
        self.worker.mode_resolved.connect(self._on_mode)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
 
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
 
    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.stop_btn.setEnabled(False)
        self.state_label.setText("Stopping…")
 
    def _on_worker_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_state("idle")
        self.state_label.setText("Stopped. Press “Start Listening” to begin")
 
    def _on_mode(self, mode):
        if mode == "live":
            self.mode_label.setText("● Live")
            self.mode_label.setStyleSheet("color:#22c55e; font-size:13px; font-weight:600;")
        else:
            self.mode_label.setText("● Demo mode")
            self.mode_label.setStyleSheet("color:#f59e0b; font-size:13px; font-weight:600;")
 
    def _on_partial(self, text):
        self.state_label.setText(f"“{text}”")
 
    def _on_assistant(self, text):
        self.chat.hide_typing()
        self.chat.add_message(text, "assistant")
 
    def _on_emotion(self, emotion):
        icon = {"happy": "🙂", "sad": "🙁", "neutral": "😐"}.get(emotion, "")
        self.emotion_label.setText(f"detected emotion: {emotion} {icon}")
 
    # ----- typed messages ---------------------------------------------------
    def _send_text(self):
        text = self.input.text().strip()
        if not text or self._text_busy:
            return
        self.input.clear()
        self.chat.add_message(text, "user")
        self.chat.show_typing()
        self._text_busy = True
        self.send_btn.setEnabled(False)
 
        speak = bool(self.worker and self.worker.isRunning())
        self.text_worker = TextWorker(self.core, text, speak)
        self.text_worker.assistant_said.connect(self._on_assistant)
        self.text_worker.emotion.connect(self._on_emotion)
        self.text_worker.finished_ok.connect(self._on_text_done)
        self.text_worker.start()
 
    def _on_text_done(self):
        self._text_busy = False
        self.send_btn.setEnabled(True)
 
    # ----- misc -------------------------------------------------------------
    def _new_chat(self):
        self.core.clear_history()
        self.chat.clear()
        self.chat.add_message("New conversation started.", "system")
        self.emotion_label.setText("")
 
    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()
 
 
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = JarvisUI()
    window.show()
    sys.exit(app.exec())
 
 
if __name__ == "__main__":
    main()