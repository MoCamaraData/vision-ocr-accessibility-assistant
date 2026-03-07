"""
Text-to-speech with two swappable backends:
  - "local"  → pyttsx3  (offline, for local testing and NAO)
  - "cloud"  → gTTS     (online, returns MP3 bytes for cloud deployment)

Usage:
    # Local
    spk = Speaker(backend="local", rate=150)
    spk.speak(gated_results)   # speaks all non-gated text aloud
    spk.repeat_last()          # repeat on spacebar press

    # Cloud
    spk = Speaker(backend="cloud", lang="en")
    mp3_bytes = spk.speak(gated_results)   # returns MP3 bytes to send to client
"""

from typing import List, Dict, Optional
import re

class Speaker:
    def __init__(self, backend: str = "local", rate: int = 150,
                 volume: float = 1.0, lang: str = "en"):
        """
        Args:
            backend: "local" (pyttsx3) or "cloud" (gTTS)
            rate:    words per minute — local only
            volume:  0.0–1.0 — local only
            lang:    language code — cloud only
        """
        self.backend = backend
        self.lang    = lang
        self._last_results: List[Dict] = []

        if backend == "local":
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   rate)
            self._engine.setProperty("volume", volume)
        elif backend == "cloud":
            # gTTS imported lazily — only needed on cloud
            from gtts import gTTS  # noqa: F401 (import check)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'local' or 'cloud'.")

    

    def speak(self, gated_results):
        self._last_results = gated_results
        spoken = [r for r in gated_results if not r["gated"]]

        if not spoken:
            print("TTS: nothing to speak (all gated)")
            return None

        # Strip results that are punctuation/symbols only — gTTS crashes on these
        spoken = [r for r in spoken if re.search(r'[a-zA-Z0-9]', r["text"])]

        if not spoken:
            print("TTS: nothing to speak (punctuation only)")
            return None

        text = " ... ".join(r["text"] for r in spoken)
        print(f"TTS speaking: {[r['text'] for r in spoken]}")

        if self.backend == "local":
            self._engine.say(text)
            self._engine.runAndWait()
            return None

        elif self.backend == "cloud":
            from gtts import gTTS
            import io
            tts = gTTS(text=text, lang=self.lang)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            return buf.getvalue()
        
    def repeat_last(self) -> Optional[bytes]:
        """Repeat the last spoken output. Triggered by spacebar in Phase 6."""
        if not self._last_results:
            print("TTS: nothing to repeat")
            return None
        return self.speak(self._last_results)
