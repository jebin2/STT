import queue
import threading

import numpy as np

from custom_logger import logger_config as logger

# Models clients are allowed to request. Anything else is rejected before a load
# is ever attempted (an unknown name would otherwise trigger a download).
ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3"}

# Whisper weights are large, so identical (model, device) pairs are shared across
# connections instead of loaded once per connection (4 concurrent large-v3
# models would otherwise OOM). faster-whisper's WhisperModel is safe to use from
# multiple threads. Entries are ref-counted and freed when the last user leaves.
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _acquire_model(model_name, device):
    key = (model_name, device)
    with _MODEL_CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is None:
            from faster_whisper import WhisperModel
            compute = "int8" if device == "cpu" else "float16"
            model = WhisperModel(model_name, device=device, compute_type=compute)
            entry = {"model": model, "refs": 0}
            _MODEL_CACHE[key] = entry
        entry["refs"] += 1
        return entry["model"]


def _release_model(model_name, device):
    key = (model_name, device)
    with _MODEL_CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is None:
            return
        entry["refs"] -= 1
        if entry["refs"] <= 0:
            del _MODEL_CACHE[key]


class _HypothesisBuffer:
    """LocalAgreement-2 commit policy.

    Each window re-transcribes the unconfirmed audio. A word is only *committed*
    once two consecutive windows agree on it (longest common prefix); everything
    after the agreed prefix stays *tentative* and may be revised by the next
    window. This removes the duplicated/unstable output that naive overlapping
    re-transcription produces. (Macháček et al., whisper_streaming.)
    """

    def __init__(self):
        self.committed = []   # confirmed (start, end, word)
        self.buffer = []      # previous window's tentative tail
        self.new = []
        self.last_committed_time = 0.0

    def insert(self, words):
        # words: list of (start, end, text) in absolute seconds.
        self.new = [w for w in words if w[0] > self.last_committed_time - 0.1]
        if self.new and self.committed:
            # Drop a leading n-gram that repeats the tail we already committed
            # (whisper sometimes re-emits the previous words verbatim).
            if abs(self.new[0][0] - self.last_committed_time) < 1.0:
                cn, nn = len(self.committed), len(self.new)
                for i in range(1, min(cn, nn, 5) + 1):
                    tail = " ".join(self.committed[-j][2] for j in range(i, 0, -1))
                    head = " ".join(self.new[j][2] for j in range(i))
                    if tail == head:
                        del self.new[:i]
                        break

    def flush(self):
        """Commit the longest common prefix of this window and the last."""
        commit = []
        while self.new and self.buffer:
            if self.new[0][2] == self.buffer[0][2]:
                commit.append(self.new[0])
                self.last_committed_time = self.new[0][1]
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed.extend(commit)
        # Only the last few committed words are needed for n-gram dedup.
        if len(self.committed) > 100:
            self.committed = self.committed[-100:]
        return commit

    def complete(self):
        """Return remaining tentative words as final (no more audio coming)."""
        rest = self.buffer
        self.buffer = []
        return rest

    def tentative_text(self):
        return " ".join(w[2] for w in self.buffer)


class StreamingSTT:
    def __init__(self, model_name="base", device="cpu", sample_rate=16000):
        if model_name not in ALLOWED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.device = device
        self.buffer = np.array([], dtype=np.float32)
        self.processed_until = 0
        # Absolute sample index of buffer[0]. Grows as _trim_buffer() discards
        # leading samples, so timestamps stay anchored to real audio time
        # instead of drifting after a trim.
        self.buffer_start = 0
        self.min_chunk = 1.0  # seconds of new audio before a window is run
        self.hyp = _HypothesisBuffer()
        self.is_finalized = False
        # add_audio() runs on the event-loop thread while process()/flush() run
        # in an executor thread. Incoming audio is handed over through this
        # thread-safe queue so that only the executor thread ever mutates
        # self.buffer, avoiding a data race.
        self._incoming = queue.Queue()

        self.model = _acquire_model(model_name, device)

    def add_audio(self, audio_bytes: bytes):
        audio_float = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        self._incoming.put(audio_float)

    def _drain_incoming(self):
        chunks = []
        while True:
            try:
                chunks.append(self._incoming.get_nowait())
            except queue.Empty:
                break
        if chunks:
            self.buffer = np.append(self.buffer, np.concatenate(chunks))

    def _trim_buffer(self):
        max_buffered = self.sample_rate * 120
        if len(self.buffer) > max_buffered:
            trim_to = self.processed_until - self.sample_rate * 30
            if trim_to > 0:
                self.buffer = self.buffer[trim_to:]
                self.processed_until -= trim_to
                self.buffer_start += trim_to

    def _transcribe_words(self, audio, time_offset):
        """Transcribe audio, returning [(start, end, text), ...] in absolute time."""
        segments, _ = self.model.transcribe(
            audio, beam_size=1, vad_filter=True, language="en", word_timestamps=True
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append((w.start + time_offset, w.end + time_offset, text))
        return words

    @staticmethod
    def _as_chunk(words):
        """Join committed words into a single transcript chunk, or None."""
        if not words:
            return None
        return {
            "start": round(words[0][0], 2),
            "end": round(words[-1][1], 2),
            "text": " ".join(w[2] for w in words),
        }

    def process(self):
        if self.is_finalized:
            return None

        self._drain_incoming()

        unprocessed = self.buffer[self.processed_until:]
        if len(unprocessed) < self.min_chunk * self.sample_rate:
            return None

        time_offset = (self.buffer_start + self.processed_until) / self.sample_rate
        try:
            words = self._transcribe_words(unprocessed, time_offset)
        except Exception as e:
            logger.error(f"[StreamingSTT] process error: {e}")
            return None

        self.hyp.insert(words)
        committed = self.hyp.flush()

        # Advance past the committed audio; tentative words stay unprocessed so
        # the next window can re-evaluate (and possibly correct) them.
        if committed:
            target = int(committed[-1][1] * self.sample_rate) - self.buffer_start
            self.processed_until = min(max(self.processed_until, target), len(self.buffer))
            self._trim_buffer()

        return {
            "commit": self._as_chunk(committed),
            "tentative": self.hyp.tentative_text(),
        }

    def flush(self):
        if self.is_finalized:
            return None
        self.is_finalized = True

        self._drain_incoming()

        unprocessed = self.buffer[self.processed_until:]
        final = []
        if len(unprocessed) >= 0.3 * self.sample_rate:
            time_offset = (self.buffer_start + self.processed_until) / self.sample_rate
            try:
                words = self._transcribe_words(unprocessed, time_offset)
                self.hyp.insert(words)
                final = self.hyp.flush()
            except Exception as e:
                logger.error(f"[StreamingSTT] flush error: {e}")
        # No more audio is coming, so commit whatever tentative words remain.
        final = final + self.hyp.complete()
        return {"commit": self._as_chunk(final)}

    def cleanup(self):
        if self.model is not None:
            self.model = None
            _release_model(self.model_name, self.device)
        self.buffer = np.array([], dtype=np.float32)
        import gc
        gc.collect()
