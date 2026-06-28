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
        self.chunk_duration = 5
        self.stride_duration = 2
        self.last_segment_end = 0
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

    def process(self):
        if self.is_finalized:
            return []

        self._drain_incoming()

        chunk_samples = self.chunk_duration * self.sample_rate
        stride_samples = self.stride_duration * self.sample_rate

        if len(self.buffer) - self.processed_until < chunk_samples:
            return []

        chunk = self.buffer[self.processed_until : self.processed_until + chunk_samples]
        time_offset = (self.buffer_start + self.processed_until) / self.sample_rate
        self.processed_until += chunk_samples - stride_samples
        self._trim_buffer()

        try:
            segments, _ = self.model.transcribe(
                chunk, beam_size=1, vad_filter=True, language="en"
            )
            results = []
            for seg in segments:
                start = seg.start + time_offset
                end = seg.end + time_offset
                text = seg.text.strip()
                if not text:
                    continue
                # Consecutive chunks overlap by stride_duration, so the overlap
                # region is transcribed twice. Skip segments that fall within
                # time we've already emitted (small tolerance for boundary jitter).
                if end <= self.last_segment_end + 0.2:
                    continue
                self.last_segment_end = max(self.last_segment_end, end)
                results.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                })
            return results
        except Exception as e:
            logger.error(f"[StreamingSTT] process error: {e}")
            return []

    def flush(self):
        if self.is_finalized:
            return []
        self.is_finalized = True

        self._drain_incoming()

        remaining = self.buffer[self.processed_until:]
        if len(remaining) < self.sample_rate * 0.5:
            return []

        try:
            segments, _ = self.model.transcribe(
                remaining, beam_size=1, vad_filter=True, language="en"
            )
            time_offset = (self.buffer_start + self.processed_until) / self.sample_rate
            results = []
            for seg in segments:
                start = seg.start + time_offset
                end = seg.end + time_offset
                text = seg.text.strip()
                if not text:
                    continue
                # The tail still overlaps the last emitted chunk by stride_duration;
                # drop anything already covered.
                if end <= self.last_segment_end + 0.2:
                    continue
                self.last_segment_end = max(self.last_segment_end, end)
                results.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                    })
            return results
        except Exception as e:
            logger.error(f"[StreamingSTT] flush error: {e}")
            return []

    def cleanup(self):
        if self.model is not None:
            self.model = None
            _release_model(self.model_name, self.device)
        self.buffer = np.array([], dtype=np.float32)
        import gc
        gc.collect()
