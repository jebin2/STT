import numpy as np


class StreamingSTT:
    def __init__(self, model_name="base", device="cpu", sample_rate=16000):
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.buffer = np.array([], dtype=np.float32)
        self.processed_until = 0
        self.chunk_duration = 5
        self.stride_duration = 2
        self.last_segment_end = 0
        self.is_finalized = False

        compute = "int8" if device == "cpu" else "float16"
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device=device, compute_type=compute)

    def add_audio(self, audio_bytes: bytes):
        audio_float = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        self.buffer = np.append(self.buffer, audio_float)

    def _trim_buffer(self):
        max_buffered = self.sample_rate * 120
        if len(self.buffer) > max_buffered:
            trim_to = self.processed_until - self.sample_rate * 30
            if trim_to > 0:
                self.buffer = self.buffer[trim_to:]
                self.processed_until -= trim_to

    def process(self):
        if self.is_finalized:
            return []

        chunk_samples = self.chunk_duration * self.sample_rate
        stride_samples = self.stride_duration * self.sample_rate

        if len(self.buffer) - self.processed_until < chunk_samples:
            return []

        chunk = self.buffer[self.processed_until : self.processed_until + chunk_samples]
        time_offset = self.processed_until / self.sample_rate
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
                if text:
                    self.last_segment_end = end
                    results.append({
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                    })
            return results
        except Exception as e:
            print(f"[StreamingSTT] error: {e}")
            return []

    def flush(self):
        if self.is_finalized:
            return []
        self.is_finalized = True

        remaining = self.buffer[self.processed_until:]
        if len(remaining) < self.sample_rate * 0.5:
            return []

        try:
            segments, _ = self.model.transcribe(
                remaining, beam_size=1, vad_filter=True, language="en"
            )
            time_offset = self.processed_until / self.sample_rate
            results = []
            for seg in segments:
                start = seg.start + time_offset
                end = seg.end + time_offset
                text = seg.text.strip()
                if text:
                    results.append({
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                    })
            return results
        except Exception as e:
            print(f"[StreamingSTT] flush error: {e}")
            return []

    def cleanup(self):
        self.model = None
        self.buffer = np.array([], dtype=np.float32)
        import gc
        gc.collect()
