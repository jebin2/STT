import numpy as np
import queue
import sys
import time
from .base import BaseSTT


class LiveSTTProcessor(BaseSTT):
    def __init__(self, model_name="base", device=None):
        super().__init__("live")
        self.device = device or self.device or "cpu"
        self.model_name = model_name
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.is_running = False
        self._load_model()

    def _load_model(self):
        from faster_whisper import WhisperModel
        compute = "int8" if self.device == "cpu" else "float16"
        self.model = WhisperModel(self.model_name, device=self.device, compute_type=compute)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def start(self):
        import sounddevice as sd

        self.is_running = True
        buffer = np.array([], dtype=np.float32)
        chunk_duration = 5
        stride_duration = 2
        chunk_samples = chunk_duration * self.sample_rate
        stride_samples = stride_duration * self.sample_rate
        processed_until = 0
        # Absolute sample index of buffer[0]; grows as we trim leading samples so
        # timestamps stay anchored to real audio time after a trim.
        buffer_start = 0

        stream = sd.InputStream(
            callback=self._audio_callback,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=4096,
        )
        stream.start()

        print(f"Live STT started (model: {self.model_name}, device: {self.device})")
        print("Speak into your microphone. Press Ctrl+C to stop.\n")

        try:
            while self.is_running:
                while not self.audio_queue.empty():
                    data = self.audio_queue.get()
                    buffer = np.append(buffer, data.flatten())

                if len(buffer) - processed_until >= chunk_samples:
                    chunk = buffer[processed_until : processed_until + chunk_samples]
                    # Offset is the chunk's real start time; capture it before
                    # advancing processed_until past this chunk.
                    time_offset = (buffer_start + processed_until) / self.sample_rate
                    processed_until += chunk_samples - stride_samples

                    if processed_until > self.sample_rate * 120:
                        trim = processed_until - self.sample_rate * 30
                        buffer = buffer[trim:]
                        processed_until -= trim
                        buffer_start += trim

                    try:
                        segments, _ = self.model.transcribe(chunk, beam_size=1, vad_filter=True)
                        for seg in segments:
                            start = seg.start + time_offset
                            end = seg.end + time_offset
                            text = seg.text.strip()
                            if text:
                                print(f"[{start:.1f}s -> {end:.1f}s] {text}")
                                sys.stdout.flush()
                    except Exception as e:
                        print(f"[error] {e}", file=sys.stderr)

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()
            self.is_running = False
            print("\nLive STT stopped.")

    def stop(self):
        self.is_running = False

    def generate_transcription(self, input_file):
        raise NotImplementedError("LiveSTT does not support file transcription. Use start() for live mic input.")
