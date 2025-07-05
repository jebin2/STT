import torch
from base_stt import BaseSTT

class FasterWhispherSTTProcessor(BaseSTT):
	"""Speech-to-text processor using OpenAI Whisper."""
	
	def __init__(self, model_name = "base", device=None):
		super().__init__("fasterwhispher")
		self.model_name = model_name
		self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
		self._load_model()

	def _load_model(self):
		"""Load OpenAI Whisper model."""
		try:
			print(f"Initializing Faster Whisper...")
			from faster_whisper import WhisperModel
			print(f"Loading model: {self.model_name}")
			self.model = WhisperModel(self.model_name, device=self.device)
			print("Model loaded successfully!")
			
		except Exception as e:
			raise RuntimeError(f"Failed to load model: {str(e)}")

	def generate_transcription(self, input_file):
		"""Generate transcription using OpenAI Whisper."""
		try:
			print(f"Transcribing: {input_file}")
			
			# Transcribe with OpenAI Whisper
			options = {
				"word_timestamps":True,
				"log_progress": True
			}
			
			segments, info = self.model.transcribe(input_file, **options)
			full_text = ""
			segment_array = []
			word_array = []

			for seg in segments:
				# Add to full text
				full_text += seg.text.strip() + " "

				# Add segment-level data
				segment_array.append({
					"start": seg.start,
					"end": seg.end,
					"text": seg.text.strip()
				})

				# Add word-level data
				for w in seg.words:
					word_array.append({
						"word": w.word,
						"start": w.start,
						"end": w.end,
						"probability": w.probability
					})

			# Final result in your desired format
			transcription_result = {
				"text": full_text.strip(),
				"language": info.language,
				"model": f"{self.type}-{self.model_name}",
				"duration": info.duration,
				"segments": {
					"segment": segment_array,
					"word": word_array
				},
				"engine": self.type
			}
			
			print(f"Transcription completed successfully!")
			return transcription_result
			
		except Exception as e:
			print(f"Transcription failed: {str(e)}")
			return None