import torch
from base_stt import BaseSTT

class OpenAISTTProcessor(BaseSTT):
	"""Speech-to-text processor using OpenAI Whisper."""
	
	def __init__(self):
		super().__init__("openai")
		self.model_name = "large-v3-turbo"
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self._load_model()

	def _load_model(self):
		"""Load OpenAI Whisper model."""
		try:
			print(f"Initializing OpenAI Whisper...")
			import whisper
			print(f"Loading model: {self.model_name}")
			self.model = whisper.load_model(self.model_name, device=self.device)
			print("Model loaded successfully!")
			
		except Exception as e:
			raise RuntimeError(f"Failed to load model: {str(e)}")

	def generate_transcription(self, input_file):
		"""Generate transcription using OpenAI Whisper."""
		try:
			print(f"Transcribing: {input_file}")
			
			# Transcribe with OpenAI Whisper
			options = {
				# "word_timestamps":True,
				"verbose": True
			}
			
			result = self.model.transcribe(input_file, **options)
			
			transcription_result = {
				"text": result["text"],
				"language": result["language"],
				"model": f"{self.type}-{self.model_name}",
				"duration": result.get("duration", 0),
				"segments": result.get("segments", []),
				"engine": self.type
			}
			
			print(f"Transcription completed successfully!")
			return transcription_result
			
		except Exception as e:
			print(f"Transcription failed: {str(e)}")
			return None