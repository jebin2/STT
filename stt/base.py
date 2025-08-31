from pathlib import Path
import os
import json
from . import common
import ffmpeg
import gc

from dotenv import load_dotenv
import os
if os.path.exists(".env"):
    print("Loaded load_dotenv")
    load_dotenv()

class BaseSTT:
	"""Base class for speech-to-text implementations"""
	
	def __init__(self, type):
		if os.getenv("USE_CPU_IF_POSSIBLE", None):
			self.device = "cpu"
		else:
			self.device = "cuda" if common.is_gpu_available() else "cpu"
		os.environ["TORCH_USE_CUDA_DSA"] = "1"
		os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
		os.environ["HF_HUB_TIMEOUT"] = "120"
		self.type = type
		self.input_file = None
		self.temp_dir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), "./temp_dir")))
		self.output_text_file = f"{self.temp_dir}/output_transcription.txt"
		self.output_json_file = f"{self.temp_dir}/output_transcription.json"
		self.model = None
		self.default_language = None

	def reset(self):
		import torch
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
		if os.path.exists(self.temp_dir):
			import shutil
			shutil.rmtree(self.temp_dir)
		os.makedirs(self.temp_dir, exist_ok=True)

	def validate_input_file(self, file_path):
		if not file_path or not os.path.exists(file_path):
			raise FileNotFoundError(f"File not found: {file_path}")

		return True
	
	def _is_video_file(self, file_path: str) -> bool:
		video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v')
		return file_path.lower().endswith(video_extensions)
	
	def _is_audio_file(self, file_path: str) -> bool:
		audio_extensions = ('.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg', '.wma')
		return file_path.lower().endswith(audio_extensions)
	
	def _extract_audio_from_video(self, video_path: str) -> str:
		temp_audio_path = f'{self.temp_dir}/input.wav'

		try:
			print(f"Trying to extract English audio from: {video_path}")

			# Attempt to extract only English audio stream
			ffmpeg.input(video_path).output(
				temp_audio_path,
				format='wav',
				acodec='pcm_s16le',
				ac=1,
				ar=16000,
				**{'map': '0:m:language:eng'}
			).overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True)
			
			if Path(temp_audio_path).exists() and Path(temp_audio_path).stat().st_size > 0:
				print(f"✅ English audio extracted to: {temp_audio_path}")
				return temp_audio_path
			else:
				raise Exception("English audio stream not found or empty.")

		except Exception as e:
			print(f"⚠️ English audio not found, falling back to default audio stream. Reason: {e}")

			# Fallback: extract default audio stream (usually stream index 0:1)
			ffmpeg.input(video_path).output(
				temp_audio_path,
				format='wav',
				acodec='pcm_s16le',
				ac=1,
				ar=16000,
				**{'map': '0:a:0'}  # fallback to first audio stream
			).overwrite_output().run()

			print(f"✅ Fallback audio extracted to: {temp_audio_path}")
			return temp_audio_path

	def save_transcription_results(self, result):
		"""Save transcription results to files.
		
		Args:
			result: Dictionary containing transcription results
			
		Returns:
			True if successful, False otherwise
		"""
		# Save text output
		with open(self.output_text_file, 'w', encoding='utf-8') as f:
			f.write(result["text"])
		print(f"Text transcription saved as {self.output_text_file}")
		
		# Save JSON output
		with open(self.output_json_file, 'w', encoding='utf-8') as f:
			json.dump(result, f, indent=4, ensure_ascii=False)
		print(f"JSON transcription saved as {self.output_json_file}")
		
		return True

	def transcribe(self, args):
		"""Main transcription method to be implemented by subclasses.
		
		Args:
			args: Arguments containing input file and options
			
		Returns:
			Dictionary with transcription results
		"""
		self.reset()
		input_file = args.get('input') if isinstance(args, dict) else getattr(args, 'input', None)

		self.validate_input_file(input_file)

		if self._is_video_file(input_file):
			print(f"Detected video file: {input_file}")
			audio_file_to_process = self._extract_audio_from_video(input_file)
			if not audio_file_to_process:
				return None, None

		elif self._is_audio_file(input_file):
			print(f"Detected audio file: {input_file}")
			audio_file_to_process = input_file

		else:
			raise ValueError("Error: Unsupported file format, Supported formats: .mp4, .avi, .mov, .mkv, .webm, .wav, .flac, .mp3, .m4a, .aac")

		result = self.generate_transcription(audio_file_to_process)
		
		if not result:
			print("Error: No transcription generated")
			return False

		success = self.save_transcription_results(result)
		
		return result if success else False

	def generate_transcription(self, input_file):
		"""Generate transcription - to be implemented by subclasses."""
		raise NotImplementedError("Subclasses must implement generate_transcription method")

	def cleanup(self):
		"""Clean up model resources."""
		if hasattr(self, 'model') and self.model is not None:
			print("Cleaning up model...")
			try:
				del self.model
				gc.collect()
				try:
					import torch
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
						torch.cuda.ipc_collect()
				except ImportError:
					pass
				print("Model memory cleaned.")
			except Exception as e:
				print(f"Error during model cleanup: {e}")

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.cleanup()

	def __del__(self):
		self.cleanup()