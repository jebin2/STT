import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger().setLevel(logging.ERROR)

import argparse
import os
import sys
import subprocess

STT_ENGINE = None
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '../hf_download')))

def server_mode(args):
	"""Run in server mode - read commands from stdin."""
	global STT_ENGINE
	
	while True:
		try:
			input_line = sys.stdin.readline().strip()
			if not input_line:
				break
			
			args.input = input_line

			result = initiate(args)
			
			if result:
				print(f"SUCCESS: {args.input}")
			else:
				print(f"ERROR: {args.input}")
			sys.stdout.flush()

		except Exception as e:
			print(f"Error in server mode: {e}")
			break

def check_for_dependency(model):
	"""
	Check and install dependencies for the given model if missing.
	Looks for <model>_requirements.txt file.
	"""
	requirements_file_name = f"{model}_requirements.txt"

	# Check if requirements file exists
	if not os.path.isfile(requirements_file_name):
		raise FileNotFoundError(f"Requirements file '{requirements_file_name}' not found for model '{model}'.")

	try:
		print(f"🔍 Checking dependencies for model: {model}...")

		if model == "parakeet":
			subprocess.check_call(
				[sys.executable, "-m", "pip", "install", "-r", "parakeet_pre-requirements.txt"]
			)

		# Use pip to install dependencies from the requirements file
		subprocess.check_call(
			[sys.executable, "-m", "pip", "install", "-r", requirements_file_name]
		)
		print(f"✅ Dependencies for '{model}' installed successfully.")
	except subprocess.CalledProcessError as e:
		print(f"❌ Failed to install dependencies for '{model}': {e}")
		sys.exit(1)

def current_env():
	"""Detect current virtual environment."""
	venv_path = os.environ.get("VIRTUAL_ENV")
	if venv_path:
		return os.path.basename(venv_path)
	raise ValueError("Please set env first")

def initiate(args):
	if not args.model:
		if current_env() == "openai_env":
			from .openai import OpenAISTTProcessor as STTEngine
		elif current_env() == "parakeet_env":
			from .parakeet import ParakeetSTTProcessor as STTEngine
		elif current_env() == "fasterwhispher_env":
			from .fasterwhispher import FasterWhispherSTTProcessor as STTEngine
	else:
		if args.model == "openai":
			from .openai import OpenAISTTProcessor as STTEngine
		elif args.model == "parakeet":
			from .parakeet import ParakeetSTTProcessor as STTEngine
		elif args.model == "fasterwhispher":
			from .fasterwhispher import FasterWhispherSTTProcessor as STTEngine

		check_for_dependency(args.model)

	global STT_ENGINE
	if not STT_ENGINE:
		STT_ENGINE = STTEngine()

	result = STT_ENGINE.transcribe(args)
	return result

def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(
		description="Speech-to-Text processor using Whisper"
	)
	parser.add_argument(
		"--server-mode", 
		action="store_true", 
		help="Run in server mode (read commands from stdin)"
	)
	parser.add_argument(
		"--input", 
		help="Input audio/video file path"
	)
	parser.add_argument(
		"--model",
		help="Input audio/video file path"
	)
	
	args = parser.parse_args()

	if args.server_mode:
		server_mode(args)
	else:
		if not args.input:
			print("Error: --input is required when not in server mode")
			return 1

		result = initiate(args)
		return 0 if result else 1

if __name__ == "__main__":
	sys.exit(main())