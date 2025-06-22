import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger().setLevel(logging.ERROR)

import argparse
import os
import sys

STT_ENGINE = None

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

def current_env():
	"""Detect current virtual environment."""
	venv_path = os.environ.get("VIRTUAL_ENV")
	if venv_path:
		return os.path.basename(venv_path)
	raise ValueError("Please set env first")

def initiate(args):
	if current_env() == "openai_env":
		from openai_stt import OpenAISTTProcessor as STTEngine
	elif current_env() == "parakeet_env":
		from parakeet_stt import ParakeetSTTProcessor as STTEngine

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