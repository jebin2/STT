# STT-Runner

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

A flexible, multi-engine command-line tool for high-quality Speech-to-Text (STT) transcription.

This tool provides a unified interface to transcribe audio and video files using different state-of-the-art STT engines, automatically handling file preparation, audio extraction, and output formatting.

### Key Features

- **Multiple Engine Support**: Easily switch between different transcription engines.
  - **OpenAI Whisper**: The official implementation.
  - **Faster-Whisper**: A faster, memory-efficient re-implementation.
  - **NVIDIA Parakeet**: High-performance model for timestamp accuracy.
- **Audio & Video Ready**: Directly processes video files (`.mp4`, `.mkv`, etc.) by extracting audio automatically.
- **Smart Chunking**: Handles long audio files efficiently using overlapping windows (especially for Parakeet).
- **Detailed Output**: Generates both a plain text file and a detailed JSON file with word-level timestamps.

## Installation

This project is packaged for installation via `pip`. You can install the base package and then add support for the specific engine(s) you need.

```bash
# Clone your repository (or use the direct git URL)
git clone https://github.com/your-repo/STT.git
cd STT
```

Install the desired engine support. You only need to install the ones you plan to use.

```bash
# For OpenAI Whisper support
pip install .[openai]

# For Faster-Whisper support
pip install .[fasterwhisper]

# For NVIDIA Parakeet support (requires NVIDIA GPU and CUDA)
pip install .[parakeet]

# To install support for all engines
pip install .[all]
```

## Usage

The tool provides a single command-line interface: `stt-transcribe`.

### Basic Transcription

The core command requires you to specify a model engine and an input file.

```bash
# Transcribe a file using OpenAI Whisper
stt-transcribe --model openai --input /path/to/my/video.mp4

# Transcribe using Faster-Whisper
stt-transcribe --model fasterwhisper --input /path/to/my/audio.wav
```

### Server Mode

For processing multiple files in a batch, you can use `--server-mode`. The tool will listen for file paths from standard input.

```bash
# Start the server with the Parakeet model
stt-transcribe --model parakeet --server-mode
```

Then, you can pipe file paths to it, one per line:
```
/path/to/file1.mp4
/path/to/another/audio.flac
/path/to/final_video.mkv
```
The server will process each file sequentially. Press `Ctrl+C` to exit.

## Supported Engines

| Engine Name | `--model` argument | Notes |
| :--- | :--- |:---|
| OpenAI Whisper | `openai` | The official Whisper `large-v3-turbo` model. |
| Faster-Whisper | `fasterwhisper` | A faster, optimized implementation of Whisper (`base` model). GPU recommended. |
| NVIDIA Parakeet | `parakeet` | High-quality model with excellent timestamp accuracy. **NVIDIA GPU required**. |


## Output Files

For each run, the tool generates the following files in a `temp_dir` folder in your current directory:

- `output_transcription.txt`: The full transcribed text.
- `output_transcription.json`: A detailed JSON object containing the full text, language, duration, and segment/word-level timestamps.