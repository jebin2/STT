# setup.py
import os
from setuptools import setup, find_packages

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
	long_description = f.read()

# Define base requirements needed for the core functionality
BASE_DEPS = [
	'torch',
	'ffmpeg-python',
	'numpy'
]

# Define optional dependencies for each STT engine.
# This allows users to install only what they need, e.g., pip install .[openai]
extras_require = {
	'openai': [
		'openai-whisper>=20231117',
		'python-dotenv'
	],
	'fasterwhisper': [
		'faster-whisper',
		'python-dotenv'
	],
	'parakeet': [
		'nemo_toolkit[asr]',
		'cuda-python>=12.3',
		'librosa',
		'soundfile',
		'typing_extensions',
		'python-dotenv'
	],
}

# Create an 'all' extra that includes all engine dependencies
all_deps = []
for deps in extras_require.values():
	all_deps.extend(deps)
extras_require['all'] = list(set(all_deps))

setup(
	name="stt-runner",
	version="1.0.0",
	author="Jebin Einstein",
	author_email="jebin@gmail.com",
	description="A flexible, multi-engine Speech-to-Text runner",
	long_description=long_description,
	long_description_content_type='text/markdown',
	url="https://github.com/jebin2/STT",

	# This finds all packages in your project
	packages=find_packages(),

	# Core dependencies installed by default
	install_requires=BASE_DEPS,

	# Optional dependencies
	extras_require=extras_require,

	# Creates the 'stt-transcribe' command-line tool
	entry_points={
		'console_scripts': [
			'stt-transcribe=stt.runner:main',
		],
	},

	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Topic :: Multimedia :: Sound/Audio :: Speech",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
	],
	python_requires='>=3.10',
)