import re
from typing import Optional, Dict, Any, List

import gc
import os
import librosa
import soundfile as sf
import ffmpeg
from base_stt import BaseSTT

class ParakeetSTTProcessor(BaseSTT):
	"""Enhanced Speech-to-Text converter with smart overlap handling."""
	
	def __init__(self):
		super().__init__("parakeet")
		os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
		self.model_name = "nvidia/parakeet-tdt-0.6b-v2"
		self.chunk_duration = 300
		self.chunk_overlap = 5
		self.sample_rate = 16000
		self._load_model()

	def _load_model(self):
		try:
			print("Initializing Nemo ASR...")
			import nemo.collections.asr as nemo_asr
			from nemo.utils.nemo_logging import Logger as nemo_log
			import logging
			nemo_log().set_verbosity(logging.ERROR)
			print(f"Loading model: {self.model_name}")
			self.model = nemo_asr.models.ASRModel.from_pretrained(
				model_name=self.model_name
			)
			print("Model loaded successfully!")
		except Exception as e:
			raise RuntimeError(f"Failed to load model: {str(e)}")

	def get_media_metadata(self, file_path):
		try:
			probe = ffmpeg.probe(file_path, v='error', select_streams='v:0', show_entries='format=duration,streams')
			duration_in_sec_float = float(probe['format']['duration'])
			duration_in_sec_int = int(duration_in_sec_float)
			size = int(os.path.getsize(file_path) // (1024 * 1024))
			fps = None
			for stream in probe['streams']:
				if stream['codec_type'] == 'video':
					fps = eval(stream['r_frame_rate'])
			return duration_in_sec_int, duration_in_sec_float, size, fps
		except Exception as e:
			print(f"Error retrieving media metadata: {e}")
			return None, None, None, None
		finally:
			gc.collect()

	def _get_audio_duration(self, audio_file: str) -> float:
		try:
			duration, _, _, _ = self.get_media_metadata(audio_file)
			return duration
		except Exception as e:
			print(f"Error getting audio duration: {e}")
			return 0
	
	def _split_audio_file(self, audio_file: str) -> List[str]:
		try:
			audio, sr = librosa.load(audio_file, sr=self.sample_rate)
			total_duration = len(audio) / sr
			
			print(f"Audio duration: {total_duration:.2f} seconds")
			print(f"Splitting into {self.chunk_duration}s chunks...")
			
			chunk_files = []
			chunk_samples = int(self.chunk_duration * sr)
			overlap_samples = int(self.chunk_overlap * sr)

			chunk_count = 0
			start_sample = 0
			
			while start_sample < len(audio):
				end_sample = min(start_sample + chunk_samples, len(audio))
				chunk_audio = audio[start_sample:end_sample]
				
				chunk_file = os.path.join(self.temp_dir, f"chunk_{chunk_count:04d}.wav")
				sf.write(chunk_file, chunk_audio, sr)
				chunk_files.append(chunk_file)
				
				print(f"Created chunk {chunk_count + 1}: {start_sample/sr:.2f}s - {end_sample/sr:.2f}s")
				
				start_sample = end_sample - overlap_samples
				chunk_count += 1
				
				if end_sample >= len(audio):
					break
			
			print(f"Created {len(chunk_files)} chunks")
			return chunk_files
		except Exception as e:
			print(f"Error splitting audio file: {e}")
			return []
	
	def _transcribe_single_chunk(self, audio_file: str) -> Optional[Dict[str, Any]]:
		try:
			outputs = self.model.transcribe(
				[audio_file],
				batch_size=1,
				timestamps=True
			)
			
			if outputs and len(outputs) > 0:
				output = outputs[0]
				
				timestamps = {}
				if hasattr(output, 'timestamp'):
					timestamps = {
						'word': output.timestamp.get('word', []),
						'segment': output.timestamp.get('segment', []),
						'char': output.timestamp.get('char', [])
					}
				
				return {
					'text': output.text,
					'timestamps': timestamps
				}
			return None
		except Exception as e:
			print(f"Error transcribing chunk: {e}")
			return None

	def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Merge chunk results by finding and removing overlapping words using timestamp matching."""
		
		if not chunk_results:
			return {'text': '', 'timestamps': {'word': [], 'segment': []}}
		
		if len(chunk_results) == 1:
			return chunk_results[0] if chunk_results[0] else {'text': '', 'timestamps': {'word': [], 'segment': []}}
		
		all_word_timestamps = []
		all_segment_timestamps = []
		
		# Process each chunk
		for i, result in enumerate(chunk_results):
			if not result or 'timestamps' not in result:
				continue
			
			timestamps = result.get('timestamps', {})
			word_timestamps = timestamps.get('word', [])
			segment_timestamps = timestamps.get('segment', [])
			
			if not word_timestamps:
				continue
			
			# Calculate time offset for this chunk (original audio time)
			time_offset = i * (self.chunk_duration - self.chunk_overlap)
			
			# Adjust current chunk timestamps to absolute time
			adjusted_curr_words = []
			for word in word_timestamps:
				adjusted_word = word.copy()
				adjusted_word['start'] = word.get('start', 0) + time_offset
				adjusted_word['end'] = word.get('end', 0) + time_offset
				adjusted_curr_words.append(adjusted_word)
			
			adjusted_curr_segments = []
			for segment in segment_timestamps:
				adjusted_segment = segment.copy()
				adjusted_segment['start'] = segment.get('start', 0) + time_offset
				adjusted_segment['end'] = segment.get('end', 0) + time_offset
				adjusted_curr_segments.append(adjusted_segment)
			
			# For first chunk, add everything
			if i == 0:
				all_word_timestamps.extend(adjusted_curr_words)
				all_segment_timestamps.extend(adjusted_curr_segments)
			
			else:
				# For subsequent chunks, find and remove timestamp overlaps
				remove_word_count = self._find_timestamp_overlap(all_word_timestamps, adjusted_curr_words, i)
				
				print(f"Chunk {i}: Skipping {remove_word_count} overlapping words based on timestamps from prev word")

				if remove_word_count > 0:
					# Skip the overlapping words
					all_word_timestamps = all_word_timestamps[:-remove_word_count]

				# Add remaining timestamps
				all_word_timestamps.extend(adjusted_curr_words)
		
		# Sort all timestamps by start time to ensure proper order
		all_word_timestamps.sort(key=lambda x: x.get('start', 0))
		all_segment_timestamps.sort(key=lambda x: x.get('start', 0))
		
		# Reconstruct text from word timestamps
		final_text = ' '.join([word.get('word', '') for word in all_word_timestamps])
		final_text = re.sub(r'\s+', ' ', final_text).strip()
		
		return {
			'text': final_text,
			'timestamps': {
				'word': all_word_timestamps,
				'segment': all_segment_timestamps
			}
		}

	def _find_timestamp_overlap(self, prev_words: List[Dict], curr_words: List[Dict], index) -> int:
		"""
		Find overlapping words using timestamp matching instead of text matching.
		
		Args:
			prev_words: List of word dictionaries from all previous chunks (with absolute timestamps)
			curr_words: List of word dictionaries from current chunk (with absolute timestamps)
			
		Returns:
			Number of words to skip from the beginning of curr_words
		"""
		if not prev_words or not curr_words:
			return 0
		
		# Get the timestamp range we expect overlap to occur in
		# Overlap should happen in the last chunk_overlap seconds of previous audio
		if prev_words:
			overlap_start_time = (self.chunk_duration * index) - (self.chunk_overlap)
		else:
			return 0
		
		# Find words in previous chunks that fall in the overlap period
		remove_word_count = 0
		for i, word in enumerate(prev_words):
			word_start = word.get('start', 0)
			word_end = word.get('end', 0)
			
			# Include words that overlap with the expected overlap period
			if word_start > overlap_start_time and word_end > overlap_start_time:
				remove_word_count += 1

		return remove_word_count
	
	def generate_transcription(self, input_file):
		"""Generate transcription using Parakeet."""
		try:
			print(f"Transcribing: {input_file}")
			duration = self._get_audio_duration(input_file)
			print(f"Processing audio duration: {duration:.2f} seconds")
			
			if duration > self.chunk_duration:
				print(f"Audio exceeds {self.chunk_duration}s, using enhanced chunking with overlap handling...")
				chunk_files = self._split_audio_file(input_file)
				
				if not chunk_files:
					return None, None
				
				chunk_results = []
				for i, chunk_file in enumerate(chunk_files):
					print(f"Processing chunk {i + 1}/{len(chunk_files)}")
					result = self._transcribe_single_chunk(chunk_file)
					chunk_results.append(result if result else {'text': '', 'timestamps': {}})
				
				final_result = self._merge_chunk_results(chunk_results)
				
			else:
				print("Processing as single file...")
				final_result = self._transcribe_single_chunk(input_file)
				
				if not final_result:
					return None, None

			transcription_result = {
				"text": final_result['text'],
				"language": "",
				"model": f"{self.type}-{self.model_name}",
				"duration": duration,
				"segments": final_result['timestamps'],
				"engine": self.type
			}
			
			print(f"Transcription completed successfully!")
			
			return transcription_result
			
		except Exception as e:
			print(f"Transcription failed: {str(e)}")
			return None