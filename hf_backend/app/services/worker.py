import asyncio
import os
import json
import shlex
import re
from app.core.config import settings
from custom_logger import logger_config as logger
from app.db import crud

worker_task = None
worker_running = False

def is_worker_running():
    return worker_running

async def start_worker():
    global worker_task, worker_running
    
    logger.info(f"start_worker called: worker_running={worker_running}")
    
    if not worker_running:
        worker_running = True
        worker_task = asyncio.create_task(worker_loop())
        logger.info("Worker task started")
    else:
        logger.info("Worker already running")

async def worker_loop():
    global worker_running
    logger.info("STT Worker started. Monitoring for new audio files...")
    
    while worker_running:
        logger.debug("Worker loop iteration, checking for files...")
        await crud.cleanup_old_entries()
        
        try:
            row = await crud.get_next_not_started()
            
            if row:
                task_id = row['id']
                filepath = row['filepath']
                filename = row['filename']
                
                logger.info(f"\n{'='*60}\nProcessing: {filename}\nID: {task_id}\n{'='*60}")
                
                await crud.update_status(task_id, 'processing')
                
                try:
                    await crud.update_progress(task_id, 5, "Starting STT...")
                    
                    command = f"cd {settings.CWD} && {settings.PYTHON_PATH} --input {shlex.quote(os.path.abspath(filepath))} --model {settings.STT_MODEL_NAME}"
                    
                    logger.debug(f"Executing command: {command}")
                    
                    process = await asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=settings.CWD,
                        env={
                            **os.environ,
                            'PYTHONUNBUFFERED': '1',
                            'CUDA_LAUNCH_BLOCKING': '1',
                            'USE_CPU_IF_POSSIBLE': 'true'
                        }
                    )
                    
                    current_chunk = 1
                    total_chunks = 1
                    
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                            
                        line_str = line.decode('utf-8', errors='replace').strip()
                        if line_str:
                            logger.info(f"[STT] {line_str}")
                            
                            # Track chunk progress
                            chunk_match = re.search(r'Processing chunk (\d+)/(\d+)', line_str)
                            if chunk_match:
                                try:
                                    current_chunk = int(chunk_match.group(1))
                                    total_chunks = int(chunk_match.group(2))
                                except: pass
                            
                            # Generic percentage matcher
                            percent_match = re.search(r'(\d+)%', line_str)
                            if percent_match:
                                try:
                                    percent = int(percent_match.group(1))
                                    if 'audio' in line_str.lower() or 'extract' in line_str.lower():
                                        await crud.update_progress(task_id, percent // 2, "Extracting audio...")
                                    elif 'transcrib' in line_str.lower() or 'model' in line_str.lower():
                                        # Calculate overall transcription progress based on chunks
                                        chunk_base = ((current_chunk - 1) / total_chunks) * 100
                                        chunk_progress = (percent / total_chunks)
                                        overall_transcription_progress = chunk_base + chunk_progress
                                        
                                        # Remap so 50-100% of the overall bar is transcription
                                        overall_progress = int(50 + (overall_transcription_progress / 2))
                                        await crud.update_progress(task_id, overall_progress, f"Transcribing... (Chunk {current_chunk}/{total_chunks})")
                                    else:
                                        await crud.update_progress(task_id, percent, "Processing...")
                                except: pass
                                
                            # Stage matchers
                            if 'initializing nemo asr' in line_str.lower():
                                await crud.update_progress(task_id, 10, "Initializing engine...")
                            elif 'extracting audio' in line_str.lower():
                                await crud.update_progress(task_id, 15, "Extracting audio...")
                            elif 'model loaded' in line_str.lower():
                                await crud.update_progress(task_id, 25, "Model loaded...")
                            elif 'processing audio duration' in line_str.lower():
                                await crud.update_progress(task_id, 35, "Analyzing audio...")
                            elif 'transcription started' in line_str.lower() and total_chunks == 1:
                                await crud.update_progress(task_id, 50, "Transcribing started...")
                            elif 'transcription completed successfully' in line_str.lower():
                                await crud.update_progress(task_id, 90, "Transcription finished.")
                            elif 'json transcription saved' in line_str.lower():
                                await crud.update_progress(task_id, 95, "Saving data...")
                    
                    await process.wait()
                    if process.returncode != 0:
                        raise Exception(f"STT process failed with return code {process.returncode}")
                    
                    await crud.update_progress(task_id, 98, "Reading results...")
                    
                    output_path = os.path.join(settings.CWD, settings.TEMP_DIR, 'output_transcription.json')
                    with open(output_path, 'r') as file:
                        result = json.loads(file.read().strip())
                    
                    # Extract result text (caption)
                    result_data = result.get('text', '') or result.get('transcription', '') or str(result)
                    
                    logger.success(f"Successfully processed: {filename}")
                    logger.info(f"Text preview: {result_data[:100]}...")
                    
                    await crud.update_status(task_id, 'completed', result=json.dumps(result))
                    
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug(f"Deleted audio file: {filepath}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {str(e)}")
                    await crud.update_status(task_id, 'failed', error=str(e))
                    
            else:
                await asyncio.sleep(settings.POLL_INTERVAL)
                
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            await asyncio.sleep(settings.POLL_INTERVAL)
