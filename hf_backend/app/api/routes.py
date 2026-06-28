from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import json
import asyncio
import aiofiles
from app.core.config import settings
from custom_logger import logger_config as logger
from app.db import crud
from app.services.worker import start_worker, is_worker_running
from app.services.streaming import StreamingSTT, ALLOWED_MODELS

router = APIRouter()

# Per-process connection counter. With multiple uvicorn workers each process has
# its own counter, so the cap is per-worker, not global.
ACTIVE_WS_CONNECTIONS = 0
MAX_WS_CONNECTIONS = 4

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in settings.ALLOWED_EXTENSIONS

@router.get("/")
async def index():
    return FileResponse('index.html')

@router.post("/api/tasks/upload")
async def upload_task(audio: UploadFile = File(...), hide_from_ui: str = Form("")):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not allowed_file(audio.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    task_id = str(uuid.uuid4())
    filename = audio.filename
    filepath = os.path.join(settings.UPLOAD_FOLDER, f"{task_id}_{filename}")
    
    try:
        async with aiofiles.open(filepath, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)
        logger.info(f"File uploaded successfully: {filename} -> {filepath}")
    except Exception as e:
        logger.error(f"Error saving uploaded file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not save file")
    
    hide_from_ui_val = 1 if hide_from_ui.lower() in ['true', '1'] else 0
    
    await crud.insert_task(task_id, filename, filepath, 'not_started', hide_from_ui_val)
    
    await start_worker()
    
    return JSONResponse(status_code=201, content={
        'id': task_id,
        'filename': filename,
        'status': 'not_started',
        'message': 'File uploaded successfully'
    })

@router.get("/api/tasks")
async def get_tasks():
    rows, queue_ids, processing_count, avg_time = await crud.get_all_tasks()
    
    tasks = []
    for row in rows:
        queue_position = None
        estimated_start_seconds = None
        
        if row['status'] == 'not_started' and row['id'] in queue_ids:
            queue_position = queue_ids.index(row['id']) + 1
            tasks_ahead = queue_position - 1 + processing_count
            estimated_start_seconds = round(tasks_ahead * avg_time)
        
        tasks.append({
            'id': row['id'],
            'filename': row['filename'],
            'status': row['status'],
            'result': "HIDDEN_IN_LIST_VIEW",
            'created_at': row['created_at'],
            'processed_at': row['processed_at'],
            'progress': row['progress'] or 0,
            'progress_text': row['progress_text'],
            'queue_position': queue_position,
            'estimated_start_seconds': estimated_start_seconds
        })
    
    return tasks

@router.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    result = await crud.get_task_by_id(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
        
    row, queue_position, estimated_start_seconds = result
    
    return {
        'id': row['id'],
        'filename': row['filename'],
        'status': row['status'],
        'result': row['result'],
        'created_at': row['created_at'],
        'processed_at': row['processed_at'],
        'progress': row['progress'] or 0,
        'progress_text': row['progress_text'],
        'queue_position': queue_position,
        'estimated_start_seconds': estimated_start_seconds
    }

@router.get("/health")
async def health():
    return {
        'status': 'healthy',
        'service': 'stt-backend',
        'worker_running': is_worker_running(),
        'ws_connections': ACTIVE_WS_CONNECTIONS,
    }

@router.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    global ACTIVE_WS_CONNECTIONS

    if ACTIVE_WS_CONNECTIONS >= MAX_WS_CONNECTIONS:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Server busy, try again later"})
        await websocket.close()
        return

    await websocket.accept()
    ACTIVE_WS_CONNECTIONS += 1
    logger.info(f"WebSocket connected ({ACTIVE_WS_CONNECTIONS}/{MAX_WS_CONNECTIONS})")

    stt = None
    task = None
    connected = True

    try:
        config_text = await websocket.receive_text()
        config = json.loads(config_text)
        model_name = config.get("model", "base")

        if model_name not in ALLOWED_MODELS:
            await websocket.send_json(
                {"type": "error", "message": f"Unsupported model: {model_name}"}
            )
            await websocket.close()
            return

        loop = asyncio.get_event_loop()
        # WhisperModel construction downloads/loads weights synchronously; run it
        # in an executor so it doesn't block the event loop (and every other
        # connection) while the model loads.
        stt = await loop.run_in_executor(
            None, lambda: StreamingSTT(model_name=model_name, device="cpu")
        )
        await websocket.send_json({"type": "ready", "sample_rate": stt.sample_rate})

        async def bg_process():
            while True:
                await asyncio.sleep(1.0)
                try:
                    result = await loop.run_in_executor(None, stt.process)
                    if not result:
                        continue
                    try:
                        if result["commit"]:
                            await websocket.send_json({"type": "commit", **result["commit"]})
                        await websocket.send_json(
                            {"type": "tentative", "text": result["tentative"]}
                        )
                    except Exception:
                        return
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.error(f"bg_process error: {e}")
                    return

        task = asyncio.create_task(bg_process())

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if message.get("bytes") is not None:
                stt.add_audio(message["bytes"])
            elif message.get("text") is not None:
                try:
                    msg = json.loads(message["text"])
                except (ValueError, TypeError):
                    continue
                if msg.get("type") == "stop":
                    # Client asked to finalize: stop reading and let the finally
                    # block flush the trailing audio while still connected.
                    break

    except WebSocketDisconnect:
        connected = False
        logger.info("WebSocket disconnected")
    except Exception as e:
        connected = False
        logger.error(f"WebSocket error: {e}")
    finally:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"bg_process cleanup error: {e}")
        if stt:
            if connected:
                try:
                    remaining = stt.flush()
                    if remaining and remaining["commit"]:
                        await websocket.send_json(
                            {"type": "commit", **remaining["commit"], "is_final": True}
                        )
                    await websocket.send_json({"type": "done"})
                    await websocket.close()
                except Exception:
                    pass
            stt.cleanup()
        ACTIVE_WS_CONNECTIONS -= 1
        logger.info(f"WebSocket closed ({ACTIVE_WS_CONNECTIONS}/{MAX_WS_CONNECTIONS})")
