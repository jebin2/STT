from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import aiofiles
from app.core.config import settings
from custom_logger import logger_config as logger
from app.db import crud
from app.services.worker import start_worker, is_worker_running

router = APIRouter()

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
        'worker_running': is_worker_running()
    }
