from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
import subprocess
import time

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('temp_dir', exist_ok=True)

# Worker state
worker_thread = None
worker_running = False

def init_db():
    conn = sqlite3.connect('audio_captions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audio_files
                 (id TEXT PRIMARY KEY,
                  filename TEXT NOT NULL,
                  filepath TEXT NOT NULL,
                  status TEXT NOT NULL,
                  caption TEXT,
                  created_at TEXT NOT NULL,
                  processed_at TEXT)''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def start_worker():
    """Start the worker thread if not already running"""
    global worker_thread, worker_running
    
    if not worker_running:
        worker_running = True
        worker_thread = threading.Thread(target=worker_loop, daemon=True)
        worker_thread.start()
        print("✅ Worker thread started")

def cleanup_old_entries():
    """Delete database entries and audio files older than 10 days"""
    from datetime import timedelta
    
    try:
        conn = sqlite3.connect('audio_captions.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Calculate cutoff date (10 days ago)
        cutoff_date = (datetime.now() - timedelta(days=10)).isoformat()
        
        # First, get all old entries to delete their audio files
        c.execute('''SELECT id, filepath FROM audio_files 
                     WHERE created_at < ?''', (cutoff_date,))
        old_entries = c.fetchall()
        
        if old_entries:
            deleted_files = 0
            deleted_rows = 0
            
            for entry in old_entries:
                # Delete the audio file if it exists
                filepath = entry['filepath']
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        deleted_files += 1
                    except Exception as e:
                        print(f"⚠️  Failed to delete old audio file {filepath}: {e}")
            
            # Delete old database entries
            c.execute('''DELETE FROM audio_files WHERE created_at < ?''', (cutoff_date,))
            deleted_rows = c.rowcount
            conn.commit()
            
            if deleted_rows > 0 or deleted_files > 0:
                print(f"🧹 Cleanup: Deleted {deleted_rows} old entries and {deleted_files} audio files (older than 10 days)")
        
        conn.close()
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")

def worker_loop():
    """Main worker loop that processes audio files"""
    print("🤖 STT Worker started. Monitoring for new audio files...")
    
    CWD = "./"
    PYTHON_PATH = "stt-transcribe"
    STT_MODEL_NAME = "parakeet"
    POLL_INTERVAL = 3  # seconds
    
    import shlex
    import json
    
    while worker_running:
        # Run cleanup before processing each task
        cleanup_old_entries()
        try:
            # Get next unprocessed file
            conn = sqlite3.connect('audio_captions.db')
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('''SELECT * FROM audio_files 
                         WHERE status = 'not_started' 
                         ORDER BY created_at ASC 
                         LIMIT 1''')
            row = c.fetchone()
            conn.close()
            
            if row:
                file_id = row['id']
                filepath = row['filepath']
                filename = row['filename']
                
                print(f"\n{'='*60}")
                print(f"🎵 Processing: {filename}")
                print(f"📝 ID: {file_id}")
                print(f"{'='*60}")
                
                # Update status to processing
                update_status(file_id, 'processing')
                
                try:
                    # Run STT command
                    print(f"🔄 Running STT on: {os.path.abspath(filepath)}")
                    command = f"""cd {CWD} && {PYTHON_PATH} --input {shlex.quote(os.path.abspath(filepath))} --model {STT_MODEL_NAME}"""
                    
                    subprocess.run(
                        command,
                        shell=True,
                        executable="/bin/bash",
                        check=True,
                        cwd=CWD,
                        env={
                            **os.environ,
                            'PYTHONUNBUFFERED': '1',
                            'CUDA_LAUNCH_BLOCKING': '1',
                            'USE_CPU_IF_POSSIBLE': 'true'
                        }
                    )
                    
                    # Read transcription result
                    output_path = f'{CWD}/temp_dir/output_transcription.json'
                    with open(output_path, 'r') as file:
                        result = json.loads(file.read().strip())
                    
                    # Extract caption text
                    caption = result.get('text', '') or result.get('transcription', '') or str(result)
                    
                    print(f"✅ Successfully processed: {filename}")
                    print(f"📄 Caption preview: {caption[:100]}...")
                    
                    # Update database with success
                    update_status(file_id, 'completed', caption=json.dumps(result))
                    
                    # Delete the audio file after successful processing
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"🗑️  Deleted audio file: {filepath}")
                    
                except Exception as e:
                    print(f"❌ Failed to process: {filename}")
                    print(f"Error: {str(e)}")
                    update_status(file_id, 'failed', error=str(e))
                    
                    # Don't delete file on failure (for debugging)
                    # Optionally delete after some time or manual review
                    
            else:
                # No files to process, sleep for a bit
                time.sleep(POLL_INTERVAL)
                
        except Exception as e:
            print(f"⚠️  Worker error: {str(e)}")
            time.sleep(POLL_INTERVAL)

def update_status(file_id, status, caption=None, error=None):
    """Update the status of a file in the database"""
    conn = sqlite3.connect('audio_captions.db')
    c = conn.cursor()
    
    if status == 'completed':
        c.execute('''UPDATE audio_files 
                     SET status = ?, caption = ?, processed_at = ?
                     WHERE id = ?''',
                  (status, caption, datetime.now().isoformat(), file_id))
    elif status == 'failed':
        c.execute('''UPDATE audio_files 
                     SET status = ?, caption = ?, processed_at = ?
                     WHERE id = ?''',
                  (status, f"Error: {error}", datetime.now().isoformat(), file_id))
    else:
        c.execute('UPDATE audio_files SET status = ? WHERE id = ?', (status, file_id))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    file_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
    file.save(filepath)
    
    conn = sqlite3.connect('audio_captions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO audio_files 
                 (id, filename, filepath, status, created_at)
                 VALUES (?, ?, ?, ?, ?)''',
              (file_id, filename, filepath, 'not_started', datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    # Start worker on first upload
    start_worker()
    
    return jsonify({
        'id': file_id,
        'filename': filename,
        'status': 'not_started',
        'message': 'File uploaded successfully'
    }), 201

def get_average_processing_time(cursor):
    """Calculate average processing time from completed files in seconds"""
    cursor.execute('''SELECT created_at, processed_at FROM audio_files 
                      WHERE status = 'completed' AND processed_at IS NOT NULL
                      ORDER BY processed_at DESC LIMIT 20''')
    completed_rows = cursor.fetchall()
    
    if not completed_rows:
        return 30.0  # Default estimate: 30 seconds per file
    
    total_seconds = 0
    count = 0
    for r in completed_rows:
        try:
            created = datetime.fromisoformat(r['created_at'])
            processed = datetime.fromisoformat(r['processed_at'])
            duration = (processed - created).total_seconds()
            if duration > 0:
                total_seconds += duration
                count += 1
        except:
            continue
    
    return total_seconds / count if count > 0 else 30.0

@app.route('/api/files', methods=['GET'])
def get_files():
    conn = sqlite3.connect('audio_captions.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get average processing time
    avg_time = get_average_processing_time(c)
    
    # Get queue (files waiting to be processed, ordered by creation time)
    c.execute('''SELECT id FROM audio_files 
                 WHERE status = 'not_started' 
                 ORDER BY created_at ASC''')
    queue_ids = [row['id'] for row in c.fetchall()]
    
    # Check if there's a file currently processing
    c.execute('''SELECT COUNT(*) as count FROM audio_files WHERE status = 'processing' ''')
    processing_count = c.fetchone()['count']
    
    c.execute('SELECT * FROM audio_files ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    
    files = []
    for row in rows:
        # Calculate queue position (1-based) for files in queue
        queue_position = None
        estimated_start_seconds = None
        
        if row['status'] == 'not_started' and row['id'] in queue_ids:
            queue_position = queue_ids.index(row['id']) + 1
            # Estimate = (files ahead + currently processing) * avg time
            files_ahead = queue_position - 1 + processing_count
            estimated_start_seconds = round(files_ahead * avg_time)
        
        files.append({
            'id': row['id'],
            'filename': row['filename'],
            'status': row['status'],
            'caption': row['caption'],
            'created_at': row['created_at'],
            'processed_at': row['processed_at'],
            'queue_position': queue_position,
            'estimated_start_seconds': estimated_start_seconds
        })
    
    return jsonify(files)

@app.route('/api/files/<file_id>', methods=['GET'])
def get_file(file_id):
    conn = sqlite3.connect('audio_captions.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM audio_files WHERE id = ?', (file_id,))
    row = c.fetchone()
    
    if row is None:
        conn.close()
        return jsonify({'error': 'File not found'}), 404
    
    # Calculate queue position and estimated time if file is waiting
    queue_position = None
    estimated_start_seconds = None
    
    if row['status'] == 'not_started':
        # Get average processing time
        avg_time = get_average_processing_time(c)
        
        # Count files ahead in queue
        c.execute('''SELECT COUNT(*) as position FROM audio_files 
                     WHERE status = 'not_started' AND created_at < ?''',
                  (row['created_at'],))
        position_row = c.fetchone()
        queue_position = position_row['position'] + 1  # 1-based position
        
        # Check if there's a file currently processing
        c.execute('''SELECT COUNT(*) as count FROM audio_files WHERE status = 'processing' ''')
        processing_count = c.fetchone()['count']
        
        # Estimate = (files ahead + currently processing) * avg time
        files_ahead = queue_position - 1 + processing_count
        estimated_start_seconds = round(files_ahead * avg_time)
    
    conn.close()
    
    return jsonify({
        'id': row['id'],
        'filename': row['filename'],
        'status': row['status'],
        'caption': row['caption'],
        'created_at': row['created_at'],
        'processed_at': row['processed_at'],
        'queue_position': queue_position,
        'estimated_start_seconds': estimated_start_seconds
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'audio-caption-generator',
        'worker_running': worker_running
    })

if __name__ == '__main__':
    init_db()
    print("\n" + "="*60)
    print("🚀 Audio Caption Generator API Server")
    print("="*60)
    print("📌 Worker will start automatically on first upload")
    print("🗑️  Audio files will be deleted after successful processing")
    print("="*60 + "\n")
    
    # Use PORT environment variable for Hugging Face compatibility
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)