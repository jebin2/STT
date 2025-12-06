# Audio Caption Generator

A Python-based audio transcription service with a neobrutalist web interface. Upload audio files via API, process them with STT (Speech-to-Text), and view results in a stunning UI.

## Features

- 🎵 Audio file upload via REST API
- 🤖 Automatic STT processing using faster-whisper
- 💾 SQLite database for queue management
- 🎨 Neobrutalist UI with smooth animations
- 🔄 Real-time status updates
- 📱 Fully responsive design

## Project Structure

```
audio-caption-project/
├── app.py              # Flask API server
├── worker.py           # Background STT processing service
├── index.html          # Frontend UI
├── requirements.txt    # Python dependencies
├── audio_captions.db   # SQLite database (auto-created)
└── uploads/            # Uploaded audio files (auto-created)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up STT Tool

Make sure you have the `stt-transcribe` tool available in your PATH or current directory. This should be the faster-whisper based transcription tool.

### 3. Start the API Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Start the Background Worker

In a separate terminal:

```bash
python worker.py
```

The worker will poll the database every 5 seconds for new files to process.

### 5. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Via Web Interface

1. Click or drag-and-drop an audio file onto the upload zone
2. Click "Upload & Process"
3. Watch the status update in real-time
4. View the generated caption once processing completes

### Via API

**Upload Audio File:**
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "audio=@/path/to/your/audio.wav"
```

**Get All Files:**
```bash
curl http://localhost:5000/api/files
```

**Get Specific File:**
```bash
curl http://localhost:5000/api/files/<file_id>
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload audio file |
| `/api/files` | GET | Get all files |
| `/api/files/<id>` | GET | Get specific file |
| `/health` | GET | Health check |

---

### `POST /api/upload`

Upload an audio file for transcription.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | File | Yes | Audio file (wav, mp3, flac, ogg, m4a, aac) |

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "audio.wav",
  "status": "not_started",
  "message": "File uploaded successfully"
}
```

**Error Responses:**

| Status | Response |
|--------|----------|
| 400 | `{"error": "No audio file provided"}` |
| 400 | `{"error": "No file selected"}` |
| 400 | `{"error": "Invalid file type"}` |

---

### `GET /api/files`

Retrieve all uploaded files with their status and captions.

**Request:** No parameters required.

**Response (200 OK):**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "audio.wav",
    "status": "completed",
    "caption": "{\"text\": \"Hello world...\", \"segments\": [...]}",
    "created_at": "2024-01-15T10:30:00.000000",
    "processed_at": "2024-01-15T10:30:45.000000",
    "queue_position": null,
    "estimated_start_seconds": null
  },
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "filename": "recording.mp3",
    "status": "processing",
    "caption": null,
    "created_at": "2024-01-15T10:35:00.000000",
    "processed_at": null,
    "queue_position": null,
    "estimated_start_seconds": null
  },
  {
    "id": "770e8400-e29b-41d4-a716-446655440002",
    "filename": "interview.wav",
    "status": "not_started",
    "caption": null,
    "created_at": "2024-01-15T10:40:00.000000",
    "processed_at": null,
    "queue_position": 1,
    "estimated_start_seconds": 45
  },
  {
    "id": "880e8400-e29b-41d4-a716-446655440003",
    "filename": "podcast.mp3",
    "status": "not_started",
    "caption": null,
    "created_at": "2024-01-15T10:45:00.000000",
    "processed_at": null,
    "queue_position": 2,
    "estimated_start_seconds": 90
  }
]
```

---

### `GET /api/files/<file_id>`

Retrieve a specific file by its ID.

**Request:**

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `file_id` | string | URL path | UUID of the file |

**Response (200 OK):**

*Example 1: Completed file*
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "audio.wav",
  "status": "completed",
  "caption": "{\"text\": \"Hello world...\", \"segments\": [...]}",
  "created_at": "2024-01-15T10:30:00.000000",
  "processed_at": "2024-01-15T10:30:45.000000",
  "queue_position": null,
  "estimated_start_seconds": null
}
```

*Example 2: File in queue (3rd position)*
```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "filename": "interview.wav",
  "status": "not_started",
  "caption": null,
  "created_at": "2024-01-15T10:40:00.000000",
  "processed_at": null,
  "queue_position": 3,
  "estimated_start_seconds": 135
}
```

**Error Responses:**

| Status | Response |
|--------|----------|
| 404 | `{"error": "File not found"}` |

---

### `GET /health`

Check the health status of the service.

**Request:** No parameters required.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "audio-caption-generator",
  "worker_running": true
}
```

## Database Schema

```sql
CREATE TABLE audio_files (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    status TEXT NOT NULL,
    caption TEXT,
    created_at TEXT NOT NULL,
    processed_at TEXT
);
```

## Status Values

| Status | Description | `queue_position` | `estimated_start_seconds` |
|--------|-------------|------------------|---------------------------|
| `not_started` | File uploaded, waiting in queue for processing | **Integer (1-based)** - Position in queue (1 = next to be processed) | **Integer** - Estimated seconds until processing starts |
| `processing` | Currently being transcribed by the worker | `null` | `null` |
| `completed` | Successfully transcribed | `null` | `null` |
| `failed` | Error occurred during transcription | `null` | `null` |

> **Note:** 
> - `queue_position`: Indicates the file's position in the processing queue. A value of `1` means this file is next to be processed.
> - `estimated_start_seconds`: Calculated based on the average processing time of the last 20 completed files. If no files have been processed yet, defaults to 30 seconds per file. The estimate accounts for both files ahead in the queue and any file currently being processed.

## Configuration

Edit these variables in `worker.py` to customize:

```python
CWD = "./"                          # Working directory
PYTHON_PATH = "stt-transcribe"      # Path to STT tool
STT_MODEL_NAME = "fasterwhispher"   # Model name
POLL_INTERVAL = 5                    # Polling interval in seconds
```

## Supported Audio Formats

- WAV
- MP3
- FLAC
- OGG
- M4A
- AAC

## Troubleshooting

**Worker not processing files:**
- Ensure the `stt-transcribe` tool is properly installed
- Check that the temp_dir exists for output
- Verify the audio file path is correct

**CORS errors:**
- Make sure flask-cors is installed
- Check that the API server is running

**Database errors:**
- Delete `audio_captions.db` and restart the API server to recreate it

## Tech Stack

- **Backend:** Flask (Python)
- **Database:** SQLite
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **STT:** faster-whisper
- **Design:** Neobrutalism with neon accents

## License

MIT