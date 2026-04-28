import os

class Config:
    PORT = int(os.environ.get('PORT', 7860))
    UPLOAD_FOLDER = 'uploads'
    TEMP_DIR = 'temp_dir'
    DATABASE_FILE = 'audio_captions.db'
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'mp4', 'mkv', 'avi', 'mov'}
    
    CWD = "./"
    PYTHON_PATH = "stt-transcribe"
    STT_MODEL_NAME = "parakeet"
    POLL_INTERVAL = 3

settings = Config()

os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)
