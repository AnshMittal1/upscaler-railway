import logging
import os
from pathlib import Path
import requests
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from io import BytesIO
from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact

# Configure logger
t_logging = {
    'level': logging.INFO,
    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
}
logging.basicConfig(**t_logging)
logger = logging.getLogger(__name__)

# Model download settings
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / 'realesr-general-x4v3.pth'
MODEL_URL = 'https://huggingface.co/jhj0517/realesr-general-x4v3/resolve/main/realesr-general-x4v3.pth'
MIN_MODEL_SIZE = 10 * 1024 * 1024  # 10 MB

# Ensure model is present and valid
def ensure_model():
    if MODEL_FILE.exists():
        size = MODEL_FILE.stat().st_size
        if size >= MIN_MODEL_SIZE:
            logger.info(f"Model already exists at {MODEL_FILE}, size={size} bytes")
            return
        else:
            logger.warning(f"Existing model at {MODEL_FILE} is too small ({size} bytes), re-downloading.")
            MODEL_FILE.unlink()

    logger.info(f"Downloading model from: {MODEL_URL}")
    try:
        rsp = requests.get(MODEL_URL, stream=True, timeout=60)
        rsp.raise_for_status()
        total = int(rsp.headers.get('content-length', 0))
        with open(MODEL_FILE, 'wb') as f:
            downloaded = 0
            for chunk in rsp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    logger.info(f"Download progress: {percent:.2f}%")
        final_size = MODEL_FILE.stat().st_size
        logger.info(f"Downloaded model to {MODEL_FILE}, size={final_size} bytes")
    except Exception as e:
        logger.error(f"Failed to download the model: {e}")
        raise RuntimeError("Could not download RealESRGAN model")

# Monkey-patch torch.load to load full checkpoint on PyTorch>=2.6
_torch_load = torch.load

def _load_full(f, *args, **kwargs):
    kwargs.pop('weights_only', None)
    return _torch_load(f, *args, weights_only=False, **kwargs)

torch.load = _load_full

# Download model if needed
ensure_model()

# Initialize FastAPI
app = FastAPI()

# Initialize RealESRGAN upsampler
try:
    logger.info("Initializing RealESRGAN model...")
    model_arch = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_conv=32, upscale=4, act_type='prelu'
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(MODEL_FILE),
        model=model_arch,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None
    )
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    upsampler = None

@app.on_event("startup")
async def startup_event():
    if upsampler is None:
        logger.error("Upsampler not ready after startup.")
    else:
        logger.info("Application startup complete, upsampler ready.")

@app.get("/health")
async def health_check():
    return {"status": "healthy" if upsampler else "unhealthy", "model_loaded": bool(upsampler)}

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...), outscale: int = Form(4)):
    if not upsampler:
        raise HTTPException(status_code=500, detail="Model not initialized properly")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    if outscale < 1 or outscale > 8:
        raise HTTPException(status_code=400, detail="outscale must be between 1 and 8")
    logger.info(f"Upscaling: file={file.filename}, factor={outscale}")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except Exception as e:
        logger.error(f"Enhance failed: {e}")
        raise HTTPException(status_code=500, detail="Upscaling failed")
    success, encoded = cv2.imencode('.png' if file.content_type=='image/png' else '.jpg', output)
    if not success:
        logger.error("Failed to encode output image.")
        raise HTTPException(status_code=500, detail="Failed to encode output")
    return StreamingResponse(BytesIO(encoded.tobytes()), media_type=file.content_type,
                             headers={"Content-Disposition": f"attachment; filename=upscaled_{file.filename}"})

@app.get("/")
async def root():
    return {"message": "Image Upscaler API", "model_loaded": bool(upsampler)}
