from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from io import BytesIO
from realesrgan import RealESRGANer
import os
import requests
from pathlib import Path
import logging

import torch
torch.load = lambda f, **kw: __import__('torch').load(f, weights_only=False, **kw)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "realesr-general-x4v3.pth"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Log existence and size of the model file
exists = MODEL_FILE.is_file()
size = MODEL_FILE.stat().st_size if exists else "N/A"
logger.info(f"Model file path: {MODEL_FILE}")
logger.info(f"MODEL EXISTS? {exists}, SIZE: {size}")

# Initialize FastAPI app
app = FastAPI()

# Initialize the model upsampler
tile, tile_pad, pre_pad, half, gpu = 0, 10, 0, False, None
upsampler = None
try:
    # Build architecture
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type='prelu'
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(MODEL_FILE),
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=gpu
    )
    logger.info("Model initialized successfully with SRVGGNetCompact architecture.")
except ImportError as e:
    logger.warning(f"Could not import SRVGGNetCompact: {e}. Trying fallback RealESRGANer...")
    try:
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(MODEL_FILE),
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            gpu_id=gpu
        )
        logger.info("Model initialized successfully with fallback RealESRGANer.")
    except Exception as e2:
        logger.error(f"Fallback initialization failed: {e2}")
except Exception as e:
    logger.error(f"Model initialization failed: {e}")

@app.on_event("startup")
async def startup_event():
    status = "ready" if upsampler else "not initialized"
    logger.info(f"Application startup complete. Upsampler status: {status}.")

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...), outscale: int = Form(4)):
    if upsampler is None:
        logger.error("Upscaler called but model is not initialized.")
        raise HTTPException(status_code=500, detail="Model not initialized properly")

    if not file.content_type or not file.content_type.startswith('image/'):
        logger.error("Invalid file type: %s", file.content_type)
        raise HTTPException(status_code=400, detail="File must be an image")

    if outscale < 1 or outscale > 8:
        logger.error("Invalid outscale parameter: %d", outscale)
        raise HTTPException(status_code=400, detail="outscale must be between 1 and 8")

    logger.info(f"Starting upscaling: file={file.filename}, outscale={outscale}")

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("cv2.imdecode returned None for file %s", file.filename)
            raise HTTPException(status_code=400, detail="Invalid image file or unsupported format")
        logger.info(f"Decoded input image shape: {img.shape}")
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
        logger.info(f"Upscaled image shape: {output.shape}")
    except Exception as e:
        logger.error(f"Upscaling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upscaling failed: {str(e)}")

    try:
        encode_format = '.png' if file.content_type == 'image/png' else '.jpg'
        success, encoded_img = cv2.imencode(encode_format, output)
        if not success:
            logger.error("cv2.imencode failed for output image.")
            raise HTTPException(status_code=500, detail="Failed to encode output image")
        media_type = "image/png" if encode_format == '.png' else "image/jpeg"
        img_bytes = encoded_img.tobytes()
        logger.info(f"Returning upscaled image of size {len(img_bytes)} bytes.")
        return StreamingResponse(
            BytesIO(img_bytes),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=upscaled_{file.filename}"}
        )
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

@app.get("/")
async def root():
    logger.info("Root endpoint called.")
    return {
        "message": "Welcome to the Image Upscaler API",
        "endpoints": {
            "POST /upscale": "Upload an image to upscale it",
            "parameters": {"file": "Image file (JPEG, PNG, etc.)", "outscale": "Scaling factor (1-8, default: 4)"}
        },
        "model_status": "Ready" if upsampler else "Not initialized"
    }

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called. Status: %s", "healthy" if upsampler else "unhealthy")
    return {"status": "healthy" if upsampler else "unhealthy", "model_loaded": upsampler is not None}
