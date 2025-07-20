from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from io import BytesIO
from realesrgan import RealESRGANer
import os
import requests
from pathlib import Path
from basicsr.archs.srvgg_arch import SRVGGNetCompact

app = FastAPI()

# Create models directory if it doesn't exist
models_dir = Path('./models')
models_dir.mkdir(exist_ok=True)

# Model path
model_path = models_dir / 'realesr-general-x4v3.pth'



# Initialize the model
try:
    # Import the correct architecture for realesr-general-x4v3
    
    
    # Create the correct model architecture for realesr-general-x4v3
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type='prelu'
    )
    
    # Initialize upsampler with the correct model
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None  # Use CPU, set to 0 for GPU if CUDA available
    )
    print("Model initialized successfully!")
    
except ImportError as e:
    print(f"Could not import SRVGGNetCompact. Trying fallback approach: {e}")
    # Fallback: try without specifying model architecture
    try:
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=None
        )
        print("Model initialized with fallback method!")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        upsampler = None
        
except Exception as e:
    print(f"Failed to initialize model: {e}")
    upsampler = None

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...), outscale: int = Form(4)):
    """
    Upscale an uploaded image by the given outscale factor.

    - **file**: image file to upscale (JPEG, PNG, etc.)
    - **outscale**: scaling factor for upscaling (integer, default: 4)
    """
    
    # Check if upsampler is initialized
    if upsampler is None:
        raise HTTPException(status_code=500, detail="Model not initialized properly")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate outscale parameter
    if outscale < 1 or outscale > 8:
        raise HTTPException(status_code=400, detail="outscale must be between 1 and 8")
    
    print(f"Starting upscaling with factor: {outscale}")
    
    try:
        # Read and decode image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file or unsupported format")
            
        print(f"Input image shape: {img.shape}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    # Perform upscaling
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
        print(f"Output image shape: {output.shape}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upscaling failed: {str(e)}")

    # Encode output image
    try:
        # Use PNG for better quality, or keep JPEG for smaller files
        encode_format = '.png' if file.content_type == 'image/png' else '.jpg'
        success, encoded_img = cv2.imencode(encode_format, output)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode output image")
        
        # Determine media type
        media_type = "image/png" if encode_format == '.png' else "image/jpeg"
        
        # Return image
        img_bytes = encoded_img.tobytes()
        return StreamingResponse(
            BytesIO(img_bytes), 
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=upscaled_{file.filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Image Upscaler API", 
        "endpoints": {
            "POST /upscale": "Upload an image to upscale it",
            "parameters": {
                "file": "Image file (JPEG, PNG, etc.)",
                "outscale": "Scaling factor (1-8, default: 4)"
            }
        },
        "model_status": "Ready" if upsampler is not None else "Not initialized"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if upsampler is not None else "unhealthy",
        "model_loaded": upsampler is not None
    }