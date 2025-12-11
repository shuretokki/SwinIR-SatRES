from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import logging
from pathlib import Path
from inference import SwinIRInference
import time
import json
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SwinIR Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    logger.info("Starting up...")

    model_path = Path(__file__).parent.parent / "experiments" / "models" / "example_vdroneswin.onnx"

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = SwinIRInference(str(model_path))
    logger.info("Model loaded successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "provider": model.session.get_providers()[0] if model else None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()

    try:
        while True:
            # Wait for image data
            data = await websocket.receive_json()

            if data.get("type") == "infer":
                image_data = data.get("image")
                max_size = data.get("max_size", 512)

                # Decode base64 image
                import base64
                image_bytes = base64.b64decode(image_data.split(',')[1])

                # Progress callback
                async def progress_callback(stage: str, progress: float, message: str):
                    await websocket.send_json({
                        "type": "progress",
                        "stage": stage,
                        "progress": progress,
                        "message": message
                    })

                try:
                    # Run inference with progress
                    start_time = time.time()
                    result_bytes = await model.infer_async(
                        image_bytes,
                        max_size=max_size,
                        progress_callback=progress_callback
                    )
                    inference_time = time.time() - start_time

                    # Send result
                    result_b64 = base64.b64encode(result_bytes).decode('utf-8')
                    await websocket.send_json({
                        "type": "result",
                        "image": f"data:image/png;base64,{result_b64}",
                        "inference_time": inference_time
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    max_size: int = 512
):
    """
    Run inference on uploaded image (legacy HTTP endpoint).

    Args:
        file: Image file (PNG, JPG, etc.)
        max_size: Maximum dimension for input (default: 512)

    Returns:
        Enhanced image as PNG
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        logger.info(f"Processing file: {file.filename}")
        image_bytes = await file.read()

        start_time = time.time()
        result_bytes = model.infer(image_bytes, max_size=max_size)
        inference_time = time.time() - start_time

        logger.info(f"Inference completed in {inference_time:.2f}s")

        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={
                "X-Inference-Time": str(inference_time),
                "Content-Disposition": f"attachment; filename=enhanced_{file.filename}"
            }
        )

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
