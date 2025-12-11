import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class SwinIRInference:
    def __init__(self, model_path: str):
        """Initialize the ONNX model with CUDA support."""
        logger.info(f"Loading model from {model_path}")

        available_providers = ort.get_available_providers()
        logger.info(f"Available providers: {available_providers}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in providers if p in available_providers]

        logger.info(f"Using providers: {providers}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"Model loaded. Input: {self.input_name}, Output: {self.output_name}")
        logger.info(f"Provider in use: {self.session.get_providers()}")

    def preprocess(self, image_bytes: bytes, max_size: int = 512) -> tuple[np.ndarray, tuple]:
        """Convert image bytes to ONNX tensor format."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size

        width, height = image.size
        max_dim = max(width, height)
        if max_dim > max_size:
            scale = max_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized from {original_size} to {image.size}")

        # Convert to numpy array (H, W, C)
        img_array = np.array(image).astype(np.float32) / 255.0

        # Convert to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension (1, C, H, W)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, image.size

    def postprocess(self, output_tensor: np.ndarray) -> bytes:
        """Convert ONNX output tensor to image bytes."""
        # Remove batch dimension
        output = output_tensor[0]

        # Convert from CHW to HWC
        output = np.transpose(output, (1, 2, 0))

        # Clip and convert to uint8
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(output)

        # Convert to bytes (PNG format)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()

    def infer(self, image_bytes: bytes, max_size: int = 512) -> bytes:
        """Run inference on image bytes and return result bytes."""
        logger.info("Starting inference")

        # Preprocess
        input_tensor, input_size = self.preprocess(image_bytes, max_size)
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        output_tensor = outputs[0]
        logger.info(f"Output tensor shape: {output_tensor.shape}")

        # Postprocess
        result_bytes = self.postprocess(output_tensor)
        logger.info("Inference complete")

        return result_bytes

    async def infer_async(self, image_bytes: bytes, max_size: int = 512, progress_callback=None) -> bytes:
        """Run inference with progress updates."""
        import asyncio

        if progress_callback:
            await progress_callback("preprocessing", 0.1, "Loading image...")

        # Preprocess
        input_tensor, input_size = self.preprocess(image_bytes, max_size)
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        if progress_callback:
            await progress_callback("preprocessing", 0.2, f"Preprocessed to {input_size}")

        # Run inference
        if progress_callback:
            await progress_callback("inference", 0.3, "Running model on GPU...")

        # Run in executor to not block
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )
        )

        if progress_callback:
            await progress_callback("inference", 0.8, "Inference complete")

        output_tensor = outputs[0]
        logger.info(f"Output tensor shape: {output_tensor.shape}")

        # Postprocess
        if progress_callback:
            await progress_callback("postprocessing", 0.9, "Generating image...")

        result_bytes = self.postprocess(output_tensor)

        if progress_callback:
            await progress_callback("done", 1.0, "Complete!")

        return result_bytes
