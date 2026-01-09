import base64
import io
import os
from PIL import Image
import torch
import transformers
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, AutoModelForDepthEstimation, AutoProcessor, AutoModelForCausalLM
from config import logger, BLIP_MODEL_NAME, BLIP_MAX_NEW_TOKENS, DEPTH_MODEL_NAME

# Minimum transformers version required for Florence support (approx)
MIN_FLORENCE_TRANSFORMERS = (4, 52, 0)  # use tuple compare for robustness

def _version_tuple(ver_str: str):
    parts = ver_str.split(".")
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except Exception:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])

class VisionSystem:
    def __init__(self):
        logger.info(f"Loading VLM from {BLIP_MODEL_NAME}...")
        
        # Determine dtype based on device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        self.is_florence = "florence" in BLIP_MODEL_NAME.lower()

        # If user explicitly wants Florence, ensure transformers is new enough
        if self.is_florence:
            tf_ver = _version_tuple(transformers.__version__)
            if tf_ver < MIN_FLORENCE_TRANSFORMERS:
                msg = (
                    f"Detected transformers {transformers.__version__}; Florence requires >= {'.'.join(map(str,MIN_FLORENCE_TRANSFORMERS))}. "
                    "Please upgrade: pip install -U 'transformers[torch]>=4.52.0' and restart."
                )
                logger.error(msg)
                # If strict mode is set, abort so user notices and can upgrade
                if os.getenv("USE_FLORENCE_STRICT", "0") == "1":
                    raise RuntimeError(msg)
                # Otherwise, fallback to BLIP gracefully
                logger.info("Falling back to BLIP base model due to transformers version.")
                self.is_florence = False

        if self.is_florence:
            # Florence can use custom code; attempt to load it but be resilient to failures
            try:
                self.processor = AutoProcessor.from_pretrained(BLIP_MODEL_NAME, trust_remote_code=True)
                # Try modern `dtype` kwarg first, fall back to `torch_dtype` for older transformers
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        BLIP_MODEL_NAME,
                        dtype=self.dtype,
                        trust_remote_code=True,
                    )
                except TypeError:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        BLIP_MODEL_NAME,
                        torch_dtype=self.dtype,
                        trust_remote_code=True,
                    )
            except Exception as e:
                # If Florence initialization fails, give actionable error about environment
                logger.error(f"Failed to load Florence VLM ({BLIP_MODEL_NAME}): {e}")
                logger.info("Tip: Florence may require a newer transformers version or specific platform support. Try upgrading to transformers >= 4.52 if you want to use Florence.")
                # If strict mode is set, raise so user can address the environment
                if os.getenv("USE_FLORENCE_STRICT", "0") == "1":
                    raise
                # Otherwise fall back to BLIP so the agent can keep running
                logger.info("Falling back to BLIP base model (Salesforce/blip-image-captioning-base)")
                self.is_florence = False
                fallback = "Salesforce/blip-image-captioning-base"
                self.processor = BlipProcessor.from_pretrained(fallback)
                try:
                    self.model = BlipForConditionalGeneration.from_pretrained(fallback, dtype=self.dtype)
                except TypeError:
                    try:
                        self.model = BlipForConditionalGeneration.from_pretrained(fallback, torch_dtype=self.dtype)
                    except TypeError:
                        self.model = BlipForConditionalGeneration.from_pretrained(fallback)
        else:
            self.processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
            try:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    BLIP_MODEL_NAME,
                    dtype=self.dtype,
                )
            except TypeError:
                try:
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        BLIP_MODEL_NAME,
                        torch_dtype=self.dtype,
                    )
                except TypeError:
                    self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"VLM loaded on {self.device} with {self.dtype} (Florence={self.is_florence})")

        logger.info("Loading Depth Anything...")
        try:
            self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
            try:
                self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                    DEPTH_MODEL_NAME,
                    dtype=self.dtype,
                )
            except TypeError:
                try:
                    self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                        DEPTH_MODEL_NAME,
                        torch_dtype=self.dtype,
                    )
                except TypeError:
                    self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            self.depth_enabled = True
            logger.info(f"Depth Anything loaded on {self.device} with {self.dtype}")
        except Exception as e:
            logger.error(f"Failed to load Depth Anything: {e}")
            self.depth_enabled = False

    def caption_b64(self, image_b64: str) -> str:
        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Resize image to speed up processing
            # Florence-2 can handle larger but 384 is safe for speed. 
            # If accuracy is poor, we can bump this up.
            img = img.resize((384, 384))
            
            if self.is_florence:
                prompt = "<MORE_DETAILED_CAPTION>"
                inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device, self.dtype)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=BLIP_MAX_NEW_TOKENS * 3, # Florence tends to be more verbose
                        do_sample=False,
                        num_beams=3,
                    )
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, 
                    task=prompt, 
                    image_size=(img.width, img.height)
                )
                cap = parsed_answer[prompt]
                
                # Cleanup excessively long output if needed or take first sentence if too verbose
                # config.py has limited context window, so we might want to truncate
                # but user asked for "meaningful clues", so longer is better for now.
                
            else:
                inputs = self.processor(img, return_tensors="pt").to(self.device, self.dtype)
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=BLIP_MAX_NEW_TOKENS)
                cap = self.processor.decode(out[0], skip_special_tokens=True).strip()
            
            return cap
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return "camera malfunction"

    def analyze_depth_b64(self, image_b64: str) -> str:
        if not self.depth_enabled:
            return "depth_unavailable"
        
        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Downscale for depth estimation speedup
            img = img.resize((518, 518)) # Depth Anything standard size
            
            # Predict depth
            inputs = self.depth_processor(images=img, return_tensors="pt").to(self.device, self.dtype)
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to lower res for analysis (e.g. 50x50)
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(50, 50),
                mode="bicubic",
                align_corners=False,
            )
            depth_map = prediction.squeeze().cpu() # 50x50 tensor

            # Normalize to 0-1 (relative depth in this frame)
            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max - d_min < 1e-6:
                return "depth_flat_or_error"
            
            norm_depth = (depth_map - d_min) / (d_max - d_min)
            
            # Define regions (Top=Background, Center=Obstacle path, Bottom=Ground)
            # Rows 0-15: Top
            # Rows 15-35: Center
            # Rows 35-50: Bottom
            
            top_mean = norm_depth[0:15, :].mean().item()
            center_mean = norm_depth[15:35, 15:35].mean().item() # Center crop
            bottom_mean = norm_depth[35:50, :].mean().item()
            
            # Heuristic interpretation
            # Close/Bright = 1.0 (High value) on inverse depth maps usually?
            # Let's assume standard output where Higher = Closer.
            
            # Check gradients
            # Clear path: Bottom (~1.0) >> Center (~0.5) >> Top (~0.0)
            # Obstacle: Bottom (~1.0) ≈ Center (~1.0) >> Top
            # Wall: Bottom ≈ Center ≈ Top
            
            return f"Close(ground)={bottom_mean:.2f}, Mid(ahead)={center_mean:.2f}, Far(sky)={top_mean:.2f}"

        except Exception as e:
            logger.error(f"Depth error: {e}")
            return "depth_error"
