import base64
import io
import re
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

try:
    from config import logger, VLM_MODEL_NAME, VLM_MAX_NEW_TOKENS, DEPTH_MODEL_NAME
except ImportError:
    from .config import logger, VLM_MODEL_NAME, VLM_MAX_NEW_TOKENS, DEPTH_MODEL_NAME


# Default to GIT model which works reliably with transformers 5.x
# Florence-2 and Moondream2 have compatibility issues
DEFAULT_VLM_MODEL = "microsoft/git-large-coco"


class VisionSystem: 
    def __init__(self):
        """Initialize the vision system with captioning and depth models."""
        # Determine model to use
        model_name = VLM_MODEL_NAME
        
        if "florence" in model_name.lower() or "moondream" in model_name.lower():
            logger.warning(
                f"{model_name} has compatibility issues with transformers 5.x. "
                f"Using GIT model instead."
            )
            model_name = DEFAULT_VLM_MODEL
        
        logger.info(f"Loading VLM from {model_name}...")
        
        # Determine device and dtype for optimal performance
        self.device, self.dtype = self._select_device()
        self.model_name = model_name
        
        # Load the vision model
        self._load_vision_model()
        
        # Load Depth Anything for obstacle detection
        self._load_depth_model()
    
    def _select_device(self) -> tuple:
        """
        Select the best available device for inference.
        
        Returns:
            tuple: (device_name, torch_dtype)
        """
        if torch.cuda.is_available():
            return "cuda", torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.float16
        else:
            return "cpu", torch.float32
    
    def _load_vision_model(self):
        """Load the main vision-language model (GIT or BLIP fallback)."""
        if "git" in self.model_name.lower():
            self._load_git()
        else:
            self._load_blip_fallback()
    
    def _load_git(self):
        """
        Load GIT (Generative Image-to-text Transformer) for image captioning.
        
        GIT is a vision-language model that generates natural language descriptions
        of images. It works reliably with transformers 5.x on all platforms.
        """
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        logger.info(f"Loading GIT from {self.model_name}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_git = True
            logger.info(f"GIT loaded on {self.device} with {self.dtype}")
        except Exception as e:
            logger.error(f"Failed to load GIT: {e}")
            self._load_blip_fallback()
    
    def _load_blip_fallback(self):
        """
        Load BLIP as ultimate fallback captioning model.
        
        BLIP (Bootstrapping Language-Image Pre-training) is a reliable fallback
        when GIT is unavailable.
        """
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        fallback = "Salesforce/blip-image-captioning-base"
        logger.info(f"Loading BLIP fallback from {fallback}...")
        
        try:
            self.processor = BlipProcessor.from_pretrained(fallback)
            self.model = BlipForConditionalGeneration.from_pretrained(
                fallback,
                torch_dtype=self.dtype
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_git = False
            logger.info(f"BLIP loaded on {self.device} with {self.dtype}")
        except Exception as e:
            logger.error(f"Failed to load BLIP: {e}")
            raise RuntimeError("No vision model available") from e
    
    def _load_depth_model(self):
        """
        Load Depth Anything for obstacle detection.
        
        Depth Anything estimates per-pixel depth, which we use to detect
        obstacles in different regions (center, left, right) of the camera view.
        """
        try:
            logger.info("Loading Depth Anything...")
            self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                DEPTH_MODEL_NAME,
                torch_dtype=self.dtype
            )
            self.depth_model.to(self.device)
            self.depth_model.eval()
            self.depth_enabled = True
            logger.info(f"Depth Anything loaded on {self.device} with {self.dtype}")
        except Exception as e:
            logger.error(f"Failed to load Depth Anything: {e}")
            self.depth_enabled = False

    def caption_b64(self, image_b64: str) -> str:
        """
        Generate a caption for a base64-encoded image.
        
        Args:
            image_b64: Base64-encoded JPEG/PNG image data
            
        Returns:
            Human-readable caption describing the image content
        """
        try:
            # Decode and prepare image
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((384, 384))  # Resize for speed
            
            # Generate caption based on model type
            if self.is_git:
                cap = self._caption_git(img)
            else:
                cap = self._caption_blip(img)
            
            # Clean up caption artifacts
            return self._clean_caption(cap)
            
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return "camera malfunction"
    
    def _caption_git(self, img: Image.Image) -> str:
        """Generate caption using GIT model."""
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=4,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    
    def _caption_blip(self, img: Image.Image) -> str:
        """Generate caption using BLIP model."""
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean up caption by removing repetitions and artifacts.
        
        Args:
            caption: Raw caption from the model
            
        Returns:
            Cleaned caption without artifacts
        """
        # Remove sequences of repeated characters (like ........ or @@@@)
        caption = re.sub(r'(.)\1{3,}', r'\1', caption)
        
        # Remove [unused0] type tokens
        caption = re.sub(r'\[\s*unused\d+\s*\]', '', caption)
        
        # Remove extra whitespace
        caption = ' '.join(caption.split())
        
        # Truncate if too long
        if len(caption) > 150:
            caption = caption[:150].rsplit(' ', 1)[0] + '...'
        
        return caption.strip()

    def analyze_depth_b64(self, image_b64: str) -> str:
        """
        Analyze depth in a base64-encoded image for obstacle detection.
        
        Divides the image into regions (center, left, right) and reports
        obstacles based on depth thresholds.
        
        Args:
            image_b64: Base64-encoded JPEG/PNG image data
            
        Returns:
            Description of obstacles detected (e.g., "obstacle ahead, obstacle left")
            or "path clear" if no obstacles
        """
        if not self.depth_enabled:
            return "depth unavailable"
        
        try:
            # Decode image
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Run depth estimation
            inputs = self.depth_processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Normalize depth to 0-1 range
            depth = predicted_depth.squeeze().cpu().numpy()
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max - depth_min > 0:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = depth
            
            # Analyze regions (center, left, right thirds)
            h, w = depth_normalized.shape
            center_region = depth_normalized[h//3:2*h//3, w//3:2*w//3]
            left_region = depth_normalized[h//3:2*h//3, :w//3]
            right_region = depth_normalized[h//3:2*h//3, 2*w//3:]
            
            center_depth = center_region.mean()
            left_depth = left_region.mean()
            right_depth = right_region.mean()
            
            # Lower depth value = closer object
            # Threshold for "close" obstacle (normalized 0-1)
            CLOSE_THRESHOLD = 0.3
            
            obstacles = []
            if center_depth < CLOSE_THRESHOLD:
                obstacles.append("obstacle ahead")
            if left_depth < CLOSE_THRESHOLD:
                obstacles.append("obstacle left")
            if right_depth < CLOSE_THRESHOLD:
                obstacles.append("obstacle right")
            
            return ", ".join(obstacles) if obstacles else "path clear"
            
        except Exception as e:
            logger.error(f"Depth analysis error: {e}")
            return "depth error"

    def process_image(self, image_b64: str) -> dict:
        """
        Process an image and return both caption and depth analysis.
        
        Args:
            image_b64: Base64-encoded JPEG/PNG image data
            
        Returns:
            Dictionary with 'caption' and 'depth' keys
        """
        return {
            "caption": self.caption_b64(image_b64),
            "depth": self.analyze_depth_b64(image_b64)
        }

