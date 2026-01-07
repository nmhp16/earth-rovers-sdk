import base64
import io
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import logger, BLIP_MODEL_NAME, BLIP_MAX_NEW_TOKENS

class VisionSystem:
    def __init__(self):
        logger.info("Loading BLIP...")
        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"BLIP loaded on {self.device}")

    def caption_b64(self, image_b64: str) -> str:
        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=BLIP_MAX_NEW_TOKENS)
            cap = self.processor.decode(out[0], skip_special_tokens=True).strip()
            return cap
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return "camera malfunction"
