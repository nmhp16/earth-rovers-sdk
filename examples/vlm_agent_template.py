import asyncio
import aiohttp
import base64
import json
import logging
# import cv2 # Optional: for visualizing if you want
# import numpy as np # Optional

# Configuration
SDK_URL = "http://localhost:8000"
CONTROL_ENDPOINT = f"{SDK_URL}/control"
VISION_ENDPOINT = f"{SDK_URL}/v2/screenshot" # Returns front, rear, and timestamp
DATA_ENDPOINT = f"{SDK_URL}/data"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def get_bot_state(session):
    """Fetches images and telemetry from the SDK."""
    try:
        # Get Telemetry
        async with session.get(DATA_ENDPOINT) as resp:
            data = await resp.json()
        
        # Get Vision
        async with session.get(VISION_ENDPOINT) as resp:
            vision = await resp.json()
            
        return data, vision
    except Exception as e:
        logger.error(f"Error fetching state: {e}")
        return None, None

async def send_action(session, linear_speed, angular_speed, lamp=0):
    """Sends a control command to the bot."""
    payload = {
        "command": {
            "linear": linear_speed,   # -1.0 to 1.0
            "angular": angular_speed, # -1.0 to 1.0 (Left/Right)
            "lamp": lamp              # 0 or 1
        }
    }
    
    try:
        async with session.post(CONTROL_ENDPOINT, json=payload) as resp:
            if resp.status != 200:
                logger.error(f"Failed to send command: {await resp.text()}")
    except Exception as e:
        logger.error(f"Error sending action: {e}")

async def run_vlm_agent():
    async with aiohttp.ClientSession() as session:
        logger.info("Starting VLM Agent Loop...")
        
        while True:
            # 1. PERCEPTION
            telemetry, vision = await get_bot_state(session)
            
            if telemetry and vision:
                # front_b64 = vision.get("front_frame")
                # rear_b64 = vision.get("rear_frame")
                
                # --- VLM LOGIC GOES HERE ---
                # Example:
                # image = decode_base64(front_b64)
                # prompt = "You are a rover. Avoid obstacles. What is your next move?"
                # action = MyVLM.predict(image, prompt, telemetry)
                
                logger.info(f"Battery: {telemetry.get('battery')}% | GPS: {telemetry.get('latitude')}, {telemetry.get('longitude')}")
                
                # 2. DECISION (Mock AI logic)
                # Just moving forward slowly as an example
                linear = 0.0 
                angular = 0.0
                
                # 3. ACTION
                await send_action(session, linear, angular)
                
            # Rate limit (e.g., 10Hz)
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(run_vlm_agent())
    except KeyboardInterrupt:
        logger.info("Agent stopped.")
