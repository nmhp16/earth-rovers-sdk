import os
import time
from pyppeteer import launch
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables with defaults
FORMAT = os.getenv("IMAGE_FORMAT", "png")
QUALITY = float(os.getenv("IMAGE_QUALITY", "1.0"))
HAS_REAR_CAMERA = os.getenv("HAS_REAR_CAMERA", "False").lower() == "true"

if FORMAT not in ["png", "jpeg", "webp"]:
    raise ValueError("Invalid image format. Supported formats: png, jpeg, webp")

if QUALITY < 0 or QUALITY > 1:
    raise ValueError("Invalid image quality. Quality should be between 0 and 1")


class BrowserService:
    def __init__(self):
        self.browser = None
        self.page = None
        self.default_viewport = {"width": 3840, "height": 2160}
        self._basic_init_done = False
        self._video_init_done = False

    async def initialize_browser_basic(self):
        if self._basic_init_done:
            return
        
        try:
            executable_path = os.getenv(
                "CHROME_EXECUTABLE_PATH",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            self.browser = await launch(
                executablePath=executable_path,
                headless=True,
                args=[
                    "--ignore-certificate-errors",
                    "--no-sandbox",
                    f"--window-size={self.default_viewport['width']},{self.default_viewport['height']}",
                ],
            )
            self.page = await self.browser.newPage()
            await self.page.setViewport(self.default_viewport)
            await self.page.setExtraHTTPHeaders(
                {"Accept-Language": "en-US,en;q=0.9"}
            )
            await self.page.goto(
                "http://127.0.0.1:8000/sdk", {"waitUntil": "networkidle2"}
            )
            await self.page.click("#join")
            
            # Wait for RTM connection (map element indicates RTM is working)
            await self.page.waitForSelector("#map", {"timeout": 15000})
            
            # Give RTM a moment to receive initial data
            await self.page.waitFor(2000)
            
            self._basic_init_done = True
            print("Browser basic init complete (RTM connected)")
        except Exception as e:
            print(f"Error in basic browser init: {e}")
            if self.browser:
                await self.browser.close()
                self.browser = None
                self.page = None
            self._basic_init_done = False
            raise

    async def initialize_browser(self):
        """Full initialization including video stream - needed for frames/screenshots."""
        # First do basic init if not done
        if not self._basic_init_done:
            await self.initialize_browser_basic()
        
        # Then wait for video if not done
        if not self._video_init_done:
            try:
                print("Waiting for video stream...")
                await self.page.waitForSelector("video", {"timeout": 30000})
                await self.page.setViewport(self.default_viewport)

                call = f"""() => {{
                    window.initializeImageParams({{
                        imageFormat: "{FORMAT}",
                        imageQuality: {QUALITY}
                    }});
                }}"""
                await self.page.evaluate(call)
                
                self._video_init_done = True
                print("Browser full init complete (video connected)")
            except Exception as e:
                print(f"Error waiting for video: {e}")
                raise

    async def take_screenshot(self, video_output_folder: str, elements: list):
        await self.initialize_browser()

        dimensions = await self.page.evaluate(
            """() => {
            return {
                width: Math.max(document.documentElement.scrollWidth, window.innerWidth),
                height: Math.max(document.documentElement.scrollHeight, window.innerHeight),
            }
        }"""
        )

        if (
            dimensions["width"] > self.default_viewport["width"]
            or dimensions["height"] > self.default_viewport["height"]
        ):
            await self.page.setViewport(dimensions)

        element_map = {"front": "#player-1000", "rear": "#player-1001", "map": "#map"}

        screenshots = {}
        for name in elements:
            if name in element_map:
                element_id = element_map[name]
                output_path = f"{video_output_folder}/{name}.png"
                element = await self.page.querySelector(element_id)
                if element:
                    start_time = time.time()  # Start time
                    await element.screenshot({"path": output_path})
                    end_time = time.time()  # End time
                    elapsed_time = (
                        end_time - start_time
                    ) * 1000  # Convert to milliseconds
                    print(f"Screenshot for {name} took {elapsed_time:.2f} ms")
                    screenshots[name] = output_path
                else:
                    print(f"Element {element_id} not found")
            else:
                print(f"Invalid element name: {name}")

        return screenshots

    async def data(self) -> dict:
        await self.initialize_browser_basic()

        bot_data = await self.page.evaluate(
            """() => {
        return window.rtm_data;
        }"""
        )

        return bot_data

    async def front(self) -> str:
        await self.initialize_browser()

        front_frame = await self.page.evaluate(
            """() => {
        return getLastBase64Frame(1000) || null;
        }"""
        )

        return front_frame

    async def rear(self) -> str:
        await self.initialize_browser()

        rear_frame = await self.page.evaluate(
            """() => {
        return getLastBase64Frame(1001) || null;
        }"""
        )

        return rear_frame

    async def send_message(self, message: dict):
        await self.initialize_browser()

        await self.page.evaluate(
            """(message) => {
                window.sendMessage(message);
            }""",
            message,
        )

    async def close_browser(self):
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None
