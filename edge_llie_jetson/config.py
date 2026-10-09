"""
EdgeLLIE - Configuration for Low-Light Image Enhancement on NVIDIA Jetson Orin Nano
"""
import os
from pydantic import BaseModel

class AppConfig(BaseModel):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Target Hardware specifications (NVIDIA Jetson Orin Nano)
    DEVICE_NAME: str = "NVIDIA Jetson Orin Nano"
    TARGET_RAM_GB: float = 8.0
    POWER_MODE: str = "15W"
    
    # Video & Camera settings
    # VIDEO_SOURCE: 'camera', 'sample', 'file', or 'sim'
    VIDEO_SOURCE: str = os.getenv("VIDEO_SOURCE", "sample")
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    CSI_SENSOR_ID: int = int(os.getenv("CSI_SENSOR_ID", "0"))
    FRAME_WIDTH: int = 256
    FRAME_HEIGHT: int = 256
    FPS: int = 30
    
    # Enhancement defaults (matching CIDNet & HVI parameters)
    DEFAULT_GAMMA: float = 1.0       # Gamma curve (0.1 - 3.0)
    DEFAULT_ALPHA_S: float = 1.0     # Saturation adjustment (0.0 - 2.0)
    DEFAULT_ALPHA_I: float = 1.0     # Intensity / Brightness boost (0.1 - 2.5)
    VIEW_MODE: str = "enhanced"      # 'enhanced', 'split', 'original'
    
    # Telemetry streaming rate (Hz)
    TELEMETRY_FPS: int = 10
    
    # Paths
    SAMPLE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "LOL", "LOLv1", "test", "low")

config = AppConfig()
