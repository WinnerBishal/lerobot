
"""
Helper script to probe, validate, and benchmark Network (ZMQ) cameras.

Since network cameras cannot be auto-discovered like USB devices, this script
connects to a specified host/port to:
1. Verify the connection and stream integrity.
2. Measure the actual incoming FPS (crucial for setting dataset.fps).
3. Save a snapshot to verify color/depth decoding.

Example:
    # Check default color stream (localhost:5556)
    python -m lerobot.scripts.lerobot_find_netcam --mode color

    # Check depth stream on specific port
    python -m lerobot.scripts.lerobot_find_netcam --mode depth --port 5555

    # Check custom stream
    python -m lerobot.scripts.lerobot_find_netcam --mode custom --port 9999 --width 640 --height 480 --channels 3
"""

import argparse
import logging
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from lerobot.cameras.network_camera.network_camera import NetworkCamera
from lerobot.cameras.network_camera.configuration_network_camera import NetworkCameraConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def probe_network_camera(args):
    """Connects to a network camera, measures FPS, and saves a snapshot."""
    
    # 1. Setup Configuration
    logger.info(f"Probing Network Camera at {args.host}:{args.port} (Mode: {args.mode})...")
    
    try:
        config = NetworkCameraConfig(
            mode=args.mode,
            host=args.host,
            port=args.port,
            width=args.width,
            height=args.height,
            channels=args.channels,
            dtype_str=args.dtype
        )
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        return

    camera = NetworkCamera(config)

    # 2. Attempt Connection
    try:
        camera.connect()
        logger.info("✅ Connection Successful!")
    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        logger.error("Ensure the ZMQ sender (ROS2 node) is running and the host/port are correct.")
        return

    # 3. Measure FPS
    logger.info("Measuring actual stream FPS (sampling for 2 seconds)...")
    
    # Get the backend capture object directly
    if not hasattr(camera, 'backend') or camera.backend is None:
        logger.error("Camera backend not initialized.")
        return

    start_count = camera.backend.frame_counter
    start_time = time.perf_counter()
    duration = 2.0
    
    # We sleep to let the background thread do its work
    time.sleep(duration)
    
    end_count = camera.backend.frame_counter
    elapsed = time.perf_counter() - start_time
    
    frames_received = end_count - start_count
    actual_fps = frames_received / elapsed
    
    logger.info(f"Measured FPS: {actual_fps:.2f}")
    logger.info(f"Recommended --dataset.fps: {int(actual_fps)}")


    # 4. Save Snapshot
    last_frame = camera.read()
    if last_frame is not None:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"netcam_{args.mode}_{args.host}_{args.port}.png"
        filepath = output_dir / filename

        # Handle Depth (16-bit) vs Color (8-bit)
        if config.channels == 1:
            # Normalize 16-bit depth for visualization (0-65535 -> 0-255)
            # This is just for preview; raw data is preserved in dataset
            depth_vis = cv2.normalize(last_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = np.uint8(depth_vis)
            Image.fromarray(depth_vis.squeeze()).save(filepath)
            logger.info(f"Depth snapshot saved to: {filepath} (Normalized for display)")
        else:
            # Color is RGB
            Image.fromarray(last_frame).save(filepath)
            logger.info(f"Color snapshot saved to: {filepath}")
    else:
        logger.warning("⚠️  No frames received to save.")

    # 5. Cleanup
    camera.disconnect()

def main():
    parser = argparse.ArgumentParser(
        description="Utility to find and verify LeRobot Network Cameras (ZMQ)."
    )

    # Connection Args
    parser.add_argument("--host", type=str, default="localhost", help="ZMQ Host IP")
    parser.add_argument("--port", type=int, default=None, help="ZMQ Port (Overrides mode default)")
    parser.add_argument("--mode", type=str, default="color", choices=["color", "depth", "custom"], 
                        help="Preset mode. 'color' defaults to port 5556, 'depth' to 5555.")

    # Custom Args (Only needed if mode=custom)
    parser.add_argument("--width", type=int, default=None, help="Frame Width")
    parser.add_argument("--height", type=int, default=None, help="Frame Height")
    parser.add_argument("--channels", type=int, default=None, help="Channels (3 for RGB, 1 for Depth)")
    parser.add_argument("--dtype", type=str, default=None, help="Numpy dtype string (e.g., 'uint8', 'uint16')")

    # Output
    parser.add_argument("--output-dir", type=Path, default="outputs/captured_images", help="Snapshot save location")

    args = parser.parse_args()
    probe_network_camera(args)

if __name__ == "__main__":
    main()