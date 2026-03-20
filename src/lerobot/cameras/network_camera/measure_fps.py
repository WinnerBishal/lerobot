import time
import zmq
import numpy as np

# --- CONFIGURATION ---
HOST = "localhost"
PORT = 5555        # Change to 5557 for depth
WIDTH = 480 #1280       # Change to 480 for depth
HEIGHT = 270 #720       # Change to 270 for depth
CHANNELS = 1 #3       # Change to 1 for depth
DTYPE = "uint16" #"uint8"    # Change to "uint16" for depth
# ---------------------

def measure_fps():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # Use CONFLATE to ignore old buffered frames and get the real-time rate
    socket.setsockopt(zmq.CONFLATE, 1)
    
    url = f"tcp://{HOST}:{PORT}"
    print(f"Connecting to {url}...")
    socket.connect(url)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # Calculate expected frame size
    dtype = np.dtype(DTYPE)
    expected_bytes = WIDTH * HEIGHT * CHANNELS * dtype.itemsize
    
    print(f"Expecting {expected_bytes} bytes per frame.")
    print("Measuring FPS... (Press Ctrl+C to stop)")

    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # This blocks until a frame arrives
            data = socket.recv()
            
            if len(data) == expected_bytes:
                frame_count += 1
            else:
                print(f"Warning: Received partial frame ({len(data)} bytes)")

            # Update FPS display every 2 seconds
            elapsed = time.time() - start_time
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(f"Incoming FPS: {fps:.2f}")
                
                # Reset counter
                frame_count = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    measure_fps()