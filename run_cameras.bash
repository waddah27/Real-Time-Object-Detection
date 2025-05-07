#!/bin/bash

# Configuration
RTSP_BASE="rtsp://colpino:colpino374@5.17.92.56:554/cam/realmonitor"
PYTHON_SCRIPT="multi_cameras.py"
VENV_PATH="./myenv"  # Path to virtual environment

# Function to start camera with virtual environment
start_camera() {
    local camera_id=$1
    local rtsp_url="${RTSP_BASE}?channel=${camera_id}&subtype=0"
    
    # Activate virtual environment and run script
    source "${VENV_PATH}/bin/activate" && \
    python3 "$PYTHON_SCRIPT" --rtsp_url "$rtsp_url" --camera_id "$camera_id"
    
    # If the process exits, it will restart automatically
    echo "Camera ${camera_id} process exited, restarting in 5 seconds..."
    sleep 5 #TODO maybe less or more ? 
    start_camera "$camera_id"
}

# Start all camera processes in background
for i in {1..4}; do
    start_camera $i &
    echo "Started camera $i with PID $!"
done

# Wait for all processes to complete (they shouldn't unless error occurs)
echo "All cameras started. Press Ctrl+C to stop all processes."
wait

# Cleanup on exit
trap "echo 'Stopping all cameras...'; pkill -f $PYTHON_SCRIPT; exit" SIGINT SIGTERM