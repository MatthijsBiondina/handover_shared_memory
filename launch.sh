#!/bin/bash

# Set up the conda environment by sourcing directly
source ~/anaconda3/etc/profile.d/conda.sh
conda activate airo-mono

# Source the .env file to export environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Environment variables from .env file have been loaded."
else
    echo "No .env file found. Ensure the .env file exists in the script's directory."
fi

# Function to kill all background jobs on Ctrl+C
cleanup() {
    echo "Terminating all background processes..."
    pkill -P $$  # Kills all child processes of this script
    exit 0
}

# Trap Ctrl+C (SIGINT) and run the cleanup function
trap cleanup SIGINT

# Run each Python script as a background process
python -m ur5e.ur5e_hardware &
python -m drake_simulation.drake_server &
python -m curobo_simulation.curobo_planner &
python -m curobo_simulation.curobo_forward_kinematics &
python -m d405.d405_hardware &
#
python -m computer_vision.pointclouds &
python -m computer_vision.yolo_hands &
python -m computer_vision.yolo_labeler &

python -m computer_vision.hands_segmentation &

#
python -m sensor_fusion.yolo_sensor_fusion &
python -m sensor_fusion.hands_sensor_fusion &

python -m decisions.select_target_object &
#
# python -m visualization.d405_plotter &
# python -m visualization.mediapipeposeplotter &
# python -m visualization.yoloplotter &
# python -m visualization.kalmanplotter &
# python -m visualization.target_plotter &

# python -m visualization.sam_plotter &
python -m procedures.approach_object &
python -m optimization.grasp_pose_picker &

# Print message indicating background launch
echo "All scripts are running in the background. Press Ctrl+C to stop."

# Wait indefinitely to keep the script running and allow trap to work
wait
