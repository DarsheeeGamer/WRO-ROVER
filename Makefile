# Makefile for WRO-ROVER project

.PHONY: all calibrate_arm robot_brain clean

all: calibrate_arm robot_brain

calibrate_arm:
	@echo "Running arm camera calibration..."
	python calibration/calibrate_arm.py

robot_brain:
	@echo "Starting robot brain..."
	python robot_control/robot_brain.py

clean:
	@echo "Cleaning up generated files..."
	rm -f stereo_cal_arm.npz
	# Add other cleanup commands here if needed