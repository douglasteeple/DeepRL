#!/bin/bash
######################################################
# Launch gazebo-arm in an xterm window
# Plus a little song and dance to get both 
# a plot file (gazebo-arm.plt)
# and a log file (gazebo-arm.log)
######################################################
xterm -e "cd $SD/RoboND-DeepRL-Project/build/aarch64/bin; (./gazebo-arm.sh | tee ../../../gazebo-arm.log) 3>&1 1>&2 2>&3 | tee ../../../gazebo-arm.plt"
