#!/bin/bash

csv_file=/capstone/ros/src/twist_controller/throttles.csv
#csv_file=/capstone/ros/src/twist_controller/steers.csv
#csv_file=/capstone/ros/src/twist_controller/brakes.csv

cp $csv_file $csv_file.bak

echo "------ md5sum before ----------"
md5sum "$csv_file"
echo "-------------------------------"

roslaunch /capstone/ros/src/twist_controller/launch/dbw_test.launch

echo "------ md5sum after ----------"
md5sum "$csv_file"
echo "------------------------------"

diff $csv_file $csv_file.bak
python /capstone/scripts/error.py $csv_file
