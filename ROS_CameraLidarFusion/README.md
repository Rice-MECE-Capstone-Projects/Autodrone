## :robot: ROS (Robot Operating System)

### ROS Neotic installation instruction from offical website
The link can be found here: 
https://wiki.ros.org/noetic/Installation/Ubuntu

Follow the steps to compelete the installation.
Then, source the Installation by:
```
source /opt/ros/noetic/setup.bash
```


Clone the ROS workspace to local workspace:
```
cd ~
git clone https://github.com/Rice-MECE-Capstone-Projects/Autodrone.git
cp -a ~/Autodrone/Drone_Function/autodrone_folder/ROS_Drone/. ~/
```

Create and build a ROS workspace with you preferred name:
```
cd ~/autodrone_ws/
catkin_make
```
Wait for the make to complete.

Source the workspace. You have to do this everytime you open a new terminal, or add this line to .bashrc file to make it permanently effective.
```
source devel/setup.bash
```

To make sure your workspace is properly overlayed by the setup script, make sure ROS_PACKAGE_PATH environment variable includes the directory you're in.
```
echo $ROS_PACKAGE_PATH
/home/youruser/autodrone_ws/src:/opt/ros/kinetic/share
```

Before running or launching any files, run:
```
roscore
```

### Bring up camera for QR code tracking
```
roscd
roslaunch camera_test camera_trackqr.launch
```

### Bring up camera and LiDAR
Check the port that device connected.
```
roscd
../dev_path.sh
sudo chmod 666 /dev/<port_name>
```

Launch the camera-LiDAR fusion
```
roslaunch fusion_sensor fusion.launch
```

-----------------------------------------------------------------------------------------------
## License
This project is licensed under Electrical and Computer Engineering Department at Rice University

<img src="https://riceconnect.rice.edu/image/engineering/ece/SOE-ECE-Rice-logo-stacked.jpg" width="500" height="140" />
