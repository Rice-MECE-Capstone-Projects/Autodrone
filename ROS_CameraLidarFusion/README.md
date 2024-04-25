## :robot: ROS (Robot Operating System): Camera-Lidar Fusion Application
This project requires a web camera, RPLIDAR A1, and Pixhawk Hex Cube Black.

**RPLIDAR A1**
<img src="https://www.slamtec.ai/wp-content/uploads/2023/11/rplidar-a1-summery-4.jpg"  />

**Pixhawk Hex Cube Black**
<img src="https://docs.px4.io/main/assets/cube_black_hero.BuuY9D1m.png"  />



### ROS Neotic installation instruction from offical website
The link can be found here: 
https://wiki.ros.org/noetic/Installation/Ubuntu

**If you are using the Ubuntu version newer than 20.04, you have to install ROS noetic in a virtual environment, follow the instruction here:** https://robostack.github.io/GettingStarted.html

Follow the steps to compelete the installation.
Then, source the Installation by:
```
source /opt/ros/noetic/setup.bash
```


Clone the ROS workspace to local workspace:
```
cd ~
git clone https://github.com/Rice-MECE-Capstone-Projects/Autodrone.git
cp -a ~/Autodrone/ROS_CameraLidarFusion/autodrone_ws ~/
```

Create and build a ROS workspace with you preferred name:
```
cd ~/autodrone_ws/
catkin_make
```
Wait for the make to complete.

Source the workspace. **Do this everytime you open a new terminal, or add this line to .bashrc file to make it permanently effective.**
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


### Launch 3D-Reconstruction
```
roslaunch lidar_3d_test fusion.launch lidar_sphere.launch
```

### Rosbag Data Capture
rosbag is a file format in ROS used for recording and playing back messages. These files are essentially archives that store data published over ROS topics, including sensor data, robot status information, or environmental data. rosbag allows developers to capture data streams during operation and replay them later for testing or analysis, facilitating easier debugging and system improvement without requiring real-time access to a running robot.

```
roslaunch fly_test fusion.launch rosbag_collect.launch
```
After the publishing node launched, run folowwing command to capture both video and lidar points.
'''
rosrun fly_test video_collect.py
'''
Then, run the following command to save "ROS format" video to MP4 file. 
```
rosrun fly_test rosbag_record.py
```

### Video Stream
Grab your local IP adress:
```
ip addr
```
Replace the ip with the IP you found, run /video_Stream/Server.py on your host PC.
```
python Server.py
```
Run /video_Stream/Client.py on your recipient PC.
```
python Client.py 
```
-----------------------------------------------------------------------------------------------
## License
This project is licensed under Electrical and Computer Engineering Department at Rice University

<img src="https://riceconnect.rice.edu/image/engineering/ece/SOE-ECE-Rice-logo-stacked.jpg" width="500" height="140" />
