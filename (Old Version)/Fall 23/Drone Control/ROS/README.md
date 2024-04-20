# Software in the Loop (SITL) Simulation Instrcution
### Launch Gazebo environment:
```
roslaunch drone_control gazebo_env.launch
```
To launch a different world configuration, users can edit the path to the file under the world folder in the gazebo_env.launch file.

### Run Ardupilot SITL program:
Run the bash script located in the scripts folder.
```
./scripts/startsitl.sh
```
This process simulates the Fight Control Unit (FCU) for the virtual drone and setup MAVProxy for the FCU.

### Estiblish the connection between the FCU and MAVROS node through MAVLink protocol:
```
roslaunch drone_control apm.launch
``` 
Now, users can control the drone and monitor its status using ROS topics and services.

### Run ROS programs in C++ or Python
For C++ programs:
* Declare C++ executables in CMakeLists.txt
    ```
    add_executable(${PROJECT_NAME}_node src/drone_control_node.cpp)
    ```
* Build executable files
    ```
    catkin_make
    ```
Run ROS programs
```
rosrun drone_control executable_file
```