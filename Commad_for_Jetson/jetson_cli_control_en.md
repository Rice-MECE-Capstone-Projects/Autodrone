# Jetson CLI Control Cheat Sheet (ROS 2 + MAVROS)

> Prerequisite: MAVROS is already running in **another terminal** and connected to the Pixhawk, e.g.:  
> `ros2 launch mavros apm.launch fcu_url:=serial:///dev/ttyACM0:921600 gcs_url:=""`

---

## 2. Check connection and flight state

```bash
source /opt/ros/humble/setup.bash   # Replace with your ROS 2 distro if needed
ros2 topic echo /mavros/state
```

You should see fields such as:

- `connected: true`
- `armed: true/false`
- `mode: STABILIZE / LOITER / AUTO / GUIDED ...`

---

## 3. Change flight mode from Jetson

Switch to **LOITER** mode:

```bash
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{base_mode: 0, custom_mode: 'LOITER'}"
```

Switch to **AUTO** (to execute a mission uploaded from Mission Planner):

```bash
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{base_mode: 0, custom_mode: 'AUTO'}"
```

Switch to **GUIDED** (for future Jetson-controlled navigation):

```bash
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{base_mode: 0, custom_mode: 'GUIDED'}"
```

---

## 4. Arm and disarm the vehicle

You can arm/disarm the vehicle directly from the Jetson.

Arm:

```bash
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
```

Disarm:

```bash
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: false}"
```

After calling these services, you should see the state change in:

- `/mavros/state` output  
- Mission Planner HUD  
- Your TX16S transmitter telemetry page  

---

## 5. Read basic telemetry from Jetson

Read attitude (IMU):

```bash
ros2 topic echo /mavros/imu/data
```

Read local position (useful for future setpoint control):

```bash
ros2 topic echo /mavros/local_position/pose
```

Read battery status:

```bash
ros2 topic echo /mavros/battery
```

---
