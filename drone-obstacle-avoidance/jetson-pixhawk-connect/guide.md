# Pixhawk–Jetson MAVROS Command Reference

This document consolidates the default MAVROS launch command, common mode switches, telemetry stream settings, and key ROS 2 topics for monitoring Pixhawk outputs from Jetson. All original commands are preserved, clarified in English, and organized for quick reference. Additional validation and safety notes are included at the end.

---

## 1. Launch MAVROS Node (Validated Baseline)

```
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=serial:///dev/ttyACM0:115200 \
  -p fcu_protocol:=v2.0 \
  -p target_system_id:=3 \
  -p target_component_id:=1
```

- Opens `/dev/ttyACM0` at 115200 baud and negotiates MAVLink v2.0.
- `target_system_id` / `target_component_id` should match Pixhawk’s IDs (default system: 1; adjust if needed).
- Verify connectivity via topics such as `/mavros/state`.

---

## 2. Flight Mode Management (ROS 2 Services)

All mode changes use the `mavros_msgs/srv/SetMode` service:

### Outdoor / GPS-Assisted Flight

```
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
"{base_mode: 0, custom_mode: 'GUIDED'}"
```

### Indoor / No-GPS Guidance

```
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
"{base_mode: 0, custom_mode: 'GUIDED_NOGPS'}"
```

### Other Common Modes (Invoke as Required)

```
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
"{base_mode: 0, custom_mode: 'LOITER'}"

ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
"{base_mode: 0, custom_mode: 'ALT_HOLD'}"

ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
"{base_mode: 0, custom_mode: 'LAND'}"
```

---

## 3. Telemetry Stream Rate Configuration

Use `mavros_msgs/srv/MessageInterval` to configure message rates (Hz):

### SERVO_OUTPUT_RAW (Motor / Servo PWM, Message ID 36)

```
ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
"{message_id: 36, message_rate: 10.0}"
```

### VFR_HUD (Throttle Percentage, Message ID 74)

```
ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
"{message_id: 74, message_rate: 5.0}"
```

### STATUSTEXT (Event-Driven, Message ID 253)

```
ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
"{message_id: 253, message_rate: 2.0}"
```

> STATUSTEXT messages are event-triggered; rejection of a fixed interval is normal.

---

## 4. Real-Time Topic Monitoring

### Motor / Servo PWM Outputs

```
ros2 topic echo /mavros/rc/out \
  --qos-reliability reliable \
  --qos-durability volatile \
  --qos-history keep_last \
  --qos-depth 50
```

### Throttle Percentage and Airspeed (VFR_HUD)

```
ros2 topic echo /mavros/vfr_hud \
  --qos-reliability reliable \
  --qos-durability volatile \
  --qos-history keep_last \
  --qos-depth 50
```

### Status Text (Event Messages)

```
ros2 topic echo /mavros/statustext/recv \
  --qos-reliability best_effort \
  --qos-durability volatile \
  --qos-history keep_last \
  --qos-depth 50
```

> Lack of output simply means no new STATUSTEXT events occurred.

---

## 5. Arming and Disarming

### Arm (Unlock Motors)

```
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
```

### Disarm (Lock Motors)

```
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: false}"
```

---

## 6. Supplemental Guidance and Validation

1. **Connection Verification**

   - Before arming, confirm `/mavros/state` reports `connected: true`, `armed: false`, `guided: true/false` as expected.
   - Use `ros2 topic hz /mavros/rc/out` to ensure PWM data streams at the configured rate.

2. **Safety Interlocks**

   - Always disarm before changing baud-rate or protocol parameters.
   - If mode switches fail, check whether the vehicle is already armed or in a failsafe state.

3. **Logging for Post-Flight Analysis**

   - Use `ros2 bag record /mavros/rc/out /mavros/vfr_hud /mavros/statustext/recv` to capture actuator signals and throttle data for diagnostics.

4. **Multi-Client Access**

   - `/dev/ttyACM0` must be dedicated to a single process. Stop MAVROS before starting custom scripts (e.g., `avoidance_midas_pro_v2.py`) to avoid port contention.

5. **Mission Planner Cross-Check**

   - When running MAVROS over serial, Mission Planner can connect via a separate USB/TELEM port or UDP feed to validate that commands propagate correctly.

6. **Mode Transition Checklist**
   - Confirm GPS lock status before entering `GUIDED`.
   - For indoor tests, ensure EKF is configured to allow `GUIDED_NOGPS`.
   - Monitor STATUSTEXT for warnings such as `EKF variance` or `GPS glitch`.

---

By following this structured reference, you can reliably launch MAVROS, issue flight mode transitions, configure telemetry intervals, observe actuator outputs, and safely arm or disarm the vehicle while maintaining clear operational visibility.
