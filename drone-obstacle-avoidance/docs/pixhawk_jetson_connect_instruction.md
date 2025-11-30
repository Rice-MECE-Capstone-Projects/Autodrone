# Pixhawk–Jetson Connection Instructions

## 1. Baseline MAVROS Command (Validated)

```
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=serial:///dev/ttyACM0:115200 \
  -p fcu_protocol:=v2.0 \
  -p target_system_id:=1 \
  -p target_component_id:=1
```

- This command connects Jetson to Pixhawk over `/dev/ttyACM0`.
- Verification example:
  ```
  penguin@penguin-desktop:~/capstone/jetson-examples/reComputer/scripts/drone-obstacle-avoidance$ ros2 topic echo /mavros/state --once
  header:
    stamp:
      sec: 1763592916
      nanosec: 995673062
    frame_id: ''
  connected: true
  armed: false
  guided: false
  manual_input: true
  mode: STABILIZE
  system_status: 5
  ---
  ```
- Result: MAVROS can connect successfully.

## 2. Current Issue

Even though Jetson ↔ Pixhawk connectivity works, Mission Planner does not display the expected data in the **Messages** tab.

## 3. Understanding the Three Independent MAVLink Links

1. **Jetson ↔ Pixhawk (direct serial)**

   - When running `python3 avoidance_midas_pro_v2.py --mavlink /dev/ttyACM0`, the script opens `/dev/ttyACM0` and communicates directly.
   - Status sample: `✅ MAVLink connected`.

2. **Jetson ↔ Pixhawk via MAVROS**

   - Running `ros2 run mavros ... fcu_url:=serial:///dev/ttyACM0:115200` also grabs `/dev/ttyACM0`.
   - Only one process can use this serial port at a time.

3. **Mission Planner ↔ Pixhawk**
   - Connected through another USB/TELEM port (e.g., TELEM1).
   - Mission Planner only sees MAVLink traffic on its own port unless Pixhawk forwards data from other ports.

## 4. Root Causes

- `/dev/ttyACM0` cannot be shared between MAVROS and `avoidance_midas_pro_v2.py`. Use one at a time.
- Mission Planner does not automatically receive MAVLink packets from Jetson’s port. Pixhawk must forward them, or Jetson must send a second stream directly to Mission Planner (UDP).

## 5. Recommended Workflow (Direct Serial + Pixhawk Forwarding)

### Step 1 — Ensure Exclusive Serial Access

Only one of the following should run:

- `ros2 run mavros mavros_node ...`
- `python3 avoidance_midas_pro_v2.py --mavlink /dev/ttyACM0`

During obstacle avoidance flights, run only `avoidance_midas_pro_v2.py` on `/dev/ttyACM0`.

### Step 2 — Configure Pixhawk Serial Port for Jetson

1. In Mission Planner: `CONFIG` → `Full Parameter List`.
2. Identify which `SERIALx` corresponds to Jetson’s USB connection (likely `SERIAL0`):
   - `SERIALx_PROTOCOL = 2` (MAVLink2).
   - `SERIALx_BAUD = 115` (115200 baud).
3. Enable forwarding of that port’s traffic to GCS:
   - Either enable the **GCS Forward** bit in `BRD_OPTIONS`.
   - Or set the **Forward to GCS** bit within `SERIALx_OPTIONS`.

### Step 3 — Observe Messages in Mission Planner

- Open `Data` → `Messages` to view MAVLink packets such as `SET_POSITION_TARGET_LOCAL_NED`.
- On the `Status` page, monitor `mode`, `guided`, `vx`, `vy`, etc., while the avoidance script runs.
- If the Messages pane appears filtered, use `Ctrl+F` → **MAVLink Inspector** to inspect `SET_POSITION_TARGET_LOCAL_NED` frequency.

## 6. Alternative Validation Path (Direct UDP to Mission Planner)

Use this to verify that Mission Planner can display Jetson packets, independent of Pixhawk forwarding.

### Mission Planner

1. Choose **UDP** in the connection dropdown.
2. Leave the port at `14550`.
3. Click **Connect**, confirm `14550` when prompted.

### PC IP Discovery (Example on Windows)

```
ipconfig
# Locate the IPv4 address reachable by Jetson, e.g., 192.168.1.100.
```

### Jetson Command

```
cd /home/penguin/capstone/jetson-examples/reComputer/scripts/drone-obstacle-avoidance

python3 avoidance_midas_pro_v2.py \
  --method affine_invariant \
  --camera 0 \
  --no-realsense \
  --mavlink udp:192.168.1.100:14550
```

- Replace `192.168.1.100` with the Mission Planner host IP.
- Expected terminal log: `🔌 Connecting MAVLink: udp:192.168.1.100:14550... ✅ MAVLink connected`.
- Mission Planner’s `Messages` tab will immediately show Jetson’s `HEARTBEAT` and `SET_POSITION_TARGET_LOCAL_NED`.

After confirming visibility via UDP, revert to the serial workflow and configure Pixhawk forwarding.

## 7. Dual-Link Reminder

- MAVROS and `avoidance_midas_pro_v2.py` cannot run concurrently on `/dev/ttyACM0`.
- If you later prefer a ROS2-based pipeline, `avoidance_midas_pro_v2.py` must be modified to publish ROS topics instead of sending MAVLink directly.

## 8. Required User Actions

1. Use only one program at a time on `/dev/ttyACM0`.
2. Configure Pixhawk’s `SERIALx_*` parameters for Jetson and enable forwarding (via `BRD_OPTIONS` or `SERIALx_OPTIONS`).
3. Reboot Pixhawk and rerun `avoidance_midas_pro_v2.py`.
4. Verify Mission Planner displays `SET_POSITION_TARGET_LOCAL_NED` in **Messages** or **MAVLink Inspector**.
5. If unsure which `SERIALx` is Jetson’s port, capture `SERIAL0…SERIAL5` parameter values and review them (or share them for assistance).

## 9. Mission Planner UDP Setup — Quick Reference

1. Mission Planner: select **UDP**, port `14550`, click **Connect**, confirm.
2. PC IP (example): `192.168.1.100`.
3. Jetson command:
   ```
   python3 avoidance_midas_pro_v2.py \
     --method affine_invariant \
     --camera 0 \
     --no-realsense \
     --mavlink udp:192.168.1.100:14550
   ```
4. Inspect `Data → Messages` for Jetson packets.

## 10. Conclusion

- Direct serial (`/dev/ttyACM0`) works for both MAVROS and `avoidance_midas_pro_v2.py`, but exclusively.
- Mission Planner requires either Pixhawk forwarding or a separate UDP feed to observe Jetson’s traffic.
- Enabling the appropriate `SERIALx_OPTIONS`/`BRD_OPTIONS` bit or temporarily using UDP ensures Mission Planner displays Jetson-originated MAVLink messages without altering core functionality.
