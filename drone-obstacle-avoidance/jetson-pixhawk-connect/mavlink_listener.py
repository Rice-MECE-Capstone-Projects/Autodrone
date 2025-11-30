#!/usr/bin/env python3
from datetime import datetime
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink

def main():
    link = 'udp:127.0.0.1:14555'   # Can be changed to 'udp:0.0.0.0:14555'
    m = mavutil.mavlink_connection(link)
    # Key: send one packet first so MAVROS records your return address
    m.mav.heartbeat_send(mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    print(f"[listener] sent initial GCS HEARTBEAT, listening on {link} ...")

    while True:
        msg = m.recv_match(blocking=True)
        if not msg:
            continue
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        sysid, compid = msg.get_srcSystem(), msg.get_srcComponent()
        d = msg.to_dict() if hasattr(msg, "to_dict") else {}
        print(f"[{ts}] {sysid}.{compid} {msg.get_type()} {d}")

if __name__ == "__main__":
    main()