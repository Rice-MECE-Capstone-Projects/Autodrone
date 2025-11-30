#!/usr/bin/env python3
import argparse, sys, time
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink

AP_COPTER_MODES = {"STABILIZE":0,"ACRO":1,"ALT_HOLD":2,"AUTO":3,"GUIDED":4,"LOITER":5,
                   "RTL":6,"CIRCLE":7,"POSITION":8,"LAND":9,"OF_LOITER":10,"DRIFT":11,
                   "SPORT":13,"FLIP":14,"AUTOTUNE":15,"POSHOLD":16,"BRAKE":17,"THROW":18,
                   "AVOID_ADSB":19,"GUIDED_NOGPS":20,"SMART_RTL":21}

def wait_heartbeat(conn, timeout=10):
    print("[send] waiting for HEARTBEAT ...")
    msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout)
    if not msg:
        print("[send] timeout: no HEARTBEAT"); sys.exit(1)
    print(f"[send] heartbeat from sys={msg.get_srcSystem()} comp={msg.get_srcComponent()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", default="udpout:127.0.0.1:14555", help="udp:IP:PORT (do not use udpout:)")
    ap.add_argument("--src-sys", type=int, default=250)
    ap.add_argument("--src-comp", type=int, default=190)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("statustext"); sp.add_argument("text")
    sp = sub.add_parser("request-message"); sp.add_argument("name")
    sub.add_parser("arm"); sub.add_parser("disarm")
    sp = sub.add_parser("set-mode"); sp.add_argument("mode")
    sp = sub.add_parser("takeoff"); sp.add_argument("alt", type=float)
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.link, source_system=args.src_sys, source_component=args.src_comp)

    # Key: "say hello" first, then wait for heartbeat
    m.mav.heartbeat_send(mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    wait_heartbeat(m)

    if args.cmd == "statustext":
        text = args.text.encode("utf-8")[:50]
        m.mav.statustext_send(mavlink.MAV_SEVERITY_INFO, text.ljust(50, b"\x00"))
        print("[send] STATUSTEXT sent")

    elif args.cmd == "request-message":
        name = args.name.upper()
        msg_id = getattr(mavlink, f"MAVLINK_MSG_ID_{name}", None)
        if msg_id is None: print(f"[send] unknown message: {name}"); sys.exit(2)
        m.mav.command_long_send(args.src_sys, args.src_comp, mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                                msg_id, 0,0,0,0,0,0)
        print(f"[send] requested {name} ({msg_id})")

    elif args.cmd == "arm":
        m.mav.command_long_send(args.src_sys, args.src_comp, mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                1,0,0,0,0,0,0); print("[send] arm sent")

    elif args.cmd == "disarm":
        m.mav.command_long_send(args.src_sys, args.src_comp, mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                0,0,0,0,0,0,0); print("[send] disarm sent")

    elif args.cmd == "set-mode":
        mode = args.mode.upper()
        if mode not in AP_COPTER_MODES:
            print(f"[send] unknown mode '{mode}'"); sys.exit(3)
        m.mav.set_mode_send(target_system=m.target_system or 1,
                            base_mode=mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                            custom_mode=AP_COPTER_MODES[mode])
        print(f"[send] set-mode {mode}")

    elif args.cmd == "takeoff":
        alt = float(args.alt)
        m.mav.command_long_send(args.src_sys, args.src_comp, mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                                0,0,0,0,0,0,alt)
        print(f"[send] takeoff {alt}m")

    # Read several responses
    time.sleep(0.2)
    for _ in range(10):
        msg = m.recv_match(blocking=True, timeout=0.5)
        if msg: print("[recv]", msg)

if __name__ == "__main__":
    main()