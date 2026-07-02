#!/usr/bin/env python3
"""
LeRobot Standalone Local Leader Arm Agent
----------------------------------------
Runs on the local machine (Windows/macOS) to read raw positions and calibration
from bimanual SO101 leader arms (Feetech STS3215 servos) and stream them over
UDP to the remote host.

Requirements:
    pip install pyserial scservo-sdk

Usage:
    python local_leader_agent.py --remote-ip <REMOTE_UBUNTU_IP> --left-port COM3 --right-port COM4
"""

import sys
import time
import socket
import json
import argparse

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Error: 'pyserial' not installed. Please run: pip install pyserial")
    sys.exit(1)

try:
    import scservo_sdk as scs
except ImportError:
    print("Error: 'scservo-sdk' not installed. Please run: pip install scservo-sdk")
    sys.exit(1)

# STS3215 Present_Position Address (size: 2 bytes)
REG_PRESENT_POSITION = 56
# Limits and offsets Addresses
REG_MIN_LIMIT = 9
REG_MAX_LIMIT = 11
REG_HOMING_OFFSET = 31

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

def scan_ports():
    ports = serial.tools.list_ports.comports()
    print("Available Serial Ports:")
    for port in ports:
        print(f"  {port.device} - {port.description}")

def open_arm(port_name, baudrate=1000000):
    if not port_name:
        return None, None
    
    print(f"Opening port {port_name} at {baudrate} baud...")
    port_handler = scs.PortHandler(port_name)
    packet_handler = scs.PacketHandler(0)  # STS3215 is Protocol 0
    
    if not port_handler.openPort():
        print(f"Error: Failed to open port {port_name}")
        return None, None
        
    if not port_handler.setBaudRate(baudrate):
        print(f"Error: Failed to set baudrate on port {port_name}")
        port_handler.closePort()
        return None, None
        
    return port_handler, packet_handler

def read_calibration(port_handler, packet_handler):
    calibration = {}
    for i, name in enumerate(MOTOR_NAMES):
        motor_id = MOTOR_IDS[i]
        
        # Read Min Limit
        val_min, comm, err = packet_handler.read2ByteTxRx(port_handler, motor_id, REG_MIN_LIMIT)
        if comm != scs.COMM_SUCCESS:
            print(f"  [Calib Warning] Failed to read Min_Limit for motor ID {motor_id} on {port_handler.port_name}")
            return None
            
        # Read Max Limit
        val_max, comm, err = packet_handler.read2ByteTxRx(port_handler, motor_id, REG_MAX_LIMIT)
        if comm != scs.COMM_SUCCESS:
            print(f"  [Calib Warning] Failed to read Max_Limit for motor ID {motor_id} on {port_handler.port_name}")
            return None
            
        # Read Homing Offset
        val_offset, comm, err = packet_handler.read2ByteTxRx(port_handler, motor_id, REG_HOMING_OFFSET)
        if comm != scs.COMM_SUCCESS:
            print(f"  [Calib Warning] Failed to read Homing_Offset for motor ID {motor_id} on {port_handler.port_name}")
            return None
            
        calibration[name] = {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": val_offset,
            "range_min": val_min,
            "range_max": val_max
        }
    return calibration

def read_positions(port_handler, packet_handler):
    # Try sync read first
    group = scs.GroupSyncRead(port_handler, packet_handler, REG_PRESENT_POSITION, 2)
    for id_ in MOTOR_IDS:
        group.addParam(id_)
        
    comm = group.txRxPacket()
    if comm == scs.COMM_SUCCESS:
        positions = []
        for id_ in MOTOR_IDS:
            if group.isAvailable(id_, REG_PRESENT_POSITION, 2):
                positions.append(group.getData(id_, REG_PRESENT_POSITION, 2))
            else:
                break
        if len(positions) == len(MOTOR_IDS):
            return positions

    # Fallback to sequential read
    positions = []
    for id_ in MOTOR_IDS:
        val, comm, err = packet_handler.read2ByteTxRx(port_handler, id_, REG_PRESENT_POSITION)
        if comm == scs.COMM_SUCCESS:
            positions.append(val)
        else:
            return None
    return positions

def main():
    parser = argparse.ArgumentParser(description="LeRobot Standalone Local Leader Arm Agent")
    parser.add_argument("--remote-ip", type=str, required=True, help="IP address of the remote Ubuntu host running LeRobot")
    parser.add_argument("--port", type=int, default=5005, help="UDP port on remote host (default: 5005)")
    parser.add_argument("--left-port", type=str, default=None, help="Serial/COM port for Left Leader Arm")
    parser.add_argument("--right-port", type=str, default=None, help="Serial/COM port for Right Leader Arm")
    parser.add_argument("--rate", type=int, default=50, help="Broadcast rate in Hz (default: 50)")
    parser.add_argument("--scan", action="store_true", help="Scan and list available COM ports")
    args = parser.parse_args()

    if args.scan:
        scan_ports()
        sys.exit(0)

    if not args.left_port and not args.right_port:
        print("Error: You must specify --left-port, --right-port, or both.")
        scan_ports()
        sys.exit(1)

    # Open arms
    left_ph, left_pd = open_arm(args.left_port)
    right_ph, right_pd = open_arm(args.right_port)

    if args.left_port and not left_ph:
        print("Error: Could not open Left Leader Arm.")
        sys.exit(1)
    if args.right_port and not right_ph:
        print("Error: Could not open Right Leader Arm.")
        sys.exit(1)

    # Read Calibration
    print("\nReading calibrations from physical arms...")
    cal_left = None
    cal_right = None
    
    if left_ph:
        cal_left = read_calibration(left_ph, left_pd)
        if not cal_left:
            print("Warning: Failed to read left arm calibration.")
        else:
            print("Successfully read Left Leader Arm calibration.")
            
    if right_ph:
        cal_right = read_calibration(right_ph, right_pd)
        if not cal_right:
            print("Warning: Failed to read right arm calibration.")
        else:
            print("Successfully read Right Leader Arm calibration.")

    # UDP socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_addr = (args.remote_ip, args.port)

    # Send Calibration Handshake Packet (Retry 3 times to ensure receipt)
    cal_packet = {
        "type": "calibration"
    }
    if left_ph:
        cal_packet["left"] = cal_left
    if right_ph:
        cal_packet["right"] = cal_right
    print(f"\nSending calibration to remote host {args.remote_ip}:{args.port}...")
    for _ in range(3):
        sock.sendto(json.dumps(cal_packet).encode("utf-8"), dest_addr)
        time.sleep(0.1)

    print("\nStarting present positions stream loop. Press Ctrl+C to stop.")
    delay = 1.0 / args.rate
    
    last_positions = {
        "left": [2048] * 6,
        "right": [2048] * 6
    }

    try:
        while True:
            t_start = time.perf_counter()
            
            # Read left
            if left_ph:
                pos = read_positions(left_ph, left_pd)
                if pos:
                    last_positions["left"] = pos
            
            # Read right
            if right_ph:
                pos = read_positions(right_ph, right_pd)
                if pos:
                    last_positions["right"] = pos
                    
            # Send payload
            payload = {
                "type": "positions"
            }
            if left_ph:
                payload["left"] = last_positions["left"]
            if right_ph:
                payload["right"] = last_positions["right"]
            sock.sendto(json.dumps(payload).encode("utf-8"), dest_addr)
            
            # Sleep to match target Hz
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.0, delay - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nStopping position stream.")
    finally:
        if left_ph:
            left_ph.closePort()
        if right_ph:
            right_ph.closePort()
        sock.close()
        print("Closed serial ports and socket. Exiting.")

if __name__ == "__main__":
    main()
