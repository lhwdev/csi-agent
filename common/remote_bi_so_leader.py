import socket
import threading
import json
import logging
import time
from pathlib import Path

from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.motors.encoding_utils import decode_sign_magnitude
from lerobot.types import RobotAction

logger = logging.getLogger(__name__)

class RemoteBiSOLeader(BiSOLeader):
    def __init__(self, config: BiSOLeaderConfig, host: str = "0.0.0.0", port: int = 5005):
        super().__init__(config)
        self.host = host
        self.port = port
        self._connected = False
        
        self.latest_raw_left = None
        self.latest_raw_right = None
        
        self.udp_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        # Returns True if both arms have calibration populated
        return bool(self.left_arm.calibration) and bool(self.right_arm.calibration)

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
            
        logger.info(f"Connecting RemoteBiSOLeader: binding UDP socket to {self.host}:{self.port}")
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow reusing address
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.5)
        
        self.stop_event.clear()
        self.udp_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.udp_thread.start()
        self._connected = True
        
        # Wait for the first handshake/positions packet
        logger.info("Waiting for local_leader_agent.py to start streaming positions...")
        start_wait = time.perf_counter()
        while self.latest_raw_left is None and self.latest_raw_right is None:
            time.sleep(0.1)
            if time.perf_counter() - start_wait > 5.0:
                logger.warning("No data received from local leader arm yet. Make sure local_leader_agent.py is running on Windows/macOS and sending data to this host.")
                break
                
        logger.info("RemoteBiSOLeader successfully connected and listening.")

    def _receive_loop(self):
        while not self.stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
                payload = json.loads(data.decode("utf-8"))
                
                pkt_type = payload.get("type")
                if pkt_type == "calibration":
                    self._handle_calibration_packet(payload)
                elif pkt_type == "positions":
                    with self.lock:
                        if "left" in payload:
                            self.latest_raw_left = payload["left"]
                        if "right" in payload:
                            self.latest_raw_right = payload["right"]
            except socket.timeout:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Error in UDP receiver loop: {e}")
                break

    def _handle_calibration_packet(self, payload):
        logger.info("Received calibration packet from local agent.")
        try:
            if "left" in payload and payload["left"]:
                cal_dict = {}
                for name, info in payload["left"].items():
                    cal_dict[name] = MotorCalibration(
                        id=info["id"],
                        drive_mode=info["drive_mode"],
                        homing_offset=info["homing_offset"],
                        range_min=info["range_min"],
                        range_max=info["range_max"]
                    )
                self.left_arm.calibration = cal_dict
                self.left_arm.bus.calibration = cal_dict
                self.left_arm._save_calibration()
                logger.info(f"Saved left arm calibration to {self.left_arm.calibration_fpath}")

            if "right" in payload and payload["right"]:
                cal_dict = {}
                for name, info in payload["right"].items():
                    cal_dict[name] = MotorCalibration(
                        id=info["id"],
                        drive_mode=info["drive_mode"],
                        homing_offset=info["homing_offset"],
                        range_min=info["range_min"],
                        range_max=info["range_max"]
                    )
                self.right_arm.calibration = cal_dict
                self.right_arm.bus.calibration = cal_dict
                self.right_arm._save_calibration()
                logger.info(f"Saved right arm calibration to {self.right_arm.calibration_fpath}")
        except Exception as e:
            logger.error(f"Failed to process calibration packet: {e}")

    def get_action(self) -> RobotAction:
        if not self._connected:
            raise RuntimeError("RemoteBiSOLeader is not connected.")
            
        with self.lock:
            raw_left = list(self.latest_raw_left) if self.latest_raw_left is not None else None
            raw_right = list(self.latest_raw_right) if self.latest_raw_right is not None else None
            
        action_dict = {}
        
        # Left Arm
        if raw_left is not None and self.left_arm.calibration:
            raw_left_ids = {i+1: raw_left[i] for i in range(len(raw_left))}
            decoded_left = {id_: decode_sign_magnitude(val, 15) for id_, val in raw_left_ids.items()}
            normalized_left = self.left_arm.bus._normalize(decoded_left)
            for id_, val in normalized_left.items():
                name = self.left_arm.bus._id_to_name(id_)
                action_dict[f"left_{name}.pos"] = val
                
        # Right Arm
        if raw_right is not None and self.right_arm.calibration:
            raw_right_ids = {i+1: raw_right[i] for i in range(len(raw_right))}
            decoded_right = {id_: decode_sign_magnitude(val, 15) for id_, val in raw_right_ids.items()}
            normalized_right = self.right_arm.bus._normalize(decoded_right)
            for id_, val in normalized_right.items():
                name = self.right_arm.bus._id_to_name(id_)
                action_dict[f"right_{name}.pos"] = val
                
        return action_dict

    def disconnect(self) -> None:
        if not self._connected:
            return
        logger.info("Disconnecting RemoteBiSOLeader...")
        self.stop_event.set()
        if hasattr(self, "sock"):
            self.sock.close()
        if self.udp_thread:
            self.udp_thread.join(timeout=1.0)
        self._connected = False
        logger.info("RemoteBiSOLeader disconnected.")
