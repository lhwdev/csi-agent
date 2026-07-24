import threading
import time
import cv2
import os
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True

class CameraStreamer:
    _instance = None
    _lock = threading.RLock()
    _listeners = []
    enabled = False

    @classmethod
    def disable(cls):
        cls.enabled = False

    @classmethod
    def enable(cls):
        cls.enabled = True

    def listen_change(cls, listener):
        _listeners.append(listener)

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraStreamer, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance

    def __init__(self, port=8000, camera_configs=None):
        if self.initialized:
            return
        
        self.port = port
        self.camera_configs = camera_configs or {
            "left_cam": "/dev/lerobot/camera_1",
            "top": "/dev/lerobot/camera_0",
            "right_cam": "/dev/lerobot/camera_2"
        }
        
        # Fallbacks for camera paths
        self.camera_fallbacks = {
            "top": ["/dev/lerobot/camera_0", 0],
            "left_cam": ["/dev/lerobot/camera_1", 1],
            "right_cam": ["/dev/lerobot/camera_2", 2]
        }
        
        self.latest_frames = {k: None for k in self.camera_configs}
        self.last_update_time = {k: 0.0 for k in self.camera_configs}
        self.active_clients = {k: 0 for k in self.camera_configs}
        self.capture_threads = {}
        self.capture_running = {}
        self.frame_locks = {k: threading.Lock() for k in self.camera_configs}
        
        self.server = None
        self.server_thread = None
        self.server_port = None
        self.running = False
        self.telemetry_lock = threading.Lock()
        self.latest_telemetry = {"state": {}, "action": {}, "timestamp": 0.0}
        self.initialized = True

    def update_frame(self, cam_key, jpeg_bytes):
        """Called externally to push frames (e.g. from robot loop)."""
        if not self.enabled:
            return
        if cam_key not in self.latest_frames:
            return
        with self.frame_locks[cam_key]:
            self.latest_frames[cam_key] = jpeg_bytes
            self.last_update_time[cam_key] = time.time()

    def get_frame(self, cam_key):
        with self.frame_locks[cam_key]:
            return self.latest_frames[cam_key]

    def update_telemetry(self, state, action):
        if not self.enabled:
            return
        sanitized_state = self._sanitize_dict(state)
        sanitized_action = self._sanitize_dict(action)
        with self.telemetry_lock:
            self.latest_telemetry = {
                "state": sanitized_state,
                "action": sanitized_action,
                "timestamp": time.time()
            }

    def get_telemetry(self):
        with self.telemetry_lock:
            return self.latest_telemetry

    def _sanitize_dict(self, d):
        if d is None:
            return {}
        import numpy as np
        sanitized = {}
        for k, v in d.items():
            # Skip large media keys (e.g. camera image frames) to avoid bloating JSON
            if k in ["left_cam", "top", "right_cam", "images", "pixels"]:
                continue
            
            # Convert PyTorch tensors to numpy
            if hasattr(v, "detach"):
                v = v.detach().cpu()
            if hasattr(v, "numpy"):
                v = v.numpy()
            
            if isinstance(v, (np.ndarray, list)):
                if hasattr(v, "tolist"):
                    v = v.tolist()
                sanitized[k] = v
            else:
                try:
                    sanitized[k] = float(v)
                except (TypeError, ValueError):
                    sanitized[k] = str(v)
        return sanitized

    def is_externally_updated(self, cam_key):
        # If frame was updated externally in the last 2.0 seconds, consider it active
        return (time.time() - self.last_update_time[cam_key]) < 2.0

    def client_connected(self, cam_key):
        if not self.enabled:
            return
        with self._lock:
            self.active_clients[cam_key] += 1
            is_ext = self.is_externally_updated(cam_key)
            # Start background capture thread if offline
            if not is_ext and not self.capture_threads.get(cam_key):
                self.start_capture_thread(cam_key)

    def client_disconnected(self, cam_key):
        if not self.enabled:
            return
        with self._lock:
            if self.active_clients[cam_key] > 0:
                self.active_clients[cam_key] -= 1
            if self.active_clients[cam_key] == 0:
                self.stop_capture_thread(cam_key)

    def start_capture_thread(self, cam_key):
        if not self.enabled:
            return
        self.capture_running[cam_key] = True
        thread = threading.Thread(
            target=self._capture_loop, 
            args=(cam_key,), 
            name=f"CaptureThread-{cam_key}", 
            daemon=True
        )
        self.capture_threads[cam_key] = thread
        thread.start()

    def stop_capture_thread(self, cam_key):
        if not self.enabled:
            return
        self.capture_running[cam_key] = False
        thread = self.capture_threads.get(cam_key)
        if thread:
            thread.join(timeout=1.0)
            if cam_key in self.capture_threads:
                del self.capture_threads[cam_key]

    def stop_all_captures(self):
        """Stops all active camera device reading threads (to release resources for teleop)."""
        if not self.enabled:
            return
        with self._lock:
            for cam_key in list(self.capture_threads.keys()):
                self.capture_running[cam_key] = False
            for cam_key in list(self.capture_threads.keys()):
                thread = self.capture_threads.get(cam_key)
                if thread:
                    thread.join(timeout=1.0)
            self.capture_threads.clear()

    def _capture_loop(self, cam_key):
        # Resolve device path or index
        paths_to_try = self.camera_fallbacks.get(cam_key, [self.camera_configs[cam_key]])
        
        cap = None
        for path in paths_to_try:
            if not self.capture_running.get(cam_key, False):
                break
            try:
                if isinstance(path, str):
                    resolved = os.path.realpath(path)
                    cap = cv2.VideoCapture(resolved)
                else:
                    cap = cv2.VideoCapture(path)
                
                if cap and cap.isOpened():
                    # Set smaller resolution for faster web streaming
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    # Check if actually reading
                    ret, _ = cap.read()
                    if ret:
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception:
                if cap:
                    cap.release()
                    cap = None
        
        if not cap or not cap.isOpened():
            # If all failed, exit thread
            return

        try:
            while self.capture_running.get(cam_key, False) and self.active_clients.get(cam_key, 0) > 0:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Compress frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    with self.frame_locks[cam_key]:
                        self.latest_frames[cam_key] = jpeg.tobytes()
                
                # Limit to ~30 FPS
                time.sleep(0.033)
        finally:
            cap.release()

    def start_server(self):
        if not self.enabled:
            return None
        with self._lock:
            if self.server is not None:
                return self.server_port
            
            # Find free port starting from self.port
            port = self.port
            while True:
                try:
                    self.server = ThreadedHTTPServer(('0.0.0.0', port), CameraStreamHandler)
                    self.server_port = port
                    break
                except OSError as e:
                    if e.errno == 98:  # Address already in use
                        port += 1
                    else:
                        raise e
            
            self.running = True
            self.server_thread = threading.Thread(
                target=self.server.serve_forever, 
                name="CameraStreamServer", 
                daemon=True
            )
            self.server_thread.start()
            return self.server_port

    def stop_server(self):
        if not self.enabled:
            return
        server_to_shutdown = None
        with self._lock:
            if self.server is not None:
                self.running = False
                self.stop_all_captures()
                server_to_shutdown = self.server
                self.server = None
                self.server_thread = None
                self.server_port = None
        if server_to_shutdown is not None:
            server_to_shutdown.shutdown()
            server_to_shutdown.server_close()

    def get_server_url(self):
        if not self.server_port:
            return None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "localhost"
        return f"http://{ip}:{self.server_port}"

class CameraStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress request logs to avoid spamming the Jupyter/console log
        pass

    def do_GET(self):
        streamer = CameraStreamer()
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        if path == "/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            telemetry_data = streamer.get_telemetry()
            import json
            self.wfile.write(json.dumps(telemetry_data).encode("utf-8"))
            return

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            
            try:
                index_path = Path(__file__).parent / "resources" / "streamer_index.html"
                with open(index_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
            except Exception as e:
                html_content = f"<html><body><h1>Error loading streamer_index.html: {e}</h1></body></html>"

            self.wfile.write(html_content.encode("utf-8"))
            return

        parts = path.split("/")
        if len(parts) == 3 and parts[1] == "stream":
            cam_key = parts[2]
            if cam_key not in streamer.camera_configs:
                self.send_error(404, "Camera not found")
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            # Parse query params for FPS throttling
            fps_params = query.get("fps", [])
            target_fps = 15.0  # Default to 15 FPS
            if fps_params:
                try:
                    target_fps = float(fps_params[0])
                except ValueError:
                    pass
            target_interval = 1.0 / target_fps if target_fps > 0 else 0.033

            streamer.client_connected(cam_key)
            try:
                last_frame = None
                last_sent_time = 0.0
                while streamer.running:
                    now = time.time()
                    elapsed = now - last_sent_time
                    if elapsed < target_interval:
                        # Sleep the remaining time of the interval
                        time.sleep(max(0.001, target_interval - elapsed))
                        continue

                    frame = streamer.get_frame(cam_key)
                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        last_frame = frame
                        last_sent_time = time.time()
                    else:
                        time.sleep(0.01)
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                streamer.client_disconnected(cam_key)
        else:
            self.send_error(404, "Not Found")
