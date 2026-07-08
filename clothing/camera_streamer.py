import threading
import time
import cv2
import os
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True

class CameraStreamer:
    _instance = None
    _lock = threading.RLock()
    _listeners = []

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
        self.initialized = True

    def update_frame(self, cam_key, jpeg_bytes):
        """Called externally to push frames (e.g. from robot loop)."""
        if cam_key not in self.latest_frames:
            return
        with self.frame_locks[cam_key]:
            self.latest_frames[cam_key] = jpeg_bytes
            self.last_update_time[cam_key] = time.time()

    def get_frame(self, cam_key):
        with self.frame_locks[cam_key]:
            return self.latest_frames[cam_key]

    def is_externally_updated(self, cam_key):
        # If frame was updated externally in the last 2.0 seconds, consider it active
        return (time.time() - self.last_update_time[cam_key]) < 2.0

    def client_connected(self, cam_key):
        with self._lock:
            self.active_clients[cam_key] += 1
            is_ext = self.is_externally_updated(cam_key)
            # Start background capture thread if offline
            if not is_ext and not self.capture_threads.get(cam_key):
                self.start_capture_thread(cam_key)

    def client_disconnected(self, cam_key):
        with self._lock:
            if self.active_clients[cam_key] > 0:
                self.active_clients[cam_key] -= 1
            if self.active_clients[cam_key] == 0:
                self.stop_capture_thread(cam_key)

    def start_capture_thread(self, cam_key):
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
        self.capture_running[cam_key] = False
        thread = self.capture_threads.get(cam_key)
        if thread:
            thread.join(timeout=1.0)
            if cam_key in self.capture_threads:
                del self.capture_threads[cam_key]

    def stop_all_captures(self):
        """Stops all active camera device reading threads (to release resources for teleop)."""
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

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            
            html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LeRobot 3-Camera Streamer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: hsl(220, 15%, 10%);
            --card-bg: hsl(220, 15%, 15%);
            --border-color: hsl(220, 15%, 22%);
            --accent-glow: linear-gradient(135deg, hsl(260, 80%, 55%) 0%, hsl(210, 80%, 55%) 100%);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
        }
        .header {
            text-align: center;
            margin-bottom: 32px;
            max-width: 800px;
        }
        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            background: var(--accent-glow);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .subtitle {
            font-size: 1rem;
            color: var(--text-muted);
            margin: 0;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 24px;
            width: 100%;
            max-width: 1400px;
        }
        .camera-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .camera-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--accent-glow);
            opacity: 0.8;
        }
        .camera-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
        }
        .camera-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 2s infinite;
        }
        .img-container {
            width: 100%;
            aspect-ratio: 4/3;
            border-radius: 10px;
            overflow: hidden;
            background-color: #000;
            border: 1px solid var(--border-color);
        }
        img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>LeRobot Camera Studio</h1>
        <p class="subtitle">Real-time MJPEG Stream representing bimanual follower configuration</p>
        <div style="margin-top: 16px; display: flex; align-items: center; gap: 8px; justify-content: center;">
            <label for="fps-select" style="font-size: 0.9rem; color: var(--text-muted);">Stream Framerate:</label>
            <select id="fps-select" style="background-color: var(--card-bg); color: var(--text-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 6px 12px; font-family: inherit; cursor: pointer; outline: none;">
                <option value="5">5 FPS (Low Bandwidth)</option>
                <option value="10">10 FPS (Medium)</option>
                <option value="15" selected>15 FPS (Default)</option>
                <option value="30">30 FPS (High Performance)</option>
            </select>
        </div>
    </div>
    <div class="grid">
        <div class="camera-card">
            <div class="camera-title"><span class="status-dot"></span> Left Camera</div>
            <div class="img-container">
                <img id="left_cam_img" src="/stream/left_cam" alt="Left Camera Stream" />
            </div>
        </div>
        <div class="camera-card">
            <div class="camera-title"><span class="status-dot"></span> Top Camera</div>
            <div class="img-container">
                <img id="top_img" src="/stream/top" alt="Top Camera Stream" />
            </div>
        </div>
        <div class="camera-card">
            <div class="camera-title"><span class="status-dot"></span> Right Camera</div>
            <div class="img-container">
                <img id="right_cam_img" src="/stream/right_cam" alt="Right Camera Stream" />
            </div>
        </div>
    </div>
    <script>
        const select = document.getElementById('fps-select');
        const imgLeft = document.getElementById('left_cam_img');
        const imgTop = document.getElementById('top_img');
        const imgRight = document.getElementById('right_cam_img');
        
        function updateStreams() {
            const fps = select.value;
            imgLeft.src = "/stream/left_cam?fps=" + fps;
            imgTop.src = "/stream/top?fps=" + fps;
            imgRight.src = "/stream/right_cam?fps=" + fps;
        }
        
        select.addEventListener('change', updateStreams);
        // Initial set
        updateStreams();
    </script>
</body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))
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
