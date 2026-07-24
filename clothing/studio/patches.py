import numpy as np
import io

# Monkey patch _CameraEncoderThread.run to avoid slow RunningQuantileStats updates on every frame
try:
    import lerobot.datasets.video_utils
    import queue
    from fractions import Fraction
    from PIL import Image
    import av
    import contextlib
    import logging

    def _patched_camera_encoder_thread_run(self):
        container = None
        output_stream = None
        frame_count = 0

        try:
            logging.getLogger("libav").setLevel(av.logging.WARNING)

            while True:
                try:
                    frame_data = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

                if frame_data is None:
                    break

                if isinstance(frame_data, np.ndarray):
                    if frame_data.ndim == 3 and frame_data.shape[0] == 3:
                        frame_data = frame_data.transpose(1, 2, 0)
                    if frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8)

                if container is None:
                    height, width = frame_data.shape[:2]
                    self.video_path.parent.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.video_path), "w")
                    output_stream = container.add_stream(self.vcodec, self.fps, options=self.codec_options)
                    output_stream.pix_fmt = self.pix_fmt
                    output_stream.width = width
                    output_stream.height = height
                    output_stream.time_base = Fraction(1, self.fps)

                pil_img = Image.fromarray(frame_data)
                video_frame = av.VideoFrame.from_image(pil_img)
                video_frame.pts = frame_count
                video_frame.time_base = Fraction(1, self.fps)
                packet = output_stream.encode(video_frame)
                if packet:
                    container.mux(packet)

                frame_count += 1

            if output_stream is not None:
                packet = output_stream.encode()
                if packet:
                    container.mux(packet)

            if container is not None:
                container.close()

            av.logging.restore_default_callback()
            self.result_queue.put(("ok", None))

        except Exception as e:
            import traceback
            logging.error(f"Encoder thread error: {e}")
            traceback.print_exc()
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()
            self.result_queue.put(("error", str(e)))

    lerobot.datasets.video_utils._CameraEncoderThread.run = _patched_camera_encoder_thread_run
except Exception as e:
    print(f"Warning: Failed to monkey patch _CameraEncoderThread.run: {e}")


# Monkey patch compute_episode_stats to avoid writing stats for intervention and round
try:
    import lerobot.datasets.compute_stats
    import lerobot.datasets.dataset_writer
    import lerobot.datasets.dataset_tools

    # Check if we have already saved the original function to prevent recursion on module reload
    if not hasattr(lerobot.datasets.compute_stats, "_original_compute_episode_stats"):
        lerobot.datasets.compute_stats._original_compute_episode_stats = lerobot.datasets.compute_stats.compute_episode_stats

    _orig_compute_episode_stats = lerobot.datasets.compute_stats._original_compute_episode_stats

    def _patched_compute_episode_stats(episode_data, features, quantile_list=None):
        filtered_episode_data = {k: v for k, v in episode_data.items() if k not in {"intervention", "round"}} if episode_data else episode_data
        filtered_features = {k: v for k, v in features.items() if k not in {"intervention", "round"}} if features else features
        return _orig_compute_episode_stats(filtered_episode_data, filtered_features, quantile_list)

    lerobot.datasets.compute_stats.compute_episode_stats = _patched_compute_episode_stats
    lerobot.datasets.dataset_writer.compute_episode_stats = _patched_compute_episode_stats
    lerobot.datasets.dataset_tools.compute_episode_stats = _patched_compute_episode_stats
except Exception as e:
    print(f"Warning: Failed to monkey patch compute_episode_stats: {e}")


class JointPlotter:
    def __init__(self, history_limit=50):
        self.history_limit = history_limit
        self.state_history = {}
        self.action_history = {}
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.joint_keys = None
        self.colors = [
            '#e63946', '#f4a261', '#e9c46a', '#2a9d8f', '#457b9d', '#1d3557', '#a8dadc'
        ]

    def update(self, state, action):
        if not state and not action:
            return None
            
        state_keys = sorted([k for k in state.keys() if k.endswith('.pos')]) if state else []
        action_keys = sorted([k for k in action.keys() if k.endswith('.pos')]) if action else []
        all_keys = sorted(list(set(state_keys + action_keys)))
        
        if not all_keys:
            return None
            
        for k in all_keys:
            if k not in self.state_history:
                self.state_history[k] = []
            if k not in self.action_history:
                self.action_history[k] = []
                
            s_val = state.get(k, None) if state else None
            a_val = action.get(k, None) if action else None
            
            if s_val is not None:
                if hasattr(s_val, "item"): s_val = s_val.item()
                self.state_history[k].append(float(s_val))
            else:
                self.state_history[k].append(None)
                
            if a_val is not None:
                if hasattr(a_val, "item"): a_val = a_val.item()
                self.action_history[k].append(float(a_val))
            else:
                self.action_history[k].append(None)
                
            if len(self.state_history[k]) > self.history_limit:
                self.state_history[k].pop(0)
            if len(self.action_history[k]) > self.history_limit:
                self.action_history[k].pop(0)
                
        if self.fig is None:
            self.joint_keys = all_keys
            self.groups = {"left": [], "right": [], "other": []}
            for k in self.joint_keys:
                if k.startswith("left_"): self.groups["left"].append(k)
                elif k.startswith("right_"): self.groups["right"].append(k)
                else: self.groups["other"].append(k)
                    
            active_groups = [g for g, keys in self.groups.items() if len(keys) > 0]
            num_subplots = len(active_groups)
            
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(4.5, 1.3 * num_subplots), dpi=110)
            self.fig.patch.set_facecolor('#ffffff')
            axs = self.fig.subplots(num_subplots, 1)
            if num_subplots == 1:
                axs = [axs]
            
            for ax, gname in zip(axs, active_groups):
                ax.set_facecolor('#f8f9fa')
                ax.set_title(f"{gname.capitalize()} Arm (Solid: State, Dashed: Action)", fontsize=8, fontweight='bold', pad=6, color='#212529')
                ax.tick_params(axis='both', labelsize=7, colors='#495057')
                ax.grid(True, color='#dee2e6', linestyle=':', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#ced4da')
                ax.spines['bottom'].set_color('#ced4da')
                
                self.axes[gname] = ax
                
                for idx, k in enumerate(self.groups[gname]):
                    clean_name = k.removeprefix("left_").removeprefix("right_").split(".")[0]
                    color = self.colors[idx % len(self.colors)]
                    s_line, = ax.plot([], [], label=clean_name, color=color, linewidth=1.5, linestyle='-')
                    a_line, = ax.plot([], [], label="_nolegend_", color=color, linewidth=1.2, linestyle='--')
                    self.lines[k] = (s_line, a_line)
                    
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, frameon=True, facecolor='#ffffff', edgecolor='#e9ecef')
                
            self.fig.tight_layout()
            
        for gname, ax in self.axes.items():
            min_y, max_y = float('inf'), float('-inf')
            for k in self.groups[gname]:
                s_data = self.state_history[k]
                a_data = self.action_history[k]
                x_data = list(range(len(s_data)))
                s_line, a_line = self.lines[k]
                
                s_plot_y = [y for y in s_data if y is not None]
                s_plot_x = [x for x, y in zip(x_data, s_data) if y is not None]
                a_plot_y = [y for y in a_data if y is not None]
                a_plot_x = [x for x, y in zip(x_data, a_data) if y is not None]
                
                s_line.set_data(s_plot_x, s_plot_y)
                a_line.set_data(a_plot_x, a_plot_y)
                
                for y in s_plot_y + a_plot_y:
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
                    
            if min_y != float('inf') and max_y != float('-inf'):
                padding = max((max_y - min_y) * 0.1, 0.1)
                ax.set_ylim(min_y - padding, max_y + padding)
                ax.set_xlim(0, self.history_limit)
                
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(self.fig)
        buf = io.BytesIO()
        self.fig.savefig(buf, format='jpeg', bbox_inches='tight', dpi=160)
        buf.seek(0)
        return buf.getvalue()

    def cleanup(self):
        self.fig = None
