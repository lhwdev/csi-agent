import cv2
import threading
import numpy as np
import ipywidgets as widgets
from typing import Dict, Any, List, Optional


class ClassifierUIMixin:
    """
    Mixin for Classifier Studio UI layout, widget configuration, telemetry formatting,
    and prediction probability display.
    """

    def build_cameras(self):
        self.top_camera_widget = widgets.Image(
            format="jpeg",
            layout=widgets.Layout(
                border="1px solid #ccc",
                border_radius="8px",
                width="100%",
                max_width="640px",
                height="auto",
                min_width="0",
            ),
        )
        self.cameras_layout = widgets.VBox(
            [
                widgets.Label("Top Cam", layout=widgets.Layout(font_weight="bold")),
                self.top_camera_widget,
            ],
            layout=widgets.Layout(align_items="center", width="100%", margin="5px"),
        )

    def build_controls(self):
        super().build_controls()

        # Class Label Selector Dropdown
        self.label_dropdown = widgets.Dropdown(
            options=[(lbl, lbl) for lbl in self.class_labels],
            value=self.class_labels[0],
            description="Active Label:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%", margin="4px 0px"),
        )

        # Action Buttons
        self.start_btn.description = "Stream Rec (R)"
        self.start_btn.button_style = "success"
        self.start_btn.disabled = False
        self.start_btn.on_click(self.on_toggle_stream_recording)

        self.snapshot_btn.description = "Snapshot (C)"
        self.snapshot_btn.button_style = "info"
        self.snapshot_btn.disabled = False
        self.snapshot_btn.on_click(self.on_snapshot_clicked)

        self.train_btn = widgets.Button(
            description="Train Classifier (T)",
            icon="cogs",
            button_style="primary",
            layout=widgets.Layout(width="100%"),
        )
        self.train_btn.on_click(self.on_train_clicked)

        # Training Controls
        self.epochs_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=15,
            step=1,
            description="Epochs:",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )

        self.lr_dropdown = widgets.Dropdown(
            options=[("1e-5", 1e-5), ("3e-5", 3e-5), ("5e-5 (Default)", 5e-5), ("1e-4", 1e-4)],
            value=5e-5,
            description="LR:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )

        # Prediction probability vector display
        self.prob_bars_widget = widgets.HTML(
            value="<div style='font-size:11px; color:#666;'>No predictions yet</div>",
            layout=widgets.Layout(width="100%"),
        )

        # Samples count summary widget
        self.counts_summary_widget = widgets.HTML()

        # Update controls row layout
        self.control_row_1.children = [self.start_btn, self.snapshot_btn, self.quit_btn]
        self.control_row_2.children = [self.label_dropdown]
        self.navigation_row.children = []

        self.controls_box.children = [
            self.control_row_1,
            self.control_row_2,
            widgets.HTML("<div style='height: 4px;'></div>"),
            self.counts_summary_widget,
        ]

    def build_training_panel(self):
        """Assembles Training tab inside studio tab widget."""
        self.train_status_widget = widgets.HTML(
            "<div style='font-size:12px; font-family:monospace;'>Status: Ready</div>"
        )

        self.training_panel = widgets.VBox(
            [
                widgets.HTML("<b>Classifier Fine-Tuning Parameters</b>"),
                self.epochs_slider,
                self.lr_dropdown,
                widgets.HTML("<div style='height: 6px;'></div>"),
                self.train_btn,
                widgets.HTML("<div style='height: 6px;'></div>"),
                self.train_status_widget,
            ],
            layout=widgets.Layout(padding="8px", width="100%"),
        )

        self.studio_tab.children = [self.telemetry_widget, self.training_panel, self.shortcuts_box]
        self.training_tab_index = 1
        self.studio_tab.set_title(0, "Telemetry")
        self.studio_tab.set_title(1, "Training")
        self.studio_tab.set_title(2, "Shortcuts")
        self.studio_tab.selected_index = 0

    def build_shortcuts(self):
        super().build_shortcuts()
        self.shortcut_legend.value = """
        <div style="background: #f8f9fa; border: 1px dashed #ccc; border-radius: 8px; padding: 8px; font-size: 11px;">
          <b>Keyboard Shortcuts:</b><br/>
          <span style="font-family: monospace;"><b>C</b>: Snapshot frame &nbsp;|&nbsp; <b>R</b>: Toggle stream recording &nbsp;|&nbsp; <b>T</b>: Open Training Tab &nbsp;|&nbsp; <b>Q</b>: Exit</span>
        </div>
        """

        def on_shortcut_change(change):
            key = change["new"]
            if not key:
                return
            key = key.lower()
            self.shortcut_input.value = ""
            if not self.shortcut_toggle.value:
                return

            if key == "c":
                self.on_snapshot_clicked(None)
            elif key == "r":
                self.on_toggle_stream_recording(None)
            elif key == "t":
                self.studio_tab.selected_index = self.training_tab_index
            elif key == "q":
                self.cleanup_studio_sync()

        self.shortcut_input.observe(on_shortcut_change, names="value")

    def update_counts_display(self):
        counts = self.image_dataset.get_class_counts()
        total = sum(counts.values())

        badges = []
        for lbl, cnt in counts.items():
            color = "#1976d2" if cnt > 0 else "#9e9e9e"
            badges.append(
                f"<span style='background-color:{color}; color:white; padding:2px 6px; border-radius:4px; margin-right:4px; font-size:10px;'>{lbl}: <b>{cnt}</b></span>"
            )

        self.counts_summary_widget.value = f"""
        <div style="font-size: 11px; margin-top: 4px;">
            <b>Dataset Images (Total: {total}):</b><br/>
            <div style="margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px;">{"".join(badges)}</div>
        </div>
        """

    def update_status_card(self):
        state_class = "status-recording" if self.is_stream_recording else ("status-resetting" if self._is_training_running else "status-idle")
        status_text = "STREAM RECORDING" if self.is_stream_recording else ("TRAINING MODEL" if self._is_training_running else "IDLE")

        fps_text = f" | {self.current_fps:.1f} FPS" if self.current_fps > 0 else ""
        total_samples = self.image_dataset.get_total_samples()

        pred_str = "N/A"
        if self.latest_pred_class is not None and self.latest_pred_class < len(self.class_labels):
            pred_str = f"{self.class_labels[self.latest_pred_class]} ({self.latest_confidence:.1%})"

        target_label = (
            self.label_dropdown.value
            if hasattr(self, "label_dropdown")
            else (self.class_labels[0] if getattr(self, "class_labels", None) else "N/A")
        )

        self.status_card.value = f"""
        <div class="status-card {state_class}">
            <div class="status-text">{status_text}</div>
            <div style="font-size: 13px; font-weight: 700; margin-top: 4px;">Target Label: {target_label}</div>
            <div style="font-size: 12px; margin-top: 2px; opacity: 0.9;">Total Images: {total_samples} | Pred: {pred_str}{fps_text}</div>
        </div>
        """

    def update_telemetry(self, obs=None, raw_action=None, current_fps=0.0):
        self.current_fps = current_fps
        self.update_status_card()

        if obs is not None:
            self.latest_state = obs
            for key, widget in [
                ("left_cam", getattr(self, "left_camera_widget", None)),
                ("top", getattr(self, "top_camera_widget", None)),
                ("right_cam", getattr(self, "right_camera_widget", None)),
            ]:
                if widget is None:
                    continue
                img_np = None
                if key in obs and obs[key] is not None:
                    img_np = obs[key]
                elif f"observation.images.{key}" in obs and obs[f"observation.images.{key}"] is not None:
                    img_np = obs[f"observation.images.{key}"]

                if img_np is not None:
                    try:
                        if hasattr(img_np, "cpu"):
                            img_np = img_np.cpu().numpy()
                        if isinstance(img_np, np.ndarray):
                            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                                img_np = np.transpose(img_np, (1, 2, 0))
                            if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                                if img_np.max() <= 1.0:
                                    img_np = (img_np * 255.0).astype(np.uint8)
                                else:
                                    img_np = img_np.astype(np.uint8)

                            h, w = img_np.shape[:2]
                            if w > 480:
                                new_h = int(h * (480.0 / w))
                                img_np = cv2.resize(img_np, (480, new_h), interpolation=cv2.INTER_AREA)

                            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                            _, jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            widget.value = jpeg.tobytes()
                    except Exception:
                        pass

        # Perform frame capture if stream recording is active
        if self.is_stream_recording and obs is not None:
            self._capture_observation_frame(obs)

        # Run model inference prediction asynchronously in background thread if not already running
        if self.classifier_tracker is not None and obs is not None:
            if not getattr(self, "_is_classifying", False):
                self._is_classifying = True
                def run_async_classification(obs_snapshot):
                    try:
                        res = self.classifier_tracker.classify_frame(obs_snapshot)
                        if res is not None:
                            probs_dict, pred_class, max_conf = res
                            self.latest_probs = probs_dict
                            self.latest_pred_class = pred_class
                            self.latest_confidence = max_conf
                            self._update_probability_bars(probs_dict, pred_class)
                    except Exception:
                        pass
                    finally:
                        self._is_classifying = False

                threading.Thread(target=run_async_classification, args=(obs,), daemon=True).start()

        telemetry_html = f"<b>CLASSIFIER STUDIO TELEMETRY</b><br/>Loop Rate: {current_fps:.1f} Hz<br/>"
        if self.latest_pred_class is not None and self.latest_pred_class < len(self.class_labels):
            telemetry_html += f"<b>Top Prediction:</b> {self.class_labels[self.latest_pred_class]} ({self.latest_confidence:.2f})<br/>"

        self.telemetry_widget.value = f'<div class="telemetry-card">{telemetry_html}{self.prob_bars_widget.value}</div>'

    def _update_probability_bars(self, probs_dict: Dict[int, float], pred_class: int):
        bars_html = "<div style='margin-top:6px; font-family:sans-serif; font-size:11px;'>"
        for idx, lbl in enumerate(self.class_labels):
            prob = probs_dict.get(idx, 0.0)
            pct = int(prob * 100)
            bar_color = "#4caf50" if idx == pred_class else "#2196f3"
            font_weight = "bold" if idx == pred_class else "normal"
            bars_html += f"""
            <div style='display:flex; align-items:center; margin-bottom:2px;'>
                <div style='width:70px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:{font_weight};'>{lbl}:</div>
                <div style='flex:1; background:#eee; border-radius:3px; height:12px; margin:0 6px; overflow:hidden;'>
                    <div style='background:{bar_color}; width:{pct}%; height:100%;'></div>
                </div>
                <div style='width:35px; text-align:right;'>{pct}%</div>
            </div>
            """
        bars_html += "</div>"
        self.prob_bars_widget.value = bars_html
