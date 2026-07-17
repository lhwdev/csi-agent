import time
import ipywidgets as widgets
from pathlib import Path
import numpy as np
from types import SimpleNamespace

class DAggerUIMixin:
    """
    Mixin for DAgger-specific UI layout, widget configuration, and telemetry formatting.
    Designed to separate the front-end Jupyter widgets from the core DAgger control loops,
    mirroring the StudioUIMixin / BaseInteractiveStudio split in ui.py / core.py.
    """

    # ------------------------------------------------------------------
    # Control building
    # ------------------------------------------------------------------

    def build_controls(self):
        super().build_controls()
        self.pause_btn.description = "Pause/Resume (P)"
        self.correction_btn = widgets.Button(description="Correction (C)", icon="edit", button_style="warning", disabled=True)
        self.train_btn = widgets.Button(description="Train (T)", icon="cogs", button_style="info", disabled=False)
        self.fps_slider = widgets.IntSlider(value=30, min=1, max=30, step=1, description='Policy FPS:', continuous_update=False)

        self.round_counter = widgets.IntText(value=getattr(self, "current_training_round", 1), description='Round:', layout=widgets.Layout(width='140px'))
        def on_round_change(change):
            self.current_training_round = change['new']
            self.update_status_card()
            self.update_training_info()
        self.round_counter.observe(on_round_change, names='value')

        self.epochs_slider = widgets.FloatSlider(
            value=getattr(self.params, "dagger_train_new_epochs", 3.0),
            min=0.5,
            max=20.0,
            step=0.5,
            description='Train Epochs:',
            continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%', max_width='100%')
        )
        def on_epochs_change(change):
            self.update_training_info()
        self.epochs_slider.observe(on_epochs_change, names='value')

        self.train_info_widget = widgets.HTML()

        self.start_btn.on_click(self.on_start_clicked)
        self.pause_btn.on_click(self.on_pause_clicked)
        self.correction_btn.on_click(self.on_correction_clicked)
        self.train_btn.on_click(self.on_train_clicked)
        self.next_step_btn.on_click(self.on_next_step_clicked)
        self.prev_step_btn.on_click(self.on_prev_step_clicked)
        self.stop_btn.on_click(self.on_stop_clicked)
        self.discard_btn.on_click(self.on_discard_clicked)

        self.control_row_1.children = [self.start_btn, self.pause_btn, self.correction_btn, self.prev_step_btn, self.next_step_btn]
        self.control_row_2.children = [self.stop_btn, self.discard_btn, self.fps_slider, self.round_counter]

    # ------------------------------------------------------------------
    # Shortcut building
    # ------------------------------------------------------------------

    def build_shortcuts(self):
        super().build_shortcuts()
        self.shortcut_legend.value = "<b>Shortcuts:</b> Space/R: Start, P: Pause/Resume, C: Correction, S: Save, D: Discard, T: Train, N: Next Step, B: Prev Step, Q: Exit"
        def on_shortcut_change(change):
            key = change['new']
            if not key: return
            key = key.lower()
            self.shortcut_input.value = ""
            if not self.shortcut_toggle.value: return

            if key == 'n' and not self.next_step_btn.disabled: self.on_next_step_clicked(None)
            elif key == 'b' and not self.prev_step_btn.disabled: self.on_prev_step_clicked(None)
            elif key in (' ', 'space', 'r'):
                if self.recording_state == "IDLE" and not self.start_btn.disabled: self.on_start_clicked(None)
                elif self.dagger_phase in ["AUTONOMOUS", "CORRECTING_LOCKED", "CORRECTING_ACTIVE", "PAUSED"] and not self.correction_btn.disabled:
                    self.on_correction_clicked(None)
            elif key == 'c' and not self.correction_btn.disabled: self.on_correction_clicked(None)
            elif key == 'p' and not self.pause_btn.disabled: self.on_pause_clicked(None)
            elif key == 's' and not self.stop_btn.disabled: self.on_stop_clicked(None)
            elif key == 'd' and not self.discard_btn.disabled: self.on_discard_clicked(None)
            elif key == 't' and not self.train_btn.disabled:
                self.prev_tab_index = self.studio_tab.selected_index
                self.studio_tab.selected_index = self.training_tab_index
            elif key == 'q': self.cleanup_studio_sync()

        self.shortcut_input.observe(on_shortcut_change, names='value')

    # ------------------------------------------------------------------
    # Navigation button state (DAgger has no multi-step navigation)
    # ------------------------------------------------------------------

    def update_navigation_buttons(self):
        pass

    # ------------------------------------------------------------------
    # Status card
    # ------------------------------------------------------------------

    def update_status_card(self):
        self.update_header()
        state_classes = {
            "IDLE": "status-idle",
            "RECORDING": "status-recording",
            "PAUSED": "status-paused",
            "SAVING": "status-saving",
            "TRAINING": "status-resetting"
        }
        state_class = state_classes.get(self.recording_state, "status-idle")

        if self.recording_state == "RECORDING":
            sub = f"Capturing HIL ({self.dagger_phase}) | {self.frames_in_episode} frames"
        elif self.recording_state == "PAUSED":
            sub = "Paused"
        elif self.recording_state == "SAVING":
            sub = "Writing round expert demonstration..."
        elif self.recording_state == "TRAINING":
            sub = "Training policy..."
        else:
            sub = f"Ready to collect round {self.current_training_round} data"

        fps_text = f" | {getattr(self, 'current_fps', 0):.1f} FPS" if getattr(self, 'current_fps', 0) > 0 else ""

        cache = getattr(self, "episode_rounds_cache", {})
        round_eps = sum(1 for v in cache.values() if v == self.current_training_round)
        added_eps = self.current_episode_idx - getattr(self, "_initial_episode_idx", self.current_episode_idx)

        self.status_card.value = f"""
        <div class="status-card {state_class}">
            <div class="status-text">{self.recording_state}</div>
            <div style="font-size: 14px; font-weight: 700; margin-top: 5px; opacity: 0.95;">Ep: {self.current_episode_idx} | Rnd: {self.current_training_round} | Rnd eps: {round_eps} | Added: {added_eps}{fps_text}</div>
            <div style="font-size: 12px; margin-top: 3px; opacity: 0.8;">{sub}</div>
        </div>
        """

    # ------------------------------------------------------------------
    # Telemetry widget
    # ------------------------------------------------------------------

    def update_telemetry(self, obs, raw_action, current_fps, policy_action=None):
        super().update_telemetry(obs, raw_action, current_fps)
        text = f"<b>DAGGER TELEMETRY</b><br/>"
        text += f"Loop: {current_fps:.1f} Hz | Phase: {self.dagger_phase}<br/>"

        if obs is not None:
            pos_keys = sorted([k for k in obs.keys() if k.endswith('.pos')])
            if pos_keys:
                text += f"<b>Follower (State):</b><br/>"
                for k in pos_keys:
                    clean_name = k.removeprefix("left_").removeprefix("right_").split(".")[0]
                    text += f" &bull; {clean_name}: {obs[k]:.2f}<br/>"

        if policy_action is not None:
            text += f"<b>Policy Action (Commanded):</b><br/>"
            for k in sorted(policy_action.keys()):
                clean_name = k.removeprefix("left_").removeprefix("right_").split(".")[0]
                text += f" &bull; {clean_name}: {policy_action[k]:.2f}<br/>"

        self.telemetry_widget.value = f'<div class="telemetry-card">{text}</div>'

    # ------------------------------------------------------------------
    # Training panel assembly (called from on_datasets_initialized)
    # ------------------------------------------------------------------

    def build_training_panel(self):
        """Assemble and append the Training tab panel to the studio tab widget."""
        self.train_status_widget = widgets.HTML("<div class='train-status' style='font-size: 11px; color: #495057; font-family: monospace; border-top: 1px solid #dee2e6; margin-top: 4px; padding-top: 4px;'>Status: Ready</div>")
        self.train_btn.layout = widgets.Layout(width='100%')
        self.training_panel = widgets.VBox([
            self.train_info_widget,
            widgets.HTML("<div style='height: 4px;'></div>"),
            self.epochs_slider,
            widgets.HTML("<div style='height: 4px;'></div>"),
            self.train_status_widget,
            widgets.HTML("<div style='height: 4px;'></div>"),
            self.train_btn
        ], layout=widgets.Layout(align_items='stretch', width='100%', max_width='100%', overflow_x='hidden'))
        self.training_panel.add_class("training-card")

        # Append training panel to Tab
        children = list(self.studio_tab.children)
        children.append(self.training_panel)
        self.studio_tab.children = children
        self.training_tab_index = len(children) - 1
        self.studio_tab.set_title(self.training_tab_index, "Training")

    # ------------------------------------------------------------------
    # Training info widget
    # ------------------------------------------------------------------

    def update_training_info(self):
        if not hasattr(self, "train_info_widget") or self.train_info_widget is None:
            return

        current_training_round = getattr(self, "current_training_round", 1)
        newest_round_frames = 0
        if not hasattr(self, "_newest_round_frames_cache"):
            self._newest_round_frames_cache = {}

        if current_training_round in self._newest_round_frames_cache:
            newest_round_frames = self._newest_round_frames_cache[current_training_round]
        elif self.datasets and self.current_dataset_step_idx < len(self.datasets):
            ds = self.datasets[self.current_dataset_step_idx]
            try:
                hf_dataset = ds.hf_dataset
                if hf_dataset is not None and "round" in hf_dataset.column_names:
                    rounds = hf_dataset["round"]
                    rounds_arr = np.array(rounds)
                    if rounds_arr.ndim > 1:
                        rounds_arr = rounds_arr.squeeze()
                    newest_round_frames = int(np.sum(rounds_arr == current_training_round))
                    self._newest_round_frames_cache[current_training_round] = newest_round_frames
            except Exception:
                pass

        train_params = self.get_online_training_params()
        beta = getattr(train_params, "dagger_rehearsal_beta", 0.3)
        new_epochs = self.epochs_slider.value
        batch_size = train_params.batch_size

        if newest_round_frames > 0:
            computed_steps = int(((new_epochs * newest_round_frames) / beta) / batch_size)
            est_steps = max(getattr(self.params, "dagger_train_steps", 100), computed_steps)
        else:
            est_steps = getattr(self.params, "dagger_train_steps", 100)

        info_html = f"""
        <div style="font-family: sans-serif; font-size: 11px; line-height: 1.3;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; margin-bottom: 4px;">
                <div><span style="color: #666;">Round:</span> <strong style="color: #212529; float: right;">{self.current_training_round}</strong></div>
                <div><span style="color: #666;">New Frames:</span> <strong style="color: #212529; float: right;">{newest_round_frames}</strong></div>
                <div><span style="color: #666;">Beta:</span> <strong style="color: #212529; float: right;">{beta}</strong></div>
                <div><span style="color: #666;">Batch Size:</span> <strong style="color: #212529; float: right;">{batch_size}</strong></div>
            </div>
            <div style="border-top: 1px solid #eee; padding-top: 4px;">
                <span style="color: #333; font-weight: bold;">Est. Steps:</span>
                <strong style="color: #228be6; font-size: 12px; float: right;">{est_steps}</strong>
            </div>
        </div>
        """
        self.train_info_widget.value = info_html

    # ------------------------------------------------------------------
    # Training progress callback
    # ------------------------------------------------------------------

    def on_train_progress(self, info):
        self._last_train_info = info

        step = info.get("step", 0)
        total = info.get("total_steps", 100)
        loss = info.get("loss", 0.0)
        val_loss = info.get("val_loss")
        status = info.get("status", "training")

        if "checkpoint_dir" in info:
            self._last_checkpoint = Path(info["checkpoint_dir"]).name

        # Calculate percentage
        percent = (step / total) * 100 if total > 0 else 0

        # Calculate time (timer starts only when status is "training")
        start_time = getattr(self, "_train_start_time", None)
        start_step = getattr(self, "_train_start_step", 0)

        if status == "training" and start_time is None:
            start_time = time.time()
            start_step = step
            self._train_start_time = start_time
            self._train_start_step = start_step

        if start_time is not None:
            elapsed_sec = time.time() - start_time
            steps_done = step - start_step
            steps_left = total - step
            if steps_done > 0 and step < total:
                remaining_sec = steps_left * (elapsed_sec / steps_done)
            else:
                remaining_sec = 0 if step >= total else None
        else:
            elapsed_sec = 0.0
            remaining_sec = None

        def format_time(seconds):
            if seconds is None:
                return "--:--"
            seconds = int(seconds)
            if seconds < 0:
                seconds = 0
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            if hrs > 0:
                return f"{hrs}:{mins:02d}:{secs:02d}"
            return f"{mins:02d}:{secs:02d}"

        elapsed_str = format_time(elapsed_sec)
        remaining_str = format_time(remaining_sec)

        # Style chip component (stable size, no layout shift)
        last_chk = getattr(self, "_last_checkpoint", "none")
        if status == "saving":
            chip_text = "Saving..."
            chip_bg = "#e8f5e9"
            chip_border = "#c8e6c9"
            chip_color = "#2e7d32"
            chip_weight = "600"
        else:
            chip_text = f"Saved to {last_chk}"
            chip_bg = "#f1f3f5"
            chip_border = "#dee2e6"
            chip_color = "#495057"
            chip_weight = "500"

        chip_html = f"""<span style="
            display: inline-flex; align-items: center; justify-content: center;
            min-width: 120px; height: 18px;
            border-radius: 12px;
            font-size: 11px; font-weight: {chip_weight};
            padding: 0 8px; box-sizing: border-box; vertical-align: middle;
            margin-left: 6px;
            border: 1px solid {chip_border};
            background-color: {chip_bg}; color: {chip_color};
        ">{chip_text}</span>"""

        # Line 1: Status display
        if status == "completed":
            status_display = "Completed"
            status_color = "#2b8a3e"
        elif status == "starting":
            status_display = "Starting"
            status_color = "#495057"
        else:
            status_display = "Training"
            status_color = "#228be6"

        line1 = f'Status: <strong style="color: {status_color};">{status_display}</strong> {chip_html}'

        # Line 2: Step & Loss & Time
        val_loss_str = f"{val_loss:.5f}" if val_loss is not None else "—"
        line2 = f"Step {step}/{total} / Loss T={loss:.5f}, V={val_loss_str} [{elapsed_str} &lt; {remaining_str}]"

        # Line 3: Progress bar with color changing when done
        pb_color = "#40c057" if status == "completed" else "#228be6"
        line3 = f"""
        <div style="display: flex; align-items: center; margin-top: 6px;">
            <div style="flex-grow: 1; background-color: #e9ecef; border-radius: 4px; height: 8px; overflow: hidden; position: relative;">
                <div style="width: {percent:.1f}%; background-color: {pb_color}; height: 100%; transition: width 0.1s ease-in-out;"></div>
            </div>
            <span style="font-size: 11px; font-weight: bold; color: #495057; margin-left: 8px; min-width: 45px; text-align: right;">{percent:.1f}%</span>
        </div>
        """

        self.train_status_widget.value = f"""
        <div class="train-status" style="font-size: 11px; color: #495057; font-family: monospace; border-top: 1px solid #dee2e6; margin-top: 4px; padding-top: 4px; line-height: 1.6;">
            <div>{line1}</div>
            <div>{line2}</div>
            {line3}
        </div>
        """
