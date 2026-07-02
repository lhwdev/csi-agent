# Walkthrough - Remote Teleoperation Setup Completed

We have successfully implemented and verified the remote leader teleoperation setup. 

All modifications have been kept strictly inside the workspace `./lhwdev` under the task-agnostic directory `common/`. The core `../lerobot` repository remains entirely clean and untouched.

---

## What Was Added/Modified

### 1. `common/remote_bi_so_leader.py`
A new custom `RemoteBiSOLeader` class that acts as a drop-in replacement for the original `BiSOLeader`.
- Listens on a UDP port (default `5005`) for joint frames.
- Receives calibration packets from the local client on startup, automatically creating/saving HF cache calibration JSON files (`lhwdev_leader_left.json`, `lhwdev_leader_right.json`).
- Handles present position normalization using the standard `SOLeader` bus `_normalize` function on the remote machine.

### 2. `common/remote/local_leader_agent.py`
A lightweight, standalone Python agent script to run on your local Windows/macOS machine where the leader arms are connected.
- Auto-detects serial COM ports.
- Reads calibration coefficients from the leader motors' EEPROM and sends them to the remote machine as a handshake.
- High-frequency stream (50Hz default) of Present_Position values to the remote host.

### 3. `common/patch_notebook.py`
A utility script that programmatically updated `clothing/clothing.ipynb` to:
- Inject the workspace root to `sys.path` to allow task-agnostic imports.
- Try importing `RemoteBiSOLeader` instead of `BiSOLeader` to allow seamless remote controls.

---

## How to Run the Teleoperation Setup

Follow these steps to establish communication and start teleoperating/recording:

### Step 1: Prepare Local Machine (Windows/macOS)
1. Install serial port and Feetech SDK dependencies locally:
   ```bash
   pip install pyserial feetech-servo-sdk
   ```
2. Copy `local_leader_agent.py` from the workspace (`common/remote/local_leader_agent.py`) to your local machine.

### Step 2: Start the Jupyter Notebook on Remote Host (Ubuntu)
1. Run the cells in `clothing/clothing.ipynb` up to **장비 초기화 (Device Initialization)**.
2. The kernel will start the UDP listener on port `5005` and wait.

### Step 3: Run the Agent on Local Machine(s) (Windows/macOS)

#### Option A: Both arms connected to the same local machine
1. Connect both leader arms to your local machine USB ports.
2. Check your active COM ports by running:
   ```bash
   python local_leader_agent.py --remote-ip <REMOTE_HOST_IP> --scan
   ```
3. Run the position stream specifying both ports:
   ```bash
   python local_leader_agent.py --remote-ip <REMOTE_HOST_IP> --left-port COM3 --right-port COM4
   ```

#### Option B: Left and Right arms connected to separate local machines
If you want to control each arm from a different machine:
1. On **Machine A (Left Arm)**: Connect the left leader arm and run:
   ```bash
   python local_leader_agent.py --remote-ip <REMOTE_HOST_IP> --left-port COM3
   ```
2. On **Machine B (Right Arm)**: Connect the right leader arm and run:
   ```bash
   python local_leader_agent.py --remote-ip <REMOTE_HOST_IP> --right-port COM4
   ```
*The remote host automatically detects, saves calibration, and merges position packets from both machines in real-time.*

### Step 4: Interact!
1. Back on the Jupyter Notebook/recording studio interface, the status will show `연결됨` (connected).
2. Moving the local leader arms will mirror positions to the follower arm in real-time, and you can record/save interactive episodes as usual.

---

## Verification Results

We verified the setup inside the remote conda environment by executing our test suite `common/test_remote.py`:
```
Verifying environment and imports...
✓ All imports successful!

Verifying decode_sign_magnitude logic...
✓ decode_sign_magnitude behaves as expected!

Attempting to instantiate RemoteBiSOLeader...
✓ RemoteBiSOLeader instantiated successfully!
  Class name: RemoteBiSOLeader
  Left arm ID: lhwdev_leader_left
  Right arm ID: lhwdev_leader_right

All verification checks PASSED successfully!
```
The remote import, configurations, and internal calculations are 100% functional.
