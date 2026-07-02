#!/usr/bin/env python3
import sys
from pathlib import Path

# Add workspace parent to path
parent_dir = str(Path(__file__).resolve().parents[1])
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("Verifying environment and imports...")
try:
    from common.remote_bi_so_leader import RemoteBiSOLeader
    from lerobot.teleoperators.so_leader import SOLeaderConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
    from lerobot.motors.encoding_utils import decode_sign_magnitude
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Import failure: {e}")
    sys.exit(1)

# Test decode_sign_magnitude behavior
print("\nVerifying decode_sign_magnitude logic...")
# 2048 with sign bit (bit 15) set: 2048 | (1 << 15) = 2048 | 32768 = 34816
# Unsigned 2048 -> direction bit 0 -> 2048
# Signed 2048 -> direction bit 1 -> -2048
try:
    val1 = decode_sign_magnitude(2048, 15)
    val2 = decode_sign_magnitude(34816, 15)
    assert val1 == 2048, f"Expected 2048, got {val1}"
    assert val2 == -2048, f"Expected -2048, got {val2}"
    print("✓ decode_sign_magnitude behaves as expected!")
except AssertionError as ae:
    print(f"✗ assertion failed: {ae}")
    sys.exit(1)

# Test instantiating RemoteBiSOLeader with fake/default config
print("\nAttempting to instantiate RemoteBiSOLeader...")
try:
    left_leader_config = SOLeaderConfig(port="/dev/lerobot/leader_1")
    right_leader_config = SOLeaderConfig(port="/dev/lerobot/leader_2")
    leader_config = BiSOLeaderConfig(
        id="lhwdev_leader",
        left_arm_config=left_leader_config,
        right_arm_config=right_leader_config,
    )
    leader = RemoteBiSOLeader(leader_config)
    print("✓ RemoteBiSOLeader instantiated successfully!")
    print(f"  Class name: {leader.__class__.__name__}")
    print(f"  Left arm ID: {leader.left_arm.id}")
    print(f"  Right arm ID: {leader.right_arm.id}")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    sys.exit(1)

print("\nAll verification checks PASSED successfully!")
