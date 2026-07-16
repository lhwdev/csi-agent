from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig
from lerobot.robots.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

left_cam_config = OpenCVCameraConfig(
    index_or_path="/dev/lerobot/camera_1",
    fps=30,
    width=160,
    height=120,
    fourcc="MJPG",
)
right_cam_config = OpenCVCameraConfig(
    index_or_path="/dev/lerobot/camera_2",
    fps=30,
    width=160,
    height=120,
    fourcc="MJPG",
)
top_cam_config = OpenCVCameraConfig(
    index_or_path="/dev/lerobot/camera_0",
    fps=30,
    width=640,
    height=480,
    fourcc="MJPG",
)

left_follower_config = SOFollowerConfig(
    port="/dev/lerobot/follower_1",
    cameras={"cam": left_cam_config},
)
right_follower_config = SOFollowerConfig(
    port="/dev/lerobot/follower_2",
    cameras={"cam": right_cam_config},
)

robot_config = BiSOFollowerConfig(
    id="lhwdev_follower_bimanual",
    left_arm_config=left_follower_config,
    right_arm_config=right_follower_config,
    cameras={"top": top_cam_config},
)

# Configure leader
left_leader_config = SOLeaderConfig(
    port="/dev/lerobot/leader_1",
)
right_leader_config = SOLeaderConfig(
    port="/dev/lerobot/leader_2",
)

leader_config = BiSOLeaderConfig(
    id="lhwdev_leader",
    left_arm_config=left_leader_config,
    right_arm_config=right_leader_config,
)

robot = BiSOFollower(robot_config)
leader = BiSOLeader(leader_config)

robot.connect()
leader.connect()
