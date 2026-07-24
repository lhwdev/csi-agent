This file provides guidance to AI agents when working with code in this workspace.

## Project

This project aims to use bi_so_follower(two so101_follower) and bi_so_leader(two so101_leader)
robots + cv2 cameras (top, left, right) to achieve VLA-based imitation learning.

- **clothing**: located at `./clothing`. Aims to unfold/fold clothes.

## Environment

All commands are automatically run inside conda lerobot environment.

Refer to `../lerobot/AGENT_GUIDE.md` for more details about lerobot codebase.

## Workspaces

- primary coding workspace: `./`. Contains clothing project.
- test codes: `test`, such as `./clothing/test`. All verification codes should go under this
  directory and may be persisted after implementation.
- lerobot repository: `../lerobot`. Do NOT modify this, unless you are sure that lerobot codebase
  itself contains bug or problem or you are asked to.

## Tooling

- `*.ipynb` Jupyteer Notebook files: Use `notebook` MCP to modify.
