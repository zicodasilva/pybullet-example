import pybullet as p
import pickle
import numpy as np
from typing import Dict
import time
import pybullet_data

# Indices from the FTE output data and the corresponding joint in URDF data -- [acinoset_idx, urdf_idx].
base_x = [0, 0]
base_y = [1, 0]
base_z = [2, 0]
base_roll = [3, 0]
base_pitch = [4, 0]
base_yaw = [5, 0]
bodyF_roll = [6, 0]
bodyF_pitch = [7, 3]
bodyF_yaw = [8, 2]
neck_roll = [9, 4]
neck_pitch = [10, 7]
neck_yaw = [11, 6]
tail1_pitch = [12, 8]
tail1_yaw = [13, 10]
tail2_pitch = [14, 11]
tail2_yaw = [15, 13]
ufl_pitch = [16, 14]
lfl_pitch = [17, 15]
hfl_pitch = [18, 16]
ufr_pitch = [19, 18]
lfr_pitch = [20, 19]
hfr_pitch = [21, 20]
ubl_pitch = [22, 22]
lbl_pitch = [23, 23]
hbl_pitch = [26, 24]
ubr_pitch = [24, 26]
lbr_pitch = [25, 27]
hbr_pitch = [27, 28]

def load_pickle(filename: str) -> Dict:
    """Reads data from a pickle file.

    Args:
        filename: Full path to the pickle file.

    Returns:
        Read data into a dictionary.
    """
    with open(filename, "rb") as handle:
        return pickle.load(handle)

# Load data.
data = load_pickle("fte.pickle")
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.setGravity(0, 0, -9.81)
# planeId = p.loadURDF("plane.urdf")
robot = p.loadURDF("cheetah.urdf", flags=p.URDF_MAINTAIN_LINK_ORDER)
print(f"# of Joints in robot: {p.getNumJoints(robot)}")

# Initialise data.
N = data["x"].shape[0]
n_pose_vars = data["x"].shape[1]
frame_id = p.addUserDebugParameter("frame", 0, N - 1, 0)
run_mode_id = p.addUserDebugParameter("run", 0, 1, 0)
for j in range(p.getNumJoints(robot)):
    joint_info = p.getJointInfo(robot, j)
    p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, targetVelocity=0.0, force=10.0)

frame_raw = -1
while (p.isConnected()):
    # Frame management.
    run_mode = p.readUserDebugParameter(run_mode_id)
    if run_mode:
        frame_raw += 1
        frame_raw %= N
    else:
        frame_raw = p.readUserDebugParameter(frame_id)
    frame = int(frame_raw)
    frame_next = frame + 1
    if (frame_next >= N):
        frame_next = frame
    frame_residual = frame_raw - frame

    # Purely simulate the robot forward from previously calculated values.
    pos_base_1 = [data["x"][frame, base_x[0]], data["x"][frame, base_y[0]], data["x"][frame, base_z[0]]]
    pos_base_2 = [data["x"][frame_next, base_x[0]], data["x"][frame_next, base_y[0]], data["x"][frame_next, base_z[0]]]
    pos_base = [
      pos_base_1[0] + frame_residual * (pos_base_2[0] - pos_base_1[0]),
      pos_base_1[1] + frame_residual * (pos_base_2[1] - pos_base_1[1]),
      pos_base_1[2] + frame_residual * (pos_base_2[2] - pos_base_1[2])
    ]
    rot_base_1 = [data["x"][frame, base_roll[0]], data["x"][frame, base_pitch[0]], np.pi - data["x"][frame, base_yaw[0]]]
    rot_base_2 = [data["x"][frame_next, base_roll[0]], data["x"][frame_next, base_pitch[0]], np.pi - data["x"][frame_next, base_yaw[0]]]
    rot_base_euler = [
      rot_base_1[0] + frame_residual * (rot_base_2[0] - rot_base_1[0]),
      rot_base_1[1] + frame_residual * (rot_base_2[1] - rot_base_1[1]),
      rot_base_1[2] + frame_residual * (rot_base_2[2] - rot_base_1[2])
    ]
    # rot_base = p.getQuaternionSlerp(p.getQuaternionFromEuler([rot_base_1[0], rot_base_1[1], rot_base_1[2]]), p.getQuaternionFromEuler([rot_base_2[0], rot_base_2[1], rot_base_2[2]]), frame_residual)
    p.resetBasePositionAndOrientation(robot, pos_base, p.getQuaternionFromEuler(rot_base_euler))
    # p.resetBasePositionAndOrientation(robot, pos_base, rot_base)

    rot_bodyF_roll = data["x"][frame, bodyF_roll[0]] + frame_residual * (data["x"][frame_next, bodyF_roll[0]] - data["x"][frame, bodyF_roll[0]])
    p.resetJointState(robot, bodyF_roll[1], rot_bodyF_roll)

    rot_bodyF_pitch = data["x"][frame, bodyF_pitch[0]] + frame_residual * (data["x"][frame_next, bodyF_pitch[0]] - data["x"][frame, bodyF_pitch[0]])
    p.resetJointState(robot, bodyF_pitch[1], rot_bodyF_pitch)

    rot_bodyF_yaw = data["x"][frame, bodyF_yaw[0]] + frame_residual * (data["x"][frame_next, bodyF_yaw[0]] - data["x"][frame, bodyF_yaw[0]])
    p.resetJointState(robot, bodyF_yaw[1], rot_bodyF_yaw)

    rot_tail1_pitch = data["x"][frame, tail1_pitch[0]] + frame_residual * (data["x"][frame_next, tail1_pitch[0]] - data["x"][frame, tail1_pitch[0]])
    p.resetJointState(robot, tail1_pitch[1], rot_tail1_pitch)

    rot_tail1_yaw = data["x"][frame, tail1_yaw[0]] + frame_residual * (data["x"][frame_next, tail1_yaw[0]] - data["x"][frame, tail1_yaw[0]])
    p.resetJointState(robot, tail1_yaw[1], rot_tail1_yaw)

    rot_tail2_pitch = data["x"][frame, tail2_pitch[0]] + frame_residual * (data["x"][frame_next, tail2_pitch[0]] - data["x"][frame, tail2_pitch[0]])
    p.resetJointState(robot, tail2_pitch[1], rot_tail2_pitch)

    rot_tail2_yaw = data["x"][frame, tail2_yaw[0]] + frame_residual * (data["x"][frame_next, tail2_yaw[0]] - data["x"][frame, tail2_yaw[0]])
    p.resetJointState(robot, tail2_yaw[1], rot_tail2_yaw)

    rot_ufl_pitch = data["x"][frame, ufl_pitch[0]] + frame_residual * (data["x"][frame_next, ufl_pitch[0]] - data["x"][frame, ufl_pitch[0]])
    p.resetJointState(robot, ufl_pitch[1], rot_ufl_pitch)

    rot_lfl_pitch = data["x"][frame, lfl_pitch[0]] + frame_residual * (data["x"][frame_next, lfl_pitch[0]] - data["x"][frame, lfl_pitch[0]])
    p.resetJointState(robot, lfl_pitch[1], rot_lfl_pitch)

    rot_hfl_pitch = data["x"][frame, hfl_pitch[0]] + frame_residual * (data["x"][frame_next, hfl_pitch[0]] - data["x"][frame, hfl_pitch[0]])
    p.resetJointState(robot, hfl_pitch[1], rot_hfl_pitch)

    rot_ufr_pitch = data["x"][frame, ufr_pitch[0]] + frame_residual * (data["x"][frame_next, ufr_pitch[0]] - data["x"][frame, ufr_pitch[0]])
    p.resetJointState(robot, ufr_pitch[1], rot_ufr_pitch)

    rot_lfr_pitch = data["x"][frame, lfr_pitch[0]] + frame_residual * (data["x"][frame_next, lfr_pitch[0]] - data["x"][frame, lfr_pitch[0]])
    p.resetJointState(robot, lfr_pitch[1], rot_lfr_pitch)

    rot_hfr_pitch = data["x"][frame, hfr_pitch[0]] + frame_residual * (data["x"][frame_next, hfr_pitch[0]] - data["x"][frame, hfr_pitch[0]])
    p.resetJointState(robot, hfr_pitch[1], rot_hfr_pitch)

    rot_ubl_pitch = data["x"][frame, ubl_pitch[0]] + frame_residual * (data["x"][frame_next, ubl_pitch[0]] - data["x"][frame, ubl_pitch[0]])
    p.resetJointState(robot, ubl_pitch[1], rot_ubl_pitch)

    rot_lbl_pitch = data["x"][frame, lbl_pitch[0]] + frame_residual * (data["x"][frame_next, lbl_pitch[0]] - data["x"][frame, lbl_pitch[0]])
    p.resetJointState(robot, lbl_pitch[1], rot_lbl_pitch)

    rot_hbl_pitch = data["x"][frame, hbl_pitch[0]] + frame_residual * (data["x"][frame_next, hbl_pitch[0]] - data["x"][frame, hbl_pitch[0]])
    p.resetJointState(robot, hbl_pitch[1], rot_hbl_pitch)

    rot_ubr_pitch = data["x"][frame, ubr_pitch[0]] + frame_residual * (data["x"][frame_next, ubr_pitch[0]] - data["x"][frame, ubr_pitch[0]])
    p.resetJointState(robot, ubr_pitch[1], rot_ubr_pitch)

    rot_lbr_pitch = data["x"][frame, lbr_pitch[0]] + frame_residual * (data["x"][frame_next, lbr_pitch[0]] - data["x"][frame, lbr_pitch[0]])
    p.resetJointState(robot, lbr_pitch[1], rot_lbr_pitch)

    rot_hbr_pitch = data["x"][frame, hbr_pitch[0]] + frame_residual * (data["x"][frame_next, hbr_pitch[0]] - data["x"][frame, hbr_pitch[0]])
    p.resetJointState(robot, hbr_pitch[1], rot_hbr_pitch)

    p.stepSimulation()

    time.sleep(1.0 / 240.0)
p.disconnect()
