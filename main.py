import pybullet as p
import pickle
import numpy as np
from typing import Dict, Literal
import time
import pybullet_data

# Indices from the FTE output data and the corresponding joint in URDF data -- [acinoset_idx, urdf_idx].
base_x = [0, 0]
base_y = [1, 0]
base_z = [2, 0]
base_roll = [3, 0]
base_pitch = [4, 0]
base_yaw = [5, 0]
bodyF_roll = [6, 3]
bodyF_pitch = [7, 2]
bodyF_yaw = [8, 1]
neck_roll = [9, 7]
neck_pitch = [10, 6]
neck_yaw = [11, 5]
tail1_pitch = [12, 18]
tail1_yaw = [13, 17]
tail2_pitch = [14, 21]
tail2_yaw = [15, 20]
ufl_pitch = [16, 8]
lfl_pitch = [17, 9]
hfl_pitch = [18, 10]
ufr_pitch = [19, 12]
lfr_pitch = [20, 13]
hfr_pitch = [21, 14]
ubl_pitch = [22, 22]
lbl_pitch = [23, 23]
hbl_pitch = [26, 24]
ubr_pitch = [24, 26]
lbr_pitch = [25, 27]
hbr_pitch = [27, 28]

def get_state(data: Dict, i: int, i_1: int, delta_i: float, state: Literal["x", "dx"]):
    base_1 = [data[state][i, base_x[0]], data[state][i, base_y[0]], data[state][i, base_z[0]]]
    base_2 = [data[state][i_1, base_x[0]], data[state][i_1, base_y[0]], data[state][i_1, base_z[0]]]
    base = [
      base_1[0] + delta_i * (base_2[0] - base_1[0]),
      base_1[1] + delta_i * (base_2[1] - base_1[1]),
      base_1[2] + delta_i * (base_2[2] - base_1[2])
    ]
    rot_base_1 = [data[state][i, base_roll[0]], data[state][i, base_pitch[0]], data[state][i, base_yaw[0]]]
    rot_base_2 = [data[state][i_1, base_roll[0]], data[state][i_1, base_pitch[0]], data[state][i_1, base_yaw[0]]]
    rot_base = [
      rot_base_1[0] + delta_i * (rot_base_2[0] - rot_base_1[0]),
      rot_base_1[1] + delta_i * (rot_base_2[1] - rot_base_1[1]),
      rot_base_1[2] + delta_i * (rot_base_2[2] - rot_base_1[2])
    ]
    rot_bodyF_roll = data[state][i, bodyF_roll[0]] + delta_i * (data[state][i_1, bodyF_roll[0]] - data[state][i, bodyF_roll[0]])
    rot_bodyF_pitch = data[state][i, bodyF_pitch[0]] + delta_i * (data[state][i_1, bodyF_pitch[0]] - data[state][i, bodyF_pitch[0]])
    rot_bodyF_yaw = data[state][i, bodyF_yaw[0]] + delta_i * (data[state][i_1, bodyF_yaw[0]] - data[state][i, bodyF_yaw[0]])
    rot_tail1_pitch = data[state][i, tail1_pitch[0]] + delta_i * (data[state][i_1, tail1_pitch[0]] - data[state][i, tail1_pitch[0]])
    rot_tail1_yaw = data[state][i, tail1_yaw[0]] + delta_i * (data[state][i_1, tail1_yaw[0]] - data[state][i, tail1_yaw[0]])
    rot_tail2_pitch = data[state][i, tail2_pitch[0]] + delta_i * (data[state][i_1, tail2_pitch[0]] - data[state][i, tail2_pitch[0]])
    rot_tail2_yaw = data[state][i, tail2_yaw[0]] + delta_i * (data[state][i_1, tail2_yaw[0]] - data[state][i, tail2_yaw[0]])
    rot_ufl_pitch = data[state][i, ufl_pitch[0]] + delta_i * (data[state][i_1, ufl_pitch[0]] - data[state][i, ufl_pitch[0]])
    rot_lfl_pitch = data[state][i, lfl_pitch[0]] + delta_i * (data[state][i_1, lfl_pitch[0]] - data[state][i, lfl_pitch[0]])
    rot_hfl_pitch = data[state][i, hfl_pitch[0]] + delta_i * (data[state][i_1, hfl_pitch[0]] - data[state][i, hfl_pitch[0]])
    rot_ufr_pitch = data[state][i, ufr_pitch[0]] + delta_i * (data[state][i_1, ufr_pitch[0]] - data[state][i, ufr_pitch[0]])
    rot_lfr_pitch = data[state][i, lfr_pitch[0]] + delta_i * (data[state][i_1, lfr_pitch[0]] - data[state][i, lfr_pitch[0]])
    rot_hfr_pitch = data[state][i, hfr_pitch[0]] + delta_i * (data[state][i_1, hfr_pitch[0]] - data[state][i, hfr_pitch[0]])
    rot_ubl_pitch = data[state][i, ubl_pitch[0]] + delta_i * (data[state][i_1, ubl_pitch[0]] - data[state][i, ubl_pitch[0]])
    rot_lbl_pitch = data[state][i, lbl_pitch[0]] + delta_i * (data[state][i_1, lbl_pitch[0]] - data[state][i, lbl_pitch[0]])
    rot_hbl_pitch = data[state][i, hbl_pitch[0]] + delta_i * (data[state][i_1, hbl_pitch[0]] - data[state][i, hbl_pitch[0]])
    rot_ubr_pitch = data[state][i, ubr_pitch[0]] + delta_i * (data[state][i_1, ubr_pitch[0]] - data[state][i, ubr_pitch[0]])
    rot_lbr_pitch = data[state][i, lbr_pitch[0]] + delta_i * (data[state][i_1, lbr_pitch[0]] - data[state][i, lbr_pitch[0]])
    rot_hbr_pitch = data[state][i, hbr_pitch[0]] + delta_i * (data[state][i_1, hbr_pitch[0]] - data[state][i, hbr_pitch[0]])

    return [base, rot_base, rot_bodyF_roll, rot_bodyF_pitch, rot_bodyF_yaw,
            rot_tail1_pitch, rot_tail1_yaw, rot_tail2_pitch, rot_tail2_yaw, rot_ufl_pitch,
            rot_lfl_pitch, rot_hfl_pitch, rot_ufr_pitch, rot_lfr_pitch, rot_hfr_pitch, rot_ubl_pitch,
            rot_lbl_pitch, rot_hbl_pitch, rot_ubr_pitch, rot_lbr_pitch, rot_hbr_pitch]

def reset_joint_state(pos, vel):
    p.resetJointState(robot, bodyF_roll[1], pos[2], vel[2])
    p.resetJointState(robot, bodyF_pitch[1], pos[3], vel[3])
    p.resetJointState(robot, bodyF_yaw[1], pos[4], vel[4])
    p.resetJointState(robot, tail1_pitch[1], pos[5], vel[5])
    p.resetJointState(robot, tail1_yaw[1], pos[6], vel[6])
    p.resetJointState(robot, tail2_pitch[1], pos[7], vel[7])
    p.resetJointState(robot, tail2_yaw[1], pos[8], vel[8])
    p.resetJointState(robot, ufl_pitch[1], pos[9], vel[9])
    p.resetJointState(robot, lfl_pitch[1], pos[10], vel[10])
    p.resetJointState(robot, hfl_pitch[1], pos[11], vel[11])
    p.resetJointState(robot, ufr_pitch[1], pos[12], vel[12])
    p.resetJointState(robot, lfr_pitch[1], pos[13], vel[13])
    p.resetJointState(robot, hfr_pitch[1], pos[14], vel[14])
    p.resetJointState(robot, ubl_pitch[1], pos[15], vel[15])
    p.resetJointState(robot, lbl_pitch[1], pos[16], vel[16])
    p.resetJointState(robot, hbl_pitch[1], pos[17], vel[17])
    p.resetJointState(robot, ubr_pitch[1], pos[18], vel[18])
    p.resetJointState(robot, lbr_pitch[1], pos[19], vel[19])
    p.resetJointState(robot, hbr_pitch[1], pos[20], vel[20])

def set_joint_target(pos, gain: float, max_force: float):
    p.setJointMotorControl2(robot, bodyF_roll[1], p.POSITION_CONTROL, targetPosition=pos[2], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, bodyF_pitch[1], p.POSITION_CONTROL, targetPosition=pos[3], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, bodyF_yaw[1], p.POSITION_CONTROL, targetPosition=pos[4], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, tail1_pitch[1], p.POSITION_CONTROL, targetPosition=pos[5], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, tail1_yaw[1], p.POSITION_CONTROL, targetPosition=pos[6], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, tail2_pitch[1], p.POSITION_CONTROL, targetPosition=pos[7], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, tail2_yaw[1], p.POSITION_CONTROL, targetPosition=pos[8], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, ufl_pitch[1], p.POSITION_CONTROL, targetPosition=pos[9], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, lfl_pitch[1], p.POSITION_CONTROL, targetPosition=pos[10], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, hfl_pitch[1], p.POSITION_CONTROL, targetPosition=pos[11], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, ufr_pitch[1], p.POSITION_CONTROL, targetPosition=pos[12], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, lfr_pitch[1], p.POSITION_CONTROL, targetPosition=pos[13], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, hfr_pitch[1], p.POSITION_CONTROL, targetPosition=pos[14], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, ubl_pitch[1], p.POSITION_CONTROL, targetPosition=pos[15], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, lbl_pitch[1], p.POSITION_CONTROL, targetPosition=pos[16], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, hbl_pitch[1], p.POSITION_CONTROL, targetPosition=pos[17], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, ubr_pitch[1], p.POSITION_CONTROL, targetPosition=pos[18], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, lbr_pitch[1], p.POSITION_CONTROL, targetPosition=pos[19], positionGain=gain, force=max_force)
    p.setJointMotorControl2(robot, hbr_pitch[1], p.POSITION_CONTROL, targetPosition=pos[20], positionGain=gain, force=max_force)

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
physicsClient = p.connect(p.GUI, options="--mp4=\"test.mp4\" --mp4fps=240")
# physicsClient = p.connect(p.GUI, options="--mp4=\"test.mp4\" --mp4fps=240")
time_step = 1.0 / 240.0
# p.setPhysicsEngineParameter(fixedTimeStep=time_step)
robot = p.loadURDF("cheetah.urdf", useFixedBase=False)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
planeId = p.loadURDF("plane.urdf")
print(f"# of Joints in robot: {p.getNumJoints(robot)}")
reset_state = False
# Initialise data.
N = data["x"].shape[0]
n_pose_vars = data["x"].shape[1]
frame_id = p.addUserDebugParameter("frame", 0, N - 1, 0)
run_mode_id = p.addUserDebugParameter("run", 0, 1, 0)
# Init state of robot.
init_x = get_state(data, 0, 1, 0.0, "x")
init_dx = get_state(data, 0, 1, 0.0, "dx")
init_x[1][2] = np.pi - init_x[1][2]
p.resetBasePositionAndOrientation(robot, init_x[0], p.getQuaternionFromEuler(init_x[1]))
reset_joint_state(init_x, init_dx)
for j in range(p.getNumJoints(robot)):
    joint_info = p.getJointInfo(robot, j)
    if reset_state:
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, targetVelocity=0.0, force=10.0)

cid = p.createConstraint(robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
frame_raw = -1
frame_prev = -1
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
    if frame_prev == frame:
        time.sleep(time_step)
        continue

    x = get_state(data, frame, frame_next, frame_residual, "x")
    dx = get_state(data, frame, frame_next, frame_residual, "dx")
    # rot_base = p.getQuaternionSlerp(p.getQuaternionFromEuler([rot_base_1[0], rot_base_1[1], rot_base_1[2]]), p.getQuaternionFromEuler([rot_base_2[0], rot_base_2[1], rot_base_2[2]]), frame_residual)
    x[1][2] = np.pi - x[1][2]

    if frame_raw == 0:
        p.resetBasePositionAndOrientation(robot, x[0], p.getQuaternionFromEuler(x[1]))
        p.resetBaseVelocity(robot, dx[0], dx[1])
        # p.resetBasePositionAndOrientation(robot, pos_base, rot_base)
        reset_joint_state(x, dx)

    if reset_state:
        p.setGravity(0, 0, -9.81)
        p.resetBasePositionAndOrientation(robot, x[0], p.getQuaternionFromEuler(x[1]))
        p.resetBaseVelocity(robot, dx[0], dx[1])
        # p.resetBasePositionAndOrientation(robot, pos_base, rot_base)
        reset_joint_state(x, dx)
    else:
        set_joint_target(x, 0.3, 1000.0)
        p.changeConstraint(cid, jointChildPivot=x[0], jointChildFrameOrientation=p.getQuaternionFromEuler(x[1]), maxForce=1000.0)

    p.stepSimulation()

    time.sleep(time_step)
    frame_prev = frame
p.disconnect()
