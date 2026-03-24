# ================================================================
# Operational Space Control
# ================================================================
# Implementing operation space control in Cartesian space
# Supports position control, attitude control, zero-space control, 
# and gravity compensation.
# Task space control for SO100 robotic arm
# ================================================================

from typing import Optional, Tuple, Union

import mujoco
import numpy as np


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion.
    Args:
        mat: 3x3 rotation matrix

    Returns:
        quaternion [w, x, y, z]
    """
    trace = np.trace(mat)

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quat_diff_active(source_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    """Calculate the difference between the current quaternion and the target quaternion.
    Args:
        source_quat: current quaternion [w, x, y, z]
        target_quat: target quaternion [w, x, y, z]

    Returns:
        difference quaternion
    """
    # q_diff = q_target * q_source^(-1)
    source_conj = np.array([source_quat[0], -source_quat[1], -source_quat[2], -source_quat[3]])
    w1, x1, y1, z1 = target_quat
    w2, x2, y2, z2 = source_conj

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternions to axis-angle representation

    Args:
        quat: quaternion [w, x, y, z]

    Returns:
        Axis-angle vector (angle * axis)
    """
    w, x, y, z = quat

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-10:
        return np.zeros(3)

    w /= norm
    x /= norm
    y /= norm
    z /= norm

    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))

    sin_half_angle = np.sqrt(1.0 - w * w)

    if sin_half_angle < 1e-10:
        return np.array([angle * x, angle * y, angle * z])

    axis = np.array([x, y, z]) / sin_half_angle

    return angle * axis


def pd_control(
    x: np.ndarray,
    x_des: np.ndarray,
    dx: np.ndarray,
    kp_kv: np.ndarray,
    ddx_max: float = 0.0,
) -> np.ndarray:
    """PD controller

    Args:
        x: current pos
        x_des: target pos
        dx: current vel
        kp_kv: PD gain [kp, kd]
        ddx_max: acc limited

    Returns:
        command
    """
    x_err = x - x_des
    dx_err = dx

    x_err *= -kp_kv[:, 0]
    dx_err *= -kp_kv[:, 1]

    if ddx_max > 0.0:
        x_err_sq_norm = np.sum(x_err**2)
        ddx_max_sq = ddx_max**2
        if x_err_sq_norm > ddx_max_sq:
            x_err *= ddx_max / np.sqrt(x_err_sq_norm)

    return x_err + dx_err


def pd_control_orientation(
    quat: np.ndarray,
    quat_des: np.ndarray,
    w: np.ndarray,
    kp_kv: np.ndarray,
    dw_max: float = 0.0,
) -> np.ndarray:
    """Attitude PD controller low-pass filter

    Args:
        quat: current quaternion
        quat_des: target quaternino
        w: Current angular velocity
        kp_kv: PD gain [kp, kd]
        dw_max: Maximum angular acceleration limit

    Returns:
        Attitude control commands
    """
    quat_err = quat_diff_active(source_quat=quat_des, target_quat=quat)
    ori_err = quat_to_axisangle(quat_err)
    w_err = w

    ori_err *= -kp_kv[:, 0]
    w_err *= -kp_kv[:, 1]

    if dw_max > 0.0:
        ori_err_sq_norm = np.sum(ori_err**2)
        dw_max_sq = dw_max**2
        if ori_err_sq_norm > dw_max_sq:
            ori_err *= dw_max / np.sqrt(ori_err_sq_norm)

    return ori_err + w_err


def opspace(
    model,
    data,
    site_id,
    dof_ids: np.ndarray,
    pos: Optional[np.ndarray] = None,
    ori: Optional[np.ndarray] = None,
    joint: Optional[np.ndarray] = None,
    pos_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    ori_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    damping_ratio: float = 1.0,
    nullspace_stiffness: float = 0.5,
    max_pos_acceleration: Optional[float] = None,
    max_ori_acceleration: Optional[float] = None,
    gravity_comp: bool = True,
) -> np.ndarray:
    """main function"""
    x_des = data.site_xpos[site_id] if pos is None else np.asarray(pos)
    q_des = data.qpos[dof_ids] if joint is None else np.asarray(joint)

    control_ori = ori is not None

    kp_pos = np.asarray(pos_gains)
    kd_pos = damping_ratio * 2 * np.sqrt(kp_pos)
    kp_kv_pos = np.stack([kp_pos, kd_pos], axis=-1)

    kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
    kd_joint = damping_ratio * 2 * np.sqrt(kp_joint)
    kp_kv_joint = np.stack([kp_joint, kd_joint], axis=-1)

    ddx_max = max_pos_acceleration if max_pos_acceleration is not None else 0.0
    dw_max = max_ori_acceleration if max_ori_acceleration is not None else 0.0

    q = data.qpos[dof_ids]
    dq = data.qvel[dof_ids]
    J_v_full = np.zeros((3, model.nv), dtype=np.float64)
    J_w_full = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, J_v_full, J_w_full, site_id)

    J_v = J_v_full[:, dof_ids]
    
    if control_ori:
        J_w = J_w_full[:, dof_ids]
        J = np.concatenate([J_v, J_w], axis=0) # 6 x N 
    else:
        J = J_v # 3 x N 

    x = data.site_xpos[site_id]
    dx = J_v @ dq
    ddx = pd_control(x=x, x_des=x_des, dx=dx, kp_kv=kp_kv_pos, ddx_max=ddx_max)

    if control_ori:
        kp_ori = np.asarray(ori_gains)
        kd_ori = damping_ratio * 2 * np.sqrt(kp_ori)
        kp_kv_ori = np.stack([kp_ori, kd_ori], axis=-1)
        
        ori_target = np.asarray(ori)
        quat_des = mat_to_quat(ori_target) if ori_target.shape == (3, 3) else ori_target
        quat = mat_to_quat(data.site_xmat[site_id].reshape((3, 3)))
        
        if quat @ quat_des < 0.0:
            quat *= -1.0
            
        w = J_w @ dq
        dw = pd_control_orientation(quat=quat, quat_des=quat_des, w=w, kp_kv=kp_kv_ori, dw_max=dw_max)
        ddx_dw = np.concatenate([ddx, dw], axis=0)
    else:
        ddx_dw = ddx

    M_full = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M_full, data.qM)
    M = M_full[dof_ids, :][:, dof_ids]

    M_inv = np.linalg.inv(M)
    
    Mx_inv = J @ M_inv @ J.T
    Mx = np.linalg.inv(Mx_inv) if abs(np.linalg.det(Mx_inv)) >= 1e-2 else np.linalg.pinv(Mx_inv, rcond=1e-2)

    tau = J.T @ Mx @ ddx_dw

    ddq = pd_control(x=q, x_des=q_des, dx=dq, kp_kv=kp_kv_joint, ddx_max=0.0)
    
    Jnull = M_inv @ J.T @ Mx
    tau += (np.eye(len(q)) - J.T @ Jnull.T) @ (M @ ddq)

    if gravity_comp:
        tau += data.qfrc_bias[dof_ids]
    
    return tau
