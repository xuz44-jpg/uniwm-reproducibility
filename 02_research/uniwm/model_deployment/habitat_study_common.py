"""Geometry shared by the small, fixed Matterport development study."""
import math
import numpy as np


def wrap(theta):
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def relative_action(first, second):
    """Meters forward/left in the first camera's level body frame, radians left."""
    x, y, yaw = first
    delta = np.asarray(second[:2]) - [x, y]
    c, s = math.cos(yaw), math.sin(yaw)
    return [float(c * delta[0] + s * delta[1]),
            float(-s * delta[0] + c * delta[1]), float(wrap(second[2] - yaw))]


def proposed_pose(pose, action):
    x, y, yaw = pose
    dx, dy, dyaw = action
    c, s = math.cos(yaw), math.sin(yaw)
    return [x + c * dx - s * dy, y + s * dx + c * dy, float(wrap(yaw + dyaw))]


def yaw_rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def project(points, rotation, K):
    rays = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(K).T
    rotated = np.einsum('ni,ji->nj', rays, rotation)
    pixels = np.einsum('ni,ji->nj', rotated, K)
    return pixels[:, :2] / pixels[:, 2:], rotated[:, 2] > 0


def scaled_K(K, width, height, new_width=256, new_height=256):
    """PIL/OpenCV pixel-center resize convention, no crop."""
    A = np.array([[new_width / width, 0., .5 * new_width / width - .5],
                  [0., new_height / height, .5 * new_height / height - .5],
                  [0., 0., 1.]])
    return A @ np.asarray(K)


def case_name(degrees):
    return 'yaw_zero' if degrees == 0 else f'yaw_{"left" if degrees > 0 else "right"}_{abs(int(degrees)):02d}'
