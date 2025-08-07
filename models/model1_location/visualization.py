import cv2, numpy as np, torch
from kornia.geometry.transform import warp_affine


# colors in BGR format
COLOR_JOINT = (0, 255, 0)
COLOR_HEAD  = (255, 255, 0)
COLOR_BODY  = (255, 255, 0)
COLOR_RHAND = (255, 0, 0)
COLOR_LHAND = (0, 255, 0)
COLOR_RLEG  = (255, 0, 0)
COLOR_LLEG  = (0, 255, 0)

# joint index to name mapping (MPII)
MPII_ID2LABEL = {
    0 : 'RAnkle',
    1 : 'RKnee',
    2 : 'RHip',
    3 : 'LHip',
    4 : 'LKnee',
    5 : 'LAnkle',
    6 : 'Pelvis',
    7 : 'Thorax',
    8 : 'UNeck',
    9 : 'THead',
    10: 'RWrist',
    11: 'RElbow',
    12: 'RShoulder',
    13: 'LShoulder',
    14: 'LElbow',
    15: 'LWrist'
}

# joint connections (MPII)
MPII_LIMBS = (
    (6, 7), (7, 8), (8, 9),
    (6, 2), (2, 1), (1, 0), (6, 3), (3, 4), (4, 5),
    (7, 12), (12, 11), (11, 10), (7, 13), (13, 14), (14, 15)
)

# joint connection colors (MPII)
MPII_COLORS = (
    COLOR_BODY, COLOR_BODY, COLOR_HEAD,
    COLOR_RLEG, COLOR_RLEG, COLOR_RLEG, COLOR_LLEG, COLOR_LLEG, COLOR_LLEG,
    COLOR_RHAND, COLOR_RHAND, COLOR_RHAND, COLOR_LHAND, COLOR_LHAND, COLOR_LHAND
)

# joint index to name mapping (COCO)
COCO_ID2LABEL = {
    0 : 'Nose',
    1 : 'Neck',
    2 : 'RShoulder',
    3 : 'RElbow',
    4 : 'RWrist',
    5 : 'LShoulder',
    6 : 'LElbow',
    7 : 'LWrist',
    8 : 'RHip',
    9 : 'RKnee',
    10: 'RAnkle',
    11: 'LHip',
    12: 'LKnee',
    13: 'LAnkle',
    14: 'REye',
    15: 'LEye',
    16: 'REar',
    17: 'LEar'
}

# joint connections (COCO)
COCO_LIMBS = (
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)
)

# joint connection colors (COCO)
COCO_COLORS = (
    COLOR_HEAD, COLOR_HEAD, COLOR_HEAD, COLOR_HEAD, COLOR_HEAD,
    COLOR_RLEG, COLOR_RLEG, COLOR_RLEG, COLOR_LLEG, COLOR_LLEG, COLOR_LLEG,
    COLOR_RHAND, COLOR_RHAND, COLOR_RHAND, COLOR_LHAND, COLOR_LHAND, COLOR_LHAND
)


# draw pose
def draw_pose(keypoints, radius=0, thickness=4, alpha=1.0, bg=None, bgfill=0):
    joints = np.int32(keypoints).reshape(-1, 2)
    if joints.shape[0] == len(MPII_ID2LABEL):
        limbs = MPII_LIMBS
        colors = MPII_COLORS
    elif joints.shape[0] == len(COCO_ID2LABEL):
        limbs = COCO_LIMBS
        colors = COCO_COLORS
    else:
        return
    dx = 0
    dy = 0
    if bg is None:
        xcoords = joints[np.where(joints[:, 0] >= 0), 0]
        ycoords = joints[np.where(joints[:, 1] >= 0), 1]
        xmin, xmax = xcoords.min(), xcoords.max()
        ymin, ymax = ycoords.min(), ycoords.max()
        h = ymax - ymin + 4 * max(radius, thickness)
        w = xmax - xmin + 4 * max(radius, thickness)
        dx = xmin - 2 * max(radius, thickness)
        dy = ymin - 2 * max(radius, thickness)
        bg = np.ones((h, w, 3)) * bgfill
    bg = np.uint8(bg)
    overlay = bg.copy()
    for (i, j), color in zip(limbs, colors):
        (x1, y1), (x2, y2) = joints[i], joints[j]
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue
        cv2.line(overlay, (x1-dx, y1-dy), (x2-dx, y2-dy), color, thickness, cv2.LINE_AA)
    for x, y in joints:
        if x < 0 or y < 0:
            continue
        cv2.circle(overlay, (x-dx, y-dy), radius, COLOR_JOINT, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, bg, 1.0 - alpha, 0)


# warp bounding box tensor
def warp_bbox(affine_matrix, size):
    affn = affine_matrix * torch.tensor([[[1., 1., size[1]], [1., 1., size[0]]]])
    bbox = torch.ones(affn.size(0), 1, size[0], size[1])
    return warp_affine(bbox, affn, size)


# draw bounding box from affine matrix
def draw_bbox_affine(affine_matrix, bg, color=(0, 0, 0), alpha=1.0, denormalize=False):
    affn = torch.tensor(affine_matrix) if not isinstance(affine_matrix, torch.Tensor) else affine_matrix.detach().cpu()
    bbox = warp_bbox(affn, bg.shape[2:])
    mask = torch.where(bbox > 0., 1., 0.).repeat(1, 3, 1, 1).long()
    bbox = mask * torch.tensor(color).reshape(1, -1, 1, 1) / 255.
    bgnd = (bg + 1.) / 2. if denormalize else bg
    bbox = mask * bbox + (1 - mask) * bgnd
    return alpha * bbox + (1. - alpha) * bgnd


# draw bounding box
def draw_bbox(bbox, bg, color=(0, 0, 0), alpha=1.0, denormalize=False):
    bgnd = ((bg + 1.) / 2.) * 255. if denormalize else bg
    bgnd = np.uint8(bgnd)
    bbox = cv2.rectangle(bgnd.copy(), np.int32(bbox), color, -1, cv2.LINE_AA)
    return alpha * bbox + (1. - alpha) * bgnd


# draw point
def draw_point(point, bg, radius=4, color=(0, 0, 0), border_thickness=1, border_color=(255, 255, 255), denormalize=False):
    bgnd = ((bg + 1.) / 2.) * 255. if denormalize else bg
    bgnd = np.uint8(bgnd)
    cv2.circle(bgnd, np.int32(point), radius+border_thickness, border_color, -1, cv2.LINE_AA)
    cv2.circle(bgnd, np.int32(point), radius, color, -1, cv2.LINE_AA)
    return bgnd
