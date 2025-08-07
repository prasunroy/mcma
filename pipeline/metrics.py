import numpy as np


def compute_distance(a, b):
    return (((a - b) ** 2).sum(-1)) ** 0.5


def compute_head_size_mpii(keypoints):
    k = np.float32(keypoints.reshape(-1, 16, 2))
    return compute_distance(k[:, 8, :], k[:, 9, :])


def compute_body_size_mpii(keypoints):
    k = np.float32(keypoints.reshape(-1, 16, 2))
    l = (k[:, 3, :] + k[:, 13, :]) / 2
    r = (k[:, 2, :] + k[:, 12, :]) / 2
    return compute_distance(l, r)


def compute_pck_mpii(keypoints, reference, alpha=0.2):
    k = np.float32(keypoints.reshape(-1, 16, 2))
    r = np.float32(reference.reshape(-1, 16, 2))
    d = compute_distance(k, r)
    t = compute_body_size_mpii(r)
    p = d <= alpha * t.reshape(-1, 1)
    return p.sum() / p.size


def compute_pckh_mpii(keypoints, reference, alpha=0.5):
    k = np.float32(keypoints.reshape(-1, 16, 2))
    r = np.float32(reference.reshape(-1, 16, 2))
    d = compute_distance(k, r)
    t = compute_head_size_mpii(r)
    p = d <= alpha * t.reshape(-1, 1)
    return p.sum() / p.size


def compute_akd(keypoints, reference):
    k = np.float32(keypoints.reshape(-1, 2))
    r = np.float32(reference.reshape(-1, 2))
    d = compute_distance(k, r)
    return d.mean()


def compute_mae(keypoints, reference):
    k = keypoints.reshape(-1)
    r = reference.reshape(-1)
    d = abs(k - r)
    return d.mean()


def compute_mse(keypoints, reference):
    k = keypoints.reshape(-1)
    r = reference.reshape(-1)
    d = (k - r) ** 2
    return d.mean()


def compute_cosine_similarity(keypoints, reference):
    k = keypoints.reshape(-1, 2)
    r = reference.reshape(-1, 2)
    c = (k * r).sum(-1) / (np.linalg.norm(k, axis=-1) * np.linalg.norm(r, axis=-1))
    return c.mean()


def compute_bbox(keypoints):
    k = keypoints.reshape(keypoints.shape[0], -1, 2)
    x1y1 = k.min(1)
    x2y2 = k.max(1)
    return np.hstack((x1y1, x2y2))


def compute_iou(bboxA, bboxB):
    # reference: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection
    bboxA = bboxA.reshape(-1, 4)
    bboxB = bboxB.reshape(-1, 4)
    zeros = np.zeros(bboxA.shape[0], dtype=np.float32)
    xA = np.max((bboxA[:, 0], bboxB[:, 0]), axis=0)
    yA = np.max((bboxA[:, 1], bboxB[:, 1]), axis=0)
    xB = np.min((bboxA[:, 2], bboxB[:, 2]), axis=0)
    yB = np.min((bboxA[:, 3], bboxB[:, 3]), axis=0)
    interArea = np.max((zeros, xB - xA + 1), axis=0) * np.max((zeros, yB - yA + 1), axis=0)
    bboxAArea = (bboxA[:, 2] - bboxA[:, 0] + 1) * (bboxA[:, 3] - bboxA[:, 1] + 1)
    bboxBArea = (bboxB[:, 2] - bboxB[:, 0] + 1) * (bboxB[:, 3] - bboxB[:, 1] + 1)
    iou = interArea / (bboxAArea + bboxBArea - interArea)
    return iou.mean()


def rescale_keypoints(keypoints, from_size, to_size):
    k = keypoints.reshape(keypoints.shape[0], -1, 2)
    return k / (np.float32(from_size) - 1) * (np.float32(to_size) - 1)
