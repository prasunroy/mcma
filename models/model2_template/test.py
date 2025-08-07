import numpy as np, os, pandas as pd, torch
from PIL import Image
from data import create_dataloader
from model import TemplatePredictor
from visualization import draw_pose


# configurations
# -----------------------------------------------------------------------------
embeds_dim = 512
num_tposes = 30
enable_gpu = True
data_rootd = '../../datasets/binge_watching'
image_size = 256
batch_size = 8
output_dir = './result'
checkpoint_path = './output/model_epoch_001_improved.pth'
pose_templates = pd.read_csv('../../datasets/binge_watching/centers_30.txt', sep=' ', header=None).values.astype(np.float32)[:, :32]
# -----------------------------------------------------------------------------


# get inputs
def get_inputs(data):
    img = data['img']
    img_p1 = data['img_patch1']
    img_p2 = data['img_patch2']
    seg = data['seg']
    seg_p1 = data['seg_patch1']
    seg_p2 = data['seg_patch2']
    em = data['pose_embedding']
    if enable_gpu and torch.cuda.is_available():
        img, img_p1, img_p2, seg, seg_p1, seg_p2, em = img.cuda(torch.device(0)), img_p1.cuda(torch.device(0)), img_p2.cuda(torch.device(0)), seg.cuda(torch.device(0)), seg_p1.cuda(torch.device(0)), seg_p2.cuda(torch.device(0)), em.cuda(torch.device(0))
    return img, img_p1, img_p2, seg, seg_p1, seg_p2, em


def normalize_keypoints(keypoints):
    k = keypoints.copy().reshape(-1, 2)
    kx = k[np.where(k[:, 0] >= 0), 0]
    ky = k[np.where(k[:, 1] >= 0), 1]
    nx = (kx - kx.min()) / (kx.max() - kx.min())
    ny = (ky - ky.min()) / (ky.max() - ky.min())
    k[np.where(k[:, 0] >= 0), 0] = nx
    k[np.where(k[:, 1] >= 0), 1] = ny
    return np.float32(k)


def get_bbox_size(keypoints):
    k = keypoints.reshape(-1, 2)
    kx = k[np.where(k[:, 0] >= 0), 0]
    ky = k[np.where(k[:, 1] >= 0), 1]
    dx = kx.max() - kx.min()
    dy = ky.max() - ky.min()
    return np.float32((dx, dy))


def get_bbox_anchor(keypoints):
    k = keypoints.reshape(-1, 2)
    kx = k[np.where(k[:, 0] >= 0), 0]
    ky = k[np.where(k[:, 1] >= 0), 1]
    return np.float32((kx.min(), ky.min()))


# visualize
def visualize(data, pred, index, save_dir):
    im_files = data['img_path']
    keypoints = data['keypoints']
    pose_classes = data['pose_class'].cpu()
    pred_classes = pred.detach().cpu().argmax(dim=1)
    for i in range(len(pred)):
        size = get_bbox_size(keypoints[i])
        offset = get_bbox_anchor(keypoints[i])
        data_pose_template = normalize_keypoints(pose_templates[pose_classes[i]]) * size + offset
        pred_pose_template = normalize_keypoints(pose_templates[pred_classes[i]]) * size + offset
        img = np.uint8(Image.open(im_files[i]).convert('RGB'))
        img_real_pose = draw_pose(keypoints[i], bg=img.copy())
        img_data_pose = draw_pose(data_pose_template, bg=img.copy())
        img_pred_pose = draw_pose(pred_pose_template, bg=img.copy())
        img_grid = np.hstack((img, img_real_pose, img_data_pose, img_pred_pose))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        Image.fromarray(img_grid).save(os.path.join(save_dir, f'b{index+1}s{i+1}.png'))


# create model
model = TemplatePredictor(embeds_dim, num_tposes).eval()
if enable_gpu and torch.cuda.is_available():
    model = model.cuda(torch.device(0))
num_params = 0
for param in model.parameters():
    num_params += param.numel()
print(f'[INFO] created model with {num_params/1e6:.1f}M parameters')
model.load_state_dict(torch.load(checkpoint_path))
print(f'[INFO] model checkpoint loaded from {checkpoint_path}')


# create dataloader
dataloader = create_dataloader(data_rootd, 'test', image_size, batch_size, shuffle=False)
print(f'[INFO] created dataloader with {len(dataloader.dataset)} samples')


# test
for i, batch in enumerate(dataloader):
    img, img_p1, img_p2, seg, seg_p1, seg_p2, em = get_inputs(batch)
    pred = model.predict(img, img_p1, img_p2, seg, seg_p1, seg_p2)
    visualize(batch, pred, i, output_dir)
    print(f'\r[TEST] processed batch {i+1}/{len(dataloader)}', end='')
print('')
