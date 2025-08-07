import numpy as np, os, torch
from PIL import Image
from data import create_dataloader
from model import DeformationPredictor
from visualization import draw_point, draw_pose


# configurations
# -----------------------------------------------------------------------------
embeds_dim = 512
latent_dim = 32
num_tposes = 30
num_joints = 16
enable_gpu = True
data_rootd = '../../datasets/binge_watching'
image_size = 256
batch_size = 8
output_dir = './result'
checkpoint_path = './output/model_epoch_001_improved.pth'
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
    df = data['deformations']
    z = torch.randn(img.size(0), 32)
    if enable_gpu and torch.cuda.is_available():
        img, img_p1, img_p2, seg, seg_p1, seg_p2, em, df, z = img.cuda(torch.device(0)), img_p1.cuda(torch.device(0)), img_p2.cuda(torch.device(0)), seg.cuda(torch.device(0)), seg_p1.cuda(torch.device(0)), seg_p2.cuda(torch.device(0)), em.cuda(torch.device(0)), df.cuda(torch.device(0)), z.cuda(torch.device(0))
    return img, img_p1, img_p2, seg, seg_p1, seg_p2, em, df, z


def normalize_keypoints(keypoints):
    k = keypoints.copy().reshape(-1, 2)
    kx = k[np.where(k[:, 0] >= 0), 0]
    ky = k[np.where(k[:, 1] >= 0), 1]
    nx = (kx - kx.min()) / (kx.max() - kx.min())
    ny = (ky - ky.min()) / (ky.max() - ky.min())
    k[np.where(k[:, 0] >= 0), 0] = nx
    k[np.where(k[:, 1] >= 0), 1] = ny
    return np.float32(k)


# translate pose
def translate_pose(from_pose, to_pose):
    c = from_pose.reshape(-1, 2)[:-1]
    k = to_pose.reshape(-1, 2)[:-1]
    cxmin = c[np.where(c[:, 0] >= 0), 0].min()
    cymin = c[np.where(c[:, 1] >= 0), 1].min()
    kxmin = k[np.where(k[:, 0] >= 0), 0].min()
    kymin = k[np.where(k[:, 1] >= 0), 1].min()
    t = np.float32((kxmin - cxmin, kymin - cymin))
    return from_pose.reshape(-1, 2) + t


# scale and deform pose
def scale_and_deform_pose(source_pose, target_pose, deformations):
    scales = get_bbox_size(target_pose)
    sp = normalize_keypoints(source_pose) * 256
    dp = sp + deformations.reshape(-1, 2)
    dp = normalize_keypoints(dp) * scales
    sp = normalize_keypoints(source_pose) * scales
    return translate_pose(sp, target_pose), translate_pose(dp, target_pose)


def get_bbox_size(keypoints):
    k = keypoints.reshape(-1, 2)
    kx = k[np.where(k[:, 0] >= 0), 0]
    ky = k[np.where(k[:, 1] >= 0), 1]
    dx = kx.max() - kx.min()
    dy = ky.max() - ky.min()
    return np.float32((dx, dy))


# visualize
def visualize(data, pred, index, save_dir):
    im_files = data['img_path']
    gt_poses = data['keypoints'].cpu().numpy()
    cc_poses = data['pose_template'].cpu().numpy()
    roi_mids = data['roi_center'].cpu().numpy()
    real_snd = data['deformations'].cpu().numpy()
    fake_snd = pred.detach().cpu().numpy()
    for i in range(len(pred)):
        gtp = gt_poses[i].reshape(-1, 2)
        ccp = cc_poses[i].reshape(-1, 2)
        ccp_s, ccp_d = scale_and_deform_pose(ccp, gtp, real_snd[i])
        prp_s, prp_d = scale_and_deform_pose(ccp, gtp, fake_snd[i])
        img = np.uint8(Image.open(im_files[i]).convert('RGB'))
        img_roi = draw_point(np.int32(roi_mids[i]), img.copy(), radius=15, color=(0, 0, 255), border_thickness=5, border_color=(255, 255, 255))
        img_gtp = draw_pose(gtp, bg=img.copy())
        img_ccp_s = draw_pose(ccp_s, bg=img.copy())
        img_ccp_d = draw_pose(ccp_d, bg=img.copy())
        img_prp_s = draw_pose(prp_s, bg=img.copy())
        img_prp_d = draw_pose(prp_d, bg=img.copy())
        img_grid = np.vstack((
            np.hstack((img, img_roi, img_ccp_s, img_ccp_d)),
            np.hstack((img_prp_s, img_prp_d, img_gtp, img.copy()*0))
        ))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        Image.fromarray(img_grid).save(os.path.join(save_dir, f'b{index+1}s{i+1}.png'))


# create model
model = DeformationPredictor(embeds_dim, latent_dim, num_tposes, num_joints).eval()
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
    img, img_p1, img_p2, seg, seg_p1, seg_p2, em, df, z = get_inputs(batch)
    pred = model.predict(img, img_p1, img_p2, seg, seg_p1, seg_p2, em, z)
    visualize(batch, pred, i, output_dir)
    print(f'\r[TEST] processed batch {i+1}/{len(dataloader)}', end='')
print('')
