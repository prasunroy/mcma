import numpy as np, os, torch, torchvision
from PIL import Image
from data import create_dataloader
from model import ScalePredictor
from visualization import draw_bbox_affine


# configurations
# -----------------------------------------------------------------------------
embeds_dim = 128
latent_dim = 32
num_tposes = 30
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
    z = torch.randn(img.size(0), latent_dim)
    if enable_gpu and torch.cuda.is_available():
        img, img_p1, img_p2, seg, seg_p1, seg_p2, em, z = img.cuda(torch.device(0)), img_p1.cuda(torch.device(0)), img_p2.cuda(torch.device(0)), seg.cuda(torch.device(0)), seg_p1.cuda(torch.device(0)), seg_p2.cuda(torch.device(0)), em.cuda(torch.device(0)), z.cuda(torch.device(0))
    return img, img_p1, img_p2, seg, seg_p1, seg_p2, em, z


# visualize
def visualize(data, pred, index, save_dir):
    im_files = data['img_path']
    roi_mids = data['roi_center'].cpu().numpy()
    real_bbp = data['bbox_params'].cpu().numpy()
    fake_bbp = pred.detach().cpu().numpy()
    for i in range(len(pred)):
        img_pl = Image.open(im_files[i]).convert('RGB')
        img_pt = torchvision.transforms.functional.to_tensor(img_pl)
        real_sx, real_sy = real_bbp[i] / image_size
        real_tx, real_ty = (roi_mids[i] - 0.5 * real_bbp[i] * (np.float32(img_pl.size) - 1.) / image_size) / np.float32(img_pl.size)
        real_theta = torch.tensor([[real_sx, 0., real_tx], [0., real_sy, real_ty]])
        fake_sx, fake_sy = fake_bbp[i] / image_size
        fake_tx, fake_ty = (roi_mids[i] - 0.5 * fake_bbp[i] * (np.float32(img_pl.size) - 1.) / image_size) / np.float32(img_pl.size)
        fake_theta = torch.tensor([[fake_sx, 0., fake_tx], [0., fake_sy, fake_ty]])
        real_bbox = draw_bbox_affine(real_theta.unsqueeze(0), img_pt.unsqueeze(0), (0, 255, 0), 0.4).squeeze()
        fake_bbox = draw_bbox_affine(fake_theta.unsqueeze(0), img_pt.unsqueeze(0), (255, 0, 0), 0.4).squeeze()
        img_grid = torch.cat((img_pt, real_bbox, fake_bbox), dim=1)
        img_grid = torchvision.transforms.functional.to_pil_image(img_grid)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        img_grid.save(os.path.join(save_dir, f'b{index+1}s{i+1}.png'))


# create model
model = ScalePredictor(embeds_dim, latent_dim, num_tposes).eval()
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
    img, img_p1, img_p2, seg, seg_p1, seg_p2, em, z = get_inputs(batch)
    pred = model.predict(img, img_p1, img_p2, seg, seg_p1, seg_p2, em, z)
    visualize(batch, pred, i, output_dir)
    print(f'\r[TEST] processed batch {i+1}/{len(dataloader)}', end='')
print('')
