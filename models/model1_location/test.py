import numpy as np, os, torch
from PIL import Image
from data import create_dataloader
from model import LocationPredictor
from visualization import draw_point


# configurations
# -----------------------------------------------------------------------------
embeds_dim = 128
latent_dim = 32
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
    seg = data['seg']
    z = torch.randn(img.size(0), latent_dim)
    if enable_gpu and torch.cuda.is_available():
        img, seg, z = img.cuda(torch.device(0)), seg.cuda(torch.device(0)), z.cuda(torch.device(0))
    return img, seg, z


# visualize
def visualize(data, pred, index, save_dir):
    im_files = data['img_path']
    roi_mids = data['roi_center'].cpu().numpy()
    roi_pred = pred.detach().cpu().numpy()
    for i in range(len(pred)):
        img_pil = Image.open(im_files[i]).convert('RGB')
        img_arr = np.uint8(img_pil)
        pred_xy = roi_pred[i] * (np.float32(img_pil.size) - 1) / image_size
        real_xy = draw_point(roi_mids[i], img_arr.copy(), radius=15, color=(0, 0, 255), border_thickness=5, border_color=(255, 255, 255))
        fake_xy = draw_point(pred_xy, img_arr.copy(), radius=15, color=(255, 0, 0), border_thickness=5, border_color=(255, 255, 255))
        img_all = Image.fromarray(np.vstack((img_arr, real_xy, fake_xy)))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        img_all.save(os.path.join(save_dir, f'b{index+1}s{i+1}.png'))


# create model
model = LocationPredictor(embeds_dim, latent_dim).eval()
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
    img, seg, z = get_inputs(batch)
    pred = model.predict(img, seg, z)
    visualize(batch, pred, i, output_dir)
    print(f'\r[TEST] processed batch {i+1}/{len(dataloader)}', end='')
print('')
