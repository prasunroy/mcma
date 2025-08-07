import datetime, matplotlib as mpl, numpy as np, os, pandas as pd, torch
from models.model1_location import LocationPredictor
from models.model2_template import TemplatePredictor
from models.model3_scale import ScalePredictor
from models.model4_deformation import DeformationPredictor
from PIL import Image, ImageFile
from torchvision import transforms as T
from visualization import draw_bbox, draw_point, draw_pose


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(11111111)


# configurations
# -----------------------------------------------------------------------------
data_dpath = '../datasets/binge_watching'
data_fpath = '../datasets/binge_watching/testlist.txt'
tmpl_fpath = '../datasets/binge_watching/centers_30.txt'
map_subdir = 'semantic_maps'

model1_ckpt = './checkpoints/model1_location_best.pth'
model2_ckpt = './checkpoints/model2_template_best.pth'
model3_ckpt = './checkpoints/model3_scale_best.pth'
model4_ckpt = './checkpoints/model4_deformation_best.pth'

patch_size = 256
device_exe = torch.device(0)

sample_indices = []
trials_per_sample = 5

fixed_location = True
fixed_template = False
fixed_scale = False

save_dpath = './result'
# -----------------------------------------------------------------------------


def normalize_keypoints(keypoints):
    k = keypoints.copy().reshape(-1, 2)
    kx = k[np.where(k[:, 0] >= 0), 0]
    ky = k[np.where(k[:, 1] >= 0), 1]
    nx = (kx - kx.min()) / (kx.max() - kx.min())
    ny = (ky - ky.min()) / (ky.max() - ky.min())
    k[np.where(k[:, 0] >= 0), 0] = nx
    k[np.where(k[:, 1] >= 0), 1] = ny
    return np.float32(k)


def get_bbox_parameters(keypoints):
    k = keypoints.reshape(-1, 2)
    kx = k[np.where(k[:, 0] >= 0), 0]
    ky = k[np.where(k[:, 1] >= 0), 1]
    x1 = kx.min()
    y1 = ky.min()
    x2 = kx.max()
    y2 = ky.max()
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    dx = x2 - x1
    dy = y2 - y1
    return np.float32((x1, y1, x2, y2, xc, yc, dx, dy))


def get_image_patches(image, point):
    x, y = point
    d1, d2 = image.height // 2, image.height // 4
    bbox1 = (x - d1, y - d1, x + d1, y + d1)
    bbox2 = (x - d2, y - d2, x + d2, y + d2)
    return image.crop(bbox1), image.crop(bbox2)


def get_pose_embedding(num_classes, pose_class):
    return np.eye(num_classes)[pose_class].astype(np.float32)


def rescale_points(points, from_size, to_size):
    return points.reshape(-1, 2) / (np.float32(from_size) - 1) * (np.float32(to_size) - 1)


def postprocess_semantic_map(semantic_map):
    colormap = np.uint8([
        [0, 0, 0], [128, 128, 0], [128, 0, 64], [255, 128, 0],
        [255, 64, 64], [64, 64, 255], [64, 0, 64], [0, 255, 128]
    ])
    colormapped_im = np.zeros((semantic_map.shape[0], semantic_map.shape[1], 3), dtype=np.uint8)
    for label in range(8):
        colormapped_im[semantic_map == label * 36] = colormap[label]
    return colormapped_im


def postprocess_depth_map(depth_map):
    # reference: https://github.com/alopezgit/DESC/issues/3#issuecomment-959803242
    mask = depth_map != 0
    disp_map = 1 / depth_map
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask, -1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im


# create models and load checkpoints
m1 = LocationPredictor(128, 32).eval().to(device_exe)
m1.load_state_dict(torch.load(model1_ckpt))
print('[LOAD] LocationPredictor...... OK')

m2 = TemplatePredictor(512, 30).eval().to(device_exe)
m2.load_state_dict(torch.load(model2_ckpt))
print('[LOAD] TemplatePredictor...... OK')

m3 = ScalePredictor(128, 32, 30).eval().to(device_exe)
m3.load_state_dict(torch.load(model3_ckpt))
print('[LOAD] ScalePredictor......... OK')

m4 = DeformationPredictor(512, 32, 30, 16).eval().to(device_exe)
m4.load_state_dict(torch.load(model4_ckpt))
print('[LOAD] DeformationPredictor... OK')


# load data
data = pd.read_csv(data_fpath, sep=' ', header=None).values
if len(sample_indices) > 0:
    data = data[sample_indices]
print(f'[INFO] Found {len(data)} test samples')
templates = pd.read_csv(tmpl_fpath, sep=' ', header=None).values.astype(np.float32)[:, :32]
print(f'[INFO] Found {len(templates)} pose templates')


# create transform
transform = T.Compose([
    T.Resize((patch_size, patch_size)),
    T.ToTensor(),
    T.Normalize(mean=(0.5,), std=(0.5,))
])


# create directory for saving results
timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_root = f'{save_dpath}/{timestamp}'
imgs_root = f'{save_root}/images'
if not os.path.isdir(imgs_root):
    os.makedirs(imgs_root)


# begin test
all_results = {f'trial_{t+1}': [] for t in range(trials_per_sample)}

for i in range(len(data)):
    # run for each sample
    img_path = os.path.join(data_dpath, 'data', data[i, 0])
    map_path = os.path.join(data_dpath, map_subdir, os.path.splitext(data[i, 0])[0] + '.png')
    keypoint = np.float32(data[i, 1:33])
    pose_cls = np.int64(data[i, -1])
    
    img_input = Image.open(img_path).convert('RGB')
    map_input = Image.open(map_path).convert('RGB')
    
    img_tensor = transform(img_input).unsqueeze(0).to(device_exe)
    map_tensor = transform(map_input).unsqueeze(0).to(device_exe)
    
    for trial in range(trials_per_sample):
        # run for each trial
        z = torch.randn(1, 32).to(device_exe)
        
        # stage-1: find location
        if fixed_location:
            location = get_bbox_parameters(keypoint)[4:6]
        else:
            location_pred = m1.predict(img_tensor, map_tensor, z).squeeze().cpu().numpy()
            location = rescale_points(location_pred, patch_size, img_input.size).reshape(-1)
        
        # create image patches
        img_patch1, img_patch2 = get_image_patches(img_input, location)
        map_patch1, map_patch2 = get_image_patches(map_input, location)
        
        img_p1_tensor = transform(img_patch1).unsqueeze(0).to(device_exe)
        img_p2_tensor = transform(img_patch2).unsqueeze(0).to(device_exe)
        map_p1_tensor = transform(map_patch1).unsqueeze(0).to(device_exe)
        map_p2_tensor = transform(map_patch2).unsqueeze(0).to(device_exe)
        
        # stage-2: find template
        if fixed_template:
            template_index = pose_cls - 1
        else:
            template_index = m2.predict(
                img_tensor, img_p1_tensor, img_p2_tensor,
                map_tensor, map_p1_tensor, map_p2_tensor
            ).squeeze().cpu().numpy().argmax()
        pose_template = normalize_keypoints(templates[template_index]) * (patch_size - 1)
        
        # create pose embedding
        pose_embedding = get_pose_embedding(len(templates), template_index)
        pose_embedding_tensor = torch.tensor(pose_embedding).unsqueeze(0).to(device_exe)
        
        # stage-3: find scale
        if fixed_scale:
            scale = get_bbox_parameters(keypoint)[6:8]
        else:
            scale_pred = m3.predict(
                img_tensor, img_p1_tensor, img_p2_tensor,
                map_tensor, map_p1_tensor, map_p2_tensor,
                pose_embedding_tensor, z
            ).squeeze().cpu().numpy()
            scale = rescale_points(scale_pred, patch_size, img_input.size).reshape(-1)
        
        # stage-4: find deformation
        deformation_pred = m4.predict(
            img_tensor, img_p1_tensor, img_p2_tensor,
            map_tensor, map_p1_tensor, map_p2_tensor,
            pose_embedding_tensor, z
        ).squeeze().cpu().numpy().reshape(-1, 2)
        
        # estimate pose
        estimated_pose = pose_template + deformation_pred
        estimated_pose = rescale_points(estimated_pose, patch_size, scale)
        offset = location - (scale / 2)
        estimated_pose += offset
        
        # visualize results
        point_radius = int(round(max(img_input.size) / 85))
        border_thickness = int(round(max(img_input.size) / 256))
        line_thickness = int(round(max(img_input.size) / 128))
        
        img_np = np.uint8(img_input)
        if map_subdir == 'semantic_maps':
            map_np = postprocess_semantic_map(np.uint8(map_input.convert('L')))
        else:
            map_np = postprocess_depth_map(np.uint8(map_input.convert('L')))
        
        x1_gt, y1_gt, x2_gt, y2_gt, xc_gt, yc_gt = get_bbox_parameters(keypoint)[:-2]
        x1, y1, x2, y2, xc, yc = get_bbox_parameters(estimated_pose)[:-2]
        
        img_bbox_gt_np = draw_bbox(
            (x1_gt, y1_gt, x2_gt-x1_gt, y2_gt-y1_gt),
            img_np.copy(),
            color=(0, 255, 0), alpha=0.4
        )
        img_bbox_gt_np = draw_point(
            (xc_gt, yc_gt),
            img_bbox_gt_np,
            radius=point_radius, color=(0, 255, 0),
            border_thickness=border_thickness, border_color=(255, 255, 255)
        )
        img_bbox_np = draw_bbox(
            (x1, y1, x2-x1, y2-y1),
            img_np.copy(),
            color=(255, 0, 0), alpha=0.4
        )
        img_bbox_np = draw_point(
            (xc, yc),
            img_bbox_np,
            radius=point_radius, color=(255, 0, 0),
            border_thickness=border_thickness, border_color=(255, 255, 255)
        )
        img_pose_gt_np = draw_pose(keypoint, thickness=line_thickness, bg=img_np.copy())
        img_pose_np = draw_pose(estimated_pose, thickness=line_thickness, bg=img_np.copy())
        
        all_images = Image.fromarray(np.hstack((
            img_np, map_np, img_bbox_gt_np, img_bbox_np, img_pose_gt_np, img_pose_np
        )))
        all_images.save(os.path.join(imgs_root, f'{i+1}--{trial+1}.png'))
        
        result = [data[i, 0]] + estimated_pose.reshape(-1).tolist() + [template_index + 1]
        all_results[f'trial_{trial+1}'].append(result)
        
        print(f'\r[TEST] processed sample {i+1}/{len(data)}', end='')

for fname, result in all_results.items():
    result_data = pd.DataFrame(result)
    result_data.to_csv(os.path.join(save_root, f'result_{fname}.csv'), header=False, index=False)

print('')

