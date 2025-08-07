import numpy as np, os, pandas as pd, random
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


ImageFile.LOAD_TRUNCATED_IMAGES = True


class BingeWatchingDataset(Dataset):
    
    def __init__(self, dataset_root, split='train', patch_size=256):
        super(BingeWatchingDataset, self).__init__()
        assert split == 'train' or split == 'test'
        data_list = 'trainlist.txt' if split=='train' else 'testlist.txt'
        data_file = os.path.join(dataset_root, data_list)
        cluster_file = os.path.join(dataset_root, 'centers_30.txt')
        self.data_root = dataset_root
        self.data = self.filter_samples(pd.read_csv(data_file, sep=' ', header=None).values)
        self.pose_templates = pd.read_csv(cluster_file, sep=' ', header=None).values.astype(np.float32)[:, :32]
        self.transform_patch = T.Compose([
            T.Resize((patch_size, patch_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.5,), std=(0.5,))
        ])
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        img_path = os.path.join(self.data_root, 'data', str(sample[0]))
        seg_path = os.path.join(self.data_root, 'semantic_maps', os.path.splitext(str(sample[0]))[0] + '.png')
        keypoints = sample[1:33].reshape(16, 2).astype(np.float32)
        pose_class = np.int32(sample[-1]) - 1
        roi_center = self.get_bbox_center(keypoints)
        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path).convert('RGB')
        img_patch1, img_patch2 = self.get_image_patches(img, roi_center)
        seg_patch1, seg_patch2 = self.get_image_patches(seg, roi_center)
        pose_embedding = self.get_pose_embedding(len(self.pose_templates), pose_class)
        norm_pose_template = self.normalize_keypoints(self.pose_templates[pose_class])
        deformations = self.get_deformations(keypoints, norm_pose_template, self.patch_size-1)
        dx, dy = self.get_bbox_size(keypoints) * (self.patch_size-1) / np.float32(img.size)
        # tx, ty = self.get_bbox_anchor(keypoints) * (self.patch_size-1) / np.float32(img.size)
        bbox_params = np.float32((dx, dy))
        scaled_roi_center = roi_center * (self.patch_size-1) / np.float32(img.size)
        return {
            'img_path': img_path,
            'seg_path': seg_path,
            'keypoints': keypoints,
            'pose_class': pose_class,
            'roi_center': roi_center,
            'img': self.transform_patch(img),
            'img_patch1': self.transform_patch(img_patch1),
            'img_patch2': self.transform_patch(img_patch2),
            'seg': self.transform_patch(seg),
            'seg_patch1': self.transform_patch(seg_patch1),
            'seg_patch2': self.transform_patch(seg_patch2),
            'pose_embedding': pose_embedding,
            'pose_template': norm_pose_template,
            'deformations': deformations,
            'bbox_params': bbox_params,
            'scaled_roi_center': scaled_roi_center
        }
    
    def filter_samples(self, data):
        filtered_data = []
        frames = sorted(set(data[:, 0].tolist()))
        for frame in frames:
            random_index = random.choice(np.where(data[:, 0]==frame)[0])
            filtered_data.append(data[random_index])
        return np.array(filtered_data)
    
    def get_bbox_center(self, keypoints):
        joints = keypoints.reshape(-1, 2)
        xcoords = joints[np.where(joints[:, 0] >= 0), 0]
        ycoords = joints[np.where(joints[:, 1] >= 0), 1]
        xcenter = (xcoords.min() + xcoords.max()) // 2
        ycenter = (ycoords.min() + ycoords.max()) // 2
        return np.float32((xcenter, ycenter))
    
    def get_image_patches(self, image, point):
        x, y = point
        d1, d2 = image.height // 2, image.height // 4
        bbox1 = (x - d1, y - d1, x + d1, y + d1)
        bbox2 = (x - d2, y - d2, x + d2, y + d2)
        return image.crop(bbox1), image.crop(bbox2)
    
    def get_pose_embedding(self, num_classes, pose_class):
        return np.eye(num_classes)[pose_class].astype(np.float32)
    
    def get_bbox_size(self, keypoints):
        k = keypoints.reshape(-1, 2)
        kx = k[np.where(k[:, 0] >= 0), 0]
        ky = k[np.where(k[:, 1] >= 0), 1]
        dx = kx.max() - kx.min()
        dy = ky.max() - ky.min()
        return np.float32((dx, dy))
    
    def get_bbox_anchor(self, keypoints):
        k = keypoints.reshape(-1, 2)
        kx = k[np.where(k[:, 0] >= 0), 0]
        ky = k[np.where(k[:, 1] >= 0), 1]
        return np.float32((kx.min(), ky.min()))
    
    def get_deformations(self, keypoints, reference_keypoints, scale=1.0):
        k = self.normalize_keypoints(keypoints.reshape(-1, 2)) * scale
        c = self.normalize_keypoints(reference_keypoints.reshape(-1, 2)) * scale
        return np.float32(k - c).reshape(-1)
    
    def normalize_keypoints(self, keypoints):
        k = keypoints.copy().reshape(-1, 2)
        kx = k[np.where(k[:, 0] >= 0), 0]
        ky = k[np.where(k[:, 1] >= 0), 1]
        nx = (kx - kx.min()) / (kx.max() - kx.min())
        ny = (ky - ky.min()) / (ky.max() - ky.min())
        k[np.where(k[:, 0] >= 0), 0] = nx
        k[np.where(k[:, 1] >= 0), 1] = ny
        return np.float32(k)


def create_dataloader(dataset_root, split='train', patch_size=256, batch_size=1, shuffle=True, num_workers=0):
    dataset = BingeWatchingDataset(dataset_root, split, patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

