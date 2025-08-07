import os, pandas as pd
from PIL import Image, ImageFile
from transformers import pipeline


ImageFile.LOAD_TRUNCATED_IMAGES = True

# configurations
# -----------------------------------------------------------------------------
DATASET_ROOT = 'datasets/binge_watching'
SAVE_DIR = 'datasets/binge_watching/depth_maps'
MODEL_NAME = 'LiheYoung/depth-anything-large-hf'
DEVICE = 'cuda:0'
# -----------------------------------------------------------------------------

pipe = pipeline(task='depth-estimation', model=MODEL_NAME, device=DEVICE)

image_files = []
for split in ['trainlist.txt', 'testlist.txt']:
    fp = os.path.join(DATASET_ROOT, split)
    image_files += pd.read_csv(fp, sep=' ', header=None).values[:, 0].tolist()
image_files = sorted(set(image_files))
max_images = len(image_files)

num_images = 0
num_errors = 0
for f in image_files:
    try:
        fp = os.path.join(DATASET_ROOT, 'data', f)
        image = Image.open(fp).convert('RGB')
        depth_map = pipe(image)['depth']
        save_path = os.path.join(SAVE_DIR, os.path.splitext(f)[0] + '.png')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        depth_map.save(save_path)
        num_images += 1
    except:
        num_errors += 1
    print(f'\rprocessed {num_images+num_errors}/{max_images} images :: {num_images} successful :: {num_errors} failed', end='')

print('')
