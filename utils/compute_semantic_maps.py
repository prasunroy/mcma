import os, pandas as pd, torch
from PIL import Image, ImageFile
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


ImageFile.LOAD_TRUNCATED_IMAGES = True

# configurations
# -----------------------------------------------------------------------------
DATASET_ROOT = 'datasets/binge_watching'
SAVE_DIR = 'datasets/binge_watching/semantic_maps_oneformer'
MODEL_NAME = 'shi-labs/oneformer_ade20k_dinat_large'
MODEL_CACHE_DIR = '.cache/oneformer'
# -----------------------------------------------------------------------------

model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
processor = OneFormerProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)

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
        semantic_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')
        with torch.no_grad():
            semantic_outputs = model(**semantic_inputs)
        semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]
        semantic_map = Image.fromarray(semantic_map.numpy().astype('uint8'))
        num_images += 1
        save_path = os.path.join(SAVE_DIR, os.path.splitext(f)[0] + '.png')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        semantic_map.save(save_path)
    except:
        num_errors += 1
    print(f'\rprocessed {num_images+num_errors}/{max_images} images :: {num_images} successful :: {num_errors} failed', end='')
