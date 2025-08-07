import glob, numpy as np, os
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

# configurations
# -----------------------------------------------------------------------------
SOURCE_DIR = 'datasets/binge_watching/semantic_maps_oneformer'
TARGET_DIR = 'datasets/binge_watching/semantic_maps'

LABEL_MAPS = {
    0:1, 8:1, 14:1, 32:1,
    3:2, 6:2, 9:2, 11:2, 13:2, 29:2, 52:2, 91:2, 94:2,
    53:3, 59:3, 96:3, 121:3,
    15:4, 33:4, 56:4, 64:4,
    19:5, 23:5, 30:5, 31:5, 69:5, 75:5, 110:5,
    7:6,
    12:7
}
# -----------------------------------------------------------------------------

image_files = sorted(glob.glob(f'{SOURCE_DIR}/**/*.png', recursive=True))

for f in image_files:
    source = np.uint8(Image.open(f))
    target = np.int32(source * 0)
    for source_label, target_label in LABEL_MAPS.items():
        target += np.where(source == source_label, target_label, 0)
    target = Image.fromarray(np.uint8(target * 36))
    fp = f.replace(SOURCE_DIR, TARGET_DIR)
    if not os.path.isdir(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))
    target.save(fp)
    print(f'processed {fp}')
