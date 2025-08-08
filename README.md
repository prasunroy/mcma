### Official code for Exploring Mutual Cross-Modal Attention for Context-Aware Human Affordance Generation.

*Accepted in the IEEE Transactions on Artificial Intelligence (TAI) 2025.*

[![badge_torch](https://img.shields.io/badge/made_with-PyTorch_2.0-EE4C2C?style=flat-square&logo=PyTorch)](https://pytorch.org/)
[![badge_arxiv](https://img.shields.io/badge/arXiv-2502.13637-brightgreen?style=flat-square)](https://arxiv.org/abs/2502.13637)

![teaser](https://github.com/user-attachments/assets/c0099350-db25-412b-ad7f-1fbe6d77a856)

### :zap: Getting Started
> Note: This release is tested on Python 3.9.16.
```bash
git clone https://github.com/prasunroy/mcma.git
cd mcma
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### :wrench: Preparing the [Binge Watching](https://www.cs.cmu.edu/~xiaolonw/affordance.html) Dataset
* Download the [dataset](https://www.dropbox.com/s/2ny1h8nvzvfo7vr/affordance_data.tar.gz) and extract into `datasets/binge_watching` directory.
* Download the [pose cluster centers](https://www.cs.cmu.edu/~xiaolonw/affordance/centers_30.txt) into `datasets/binge_watching` directory.
```bash
mkdir datasets
cd datasets
wget https://www.dropbox.com/s/2ny1h8nvzvfo7vr/affordance_data.tar.gz
wget https://www.cs.cmu.edu/~xiaolonw/affordance/centers_30.txt
tar -xzf affordance_data.tar.gz
mv centers_30.txt affordance_data/
rm -r affordance_data/pose_gt_testset/
rm -r affordance_data/*.m
rm affordance_data/pose_all_list.txt
rm affordance_data/README.txt
rm affordance_data.tar.gz
mv affordance_data/ binge_watching/
cd ..
```
* Generate the semantic maps.
```bash
python utils/compute_semantic_maps.py
```
> Alternatively, download the [precomputed semantic maps](https://drive.google.com/file/d/1CtbtZCamIxQ2Cl8QYJ0AVKK-eIM-BZQ1/view) and extract into `datasets/binge_watching/semantic_maps_oneformer` directory.
* Reduce the initial 150 semantic labels to 8 groups.
```bash
python utils/remap_semantic_labels.py
```

### :rocket: Training / Testing Models and Running Inference Pipeline
* Configure and run `train.py` / `test.py` from individual subdirectories under the `models` directory.
* Copy the best checkpoint of each model into `pipeline/checkpoints`. Then configure and run `test.py` from the `pipeline` directory.

### :heart: Citation
```
@article{roy2025exploring,
  title   = {Exploring Mutual Cross-Modal Attention for Context-Aware Human Affordance Generation},
  author  = {Roy, Prasun and Bhattacharya, Saumik and Ghosh, Subhankar and Pal, Umapada and Blumenstein, Michael},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2025},
  issn    = {2691-4581},
  doi     = {https://doi.org/10.1109/TAI.2025.3581897}
}
```

### :page_facing_up: License
```
Copyright 2025 by the authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

<br>

##### Made with :heart: and :pizza: on Earth.
