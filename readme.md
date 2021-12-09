# 16822 Class Project

This project is based on the paper [Predicting Semantic Map Representations from Images with Pyramid Occupancy Networks](https://arxiv.org/pdf/2003.13402.pdf).
### Argoverse
To train on the Argoverse dataset:
1. Download the Argoverse tracking data from https://www.argoverse.org/data.html#tracking-link. Our models were trained on version 1.1, you will need to download the four training blobs, validation blob, and the HD map data.
2. Install the Argoverse devkit from https://github.com/argoai/argoverse-api
3. Cd to `mono-semantic-maps`
5. Edit the `configs/datasets/argoverse.yml` file, setting the `dataroot` and `label_root` entries to the location of the install Argoverse data and the desired ground truth folder respectively.
5. Run our data generation script: `python scripts/make_argoverse_labels.py`. This script will also take a while to run! 

## Training
Run semantic.py script to generate semantic masks. Then run ipm_semantic.py and ipm_image.py to generate IPMed image/semantic masks.

Once ground truth labels and IPMed image/semantic masks have been generated, you can train base model by running the `train.py` script in the root directory: 
```
python train.py --dataset argoverse --model pyramid
```
or train the base model + image/semantic ipm by running 
```
python train.py --dataset argoverse --model pyramid_ipm
```

