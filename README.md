# SSL-MIMOSA: Self-Supervised Learning for Fast Multiparameter Estimation Including T₂* Mapping in Quantitative MRI with MIMOSA

![Alt text](figure/SSL-MIMOSA.png?raw=true "SSL-MIMOSA")

This is the official code for **"SSL-MIMOSA: Self-Supervised Learning for Fast Multiparameter Estimation Including T₂* Mapping in Quantitative MRI with MIMOSA"**.

The code structure is based on [SSL-QALAS](https://github.com/yohan-jun/SSL-QALAS) and [fastMRI](https://github.com/facebookresearch/fastMRI).

## Installation
For dependencies and installation, run the following:

```bash
conda env create -f environment.yml
conda activate ssl_mimosa
pip install -e .
```
## Generating Training and Validation Data
To make the .h5 file, run the `MIMOSA_plus_save_h5.m` matlab file.

This assumes the same subject data is used for validation (i.e., subject specific training and validation), so `train_data.h5` is used for `val_data.h5`.

Sample data can be found [here](https://drive.google.com/drive/folders/1ISMRvQYMNe-zX9sbl8K4Gisd32RDa9Ov?usp=sharing). This includes the IE dictionary.

## Model Training
To train the model, run `train_mimosa.py` as below:

```bash
python SSL-MIMOSA-main/mimosa/train_mimosa.py --check_val_every_n_epoch 1 --num_sanity_val_steps 0 --sample_rate 1.0
```

Note: some of the variables (e.g., turbo factor or echo spacing) might need to be updated in the  `fastmri/models/qalas_map.py` based on your sequence.

## Training and Validation Logs
To track the training and validation logs, run the tensorboard as below:

```bash
tensorboard --logdir=mimosa_plus_log/lightning_logs
```

## Inference
To infer the model, run `inference_mimosa_map.py` as below:
Note: the directories should be updated to reflect your directories, and the checkpoint file should be updated corresponding to your training outcome.

```bash
python SSL-MIMOSA-main/mimosa/inference_mimosa_map.py --data_path /autofs/space/marduk_001/users/tommy/mimosa_plus_data/multicoil_val --state_dict_file /autofs/space/marduk_001/users/tommy/mimosa_plus_log/checkpoints/epoch=990-step=991.ckpt --output_path /autofs/space/marduk_001/users/tommy/mimosa_plus_data
```

The reconstructed maps can be read on Matlab using the `h5read` matlab function:

```bash
T1 = h5read('train_data.h5','/reconstruction_t1');
T2 = h5read('train_data.h5','/reconstruction_t2');
T2s = h5read('train_data.h5','/reconstruction_t2s');
PD = h5read('train_data.h5','/reconstruction_pd');
IE = h5read('train_data.h5','/reconstruction_ie');
```