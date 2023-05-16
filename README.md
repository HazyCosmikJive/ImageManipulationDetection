# 🔬 ImageManipulationDetection

A simple codebase for Image Manipulation Detection task, based on image segmentation.

~~(homework project for MultiMedia in UCAS)~~

## Data

- `data/datalist` provides datalist of CASIA used for this codebase; In each line of the infolist, the data is organized as : `[filenpath \t maskpath \t forgery_type];`
  - Filepath and maskpath only contain the filename, thus data_root path is needed in config yaml.
  - For a real image, the maskpath is None, and will be generated a black mask (indicating that it's all real)
  - forgery_type contains: (in CASIA for example)
    - au: authentic
    - sp: splicing
    - cm: copymove
  - We can perform further analysis on model's performance on different kinds of forgery types.

## Code Structure

```
ImageManipulationDetection
├── data
│   ├── dataset_entry  # entry for datasets and dataloaders
│   ├── datasets
│   └── transforms
├── exp  # config yamls
│   └── annotations
│       └── ${mode}_train.txt
├─ models  # different models
│   └── model_entry  # entry for different models
│    	├── smp_models  # models from segmentation_models_pytorch lib
│	├── deeplab_srm  # add SRMConv / BayarConv as noise branch
│	├── deeplabv3  # ori deeplabv3 models
│	└── others...
├── tools  # training relevent
│   ├── evaluation  # not implemented... was meant to be a evaluation entry
│   ├── inference  # model inference
│   └── train  # model train
├── utils  # other tools
│   ├── checkpoint  # saving and loading model
│   ├── dist  # distributed training, not fully implemented
│   ├── init_config  # config yaml relevent
│   ├── log  # logger setting
│   ├── metric  # evaluation metrics for ps segmentation tasks
│   ├── parser  # parse args from .sh bash file
│   ├── scheduler  # lr scheduler
│   └── writer  # tensorboard writer
├── main  # main entry
└── train.sh  # bash scripts

```

## How to use

```
bash train.sh exps/deeplab_casia.yaml 0
#             config                  gpu-id
```

`exps/deeplab_casia.yaml` uses model: `DeepLabV3Plus_srm`, which adds a noise branch to the ori deeplab model.

### configs

The details of configs are commented in `exps/deeplab_casia.yaml`.

### models

model deeplab_srm is a very simple model that uses SRM filter or BayarConv to extract noise information from an image. After extracting noise information using another encoder, the noise feature and RGB image feature will be concated and fused with an attention module. This model could achieve comparable performance with models before 2020 but is out-of-date now...

## Others

This codebase might contains some bugs that was not noticed and may further be improved on its efficiency. Results on this codebase are not fully tested and compared with previous experiments conducted on another codebase using same model and same datasets.
