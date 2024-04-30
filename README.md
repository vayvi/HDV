# Historical Diagram Vectorization

This repo is the official implementation for [Historical Astronomical Diagrams Decomposition in Geometric Primitives](http://imagine.enpc.fr/~kallelis/icdar2024/).

This repo builds on the code for [DINO-DETR](https://github.com/IDEA-Research/DINO), the official implementation of the paper "[DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)".

# Introduction
We present a model which modifies DINO-DETR to perform historical astronomical diagram vectorization by predicting simple geometric primitives, such as lines, circles, and arcs.

![method](figures/architecture_figure.jpg "model arch")

# Getting Started
<details>
  <summary>1. Installation</summary>

The model was trained with `python=3.11.0`, `pytorch=2.1.0`, `cuda=11.8` and 
builds on the DETR-variants [DINO](https://arxiv.org/abs/2203.03605)/[DN](https://arxiv.org/abs/2203.01305)/[DAB](https://arxiv.org/abs/2201.12329) and [Deformable-DETR](https://arxiv.org/abs/2010.04159). 

1. Clone this repository and create virtual environment
   ```bash
   git clone git@github.com:vayvi/HDV.git
   cd HDV/
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Follow instructions to install a [Pytorch](https://pytorch.org/get-started/locally/) version compatible with your system and CUDA version
3. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Compiling CUDA operators
    ```bash
    python src/models/dino/ops/setup.py build install # 'cuda not availabel', run => export CUDA_HOME=/usr/local/cuda-<version>
    # unit test (should see all checking is True) # could output an outofmemory error
    python src/models/dino/ops/test.py
    ```
5. Installing the local package for synthetic data generation
    ```bash
    pip install -e synthetic/.
    ```
</details>

<details>
    <summary>2. Annotated Dataset and Model Checkpoint</summary>

Our annotated dataset along with our main model checkpoints can be found [here](https://drive.google.com/drive/folders/1W3SdaGah2l8QIxPcQt4i3s446NAzPx4J?usp=sharing). 
Annotations are in SVG format. We provide helper functions for parsing svg files in Python if you would like to process a custom annotated dataset.

To download the manually annotated dataset, run:
```bash
bash scripts/download_eida_data.sh
```

Datasets should be organized as follows:
```bash
HDV/
  data/
    └── eida_dataset/
      └── images_and_svgs/
    └── custom_dataset/
      └── images_and_svgs/
```

To download the pretrained models, run:
```bash
bash scripts/download_pretrained_models.sh
```

Checkpoints should be organized as follows:
```bash
HDV/
  logs/
    └── main_model/
      └── checkpoint0012.pth
      └── checkpoint0036.pth
      └── config_cfg.py
    └── other_model/
      └── checkpoint0044.pth
      └── config_cfg.py
    ...
```

You can process the ground-truth data for evaluation using:
```bash
bash scripts/process_annotated_data.sh "eida_dataset" # or "custom_dataset", etc.
```
</details>

<details>
<summary>3. Synthetic Dataset</summary>

### Generate Synthetic Dataset

The synthetic dataset generation process requires a resource of text and document backgrounds. 
We use the resources available in [docExtractor](https://github.com/monniert/docExtractor) and [diagram-extraction](https://github.com/Segolene-Albouy/Diagram-extraction).
The code for generating the synthetic data is also heavily based on docExtractor.

To get the synthetic resource (backgrounds) for the synthetic dataset you can launch:
```bash
bash scripts/download_synthetic_resource.sh
```

### Or download it

Download the synthetic resource folder [here](https://www.dropbox.com/s/tiqqb166f5ygzx2/synthetic_resource.zip?dl=0) and unzip it in the data folder.

</details>

# Evaluation and Testing

<details>
  <summary>1. Evaluate our pretrained models</summary>

After downloading and processing the evaluation dataset, you can evaluate the pretrained model as follows.
Download a model checkpoint, for example "checkpoint0044.pth" and launch

```bash
bash scripts/evaluate_on_eida_final.sh <model_name> <epoch_number>
```

For example:
```bash
bash scripts/evaluate_on_eida_final.sh main_model 0044
```

You should get the AP for different primitives and for different distance thresholds.
</details>

<details>

  <summary>2. Inference and Visualization</summary>

For inference and visualizing results over custom images, you can use the [notebook](src/inference.ipynb).

</details>

# Training
<details>
  <summary>1. Training from scratch on synthetic data</summary>
To re-train the model from scratch on the synthetic dataset, you can launch 

```bash
bash scripts/train_model.sh config/
```
</details>

<details>
  <summary>2. Training on a custom dataset</summary>
To train on a custom dataset, the custom dataset annotations should be in a COCO-like format, and should be in 

```bash
  data/
    └── custom_dataset_processed/
      └── annotations/
      └── train/
      └── val/
```
You should then adjust the coco_path variable to `custom_dataset_processed` in the [config](src/config/DINO_4scale.py) file.
</details>

# Bibtex
If you find this work useful, please consider citing:

```
@misc{kalleli2024historical,
    title={Historical Astronomical Diagrams Decomposition in Geometric Primitives},
    author={Syrine Kalleli and Scott Trigg and Ségolène Albouy and Matthieu Husson and Mathieu Aubry},
    year={2024},
    eprint={2403.08721},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
