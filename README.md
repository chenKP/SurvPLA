# Pathological Diagnosis Prior Guided Latent Space Alignment for Multimodal Survival Prediction

## Environment

Tested with **Python 3.10**, **PyTorch 2.6 + CUDA 12.4**. A GPU is required (CONCH text encoder + training).

### 1. Create the conda env

```bash
conda create -n surpla python=3.10 -y
conda activate surpla
```

Or from `environment.yml` (PyTorch still installed separately in the next step):

```bash
conda env create -f environment.yml
conda activate surpla
```
### 2. Install CONCH and weights

CONCH is the frozen text encoder.

```bash
pip install git+https://github.com/mahmoodlab/CONCH.git
```

Weights are gated on Hugging Face ([MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)). After access is granted:

```bash
mkdir -p model
# download pytorch_model.bin, then:
cp /path/to/pytorch_model.bin ./model/pytorch_model_CONCH.bin
```

Override the path if needed:

```bash
export CONCH_CKPT=/path/to/pytorch_model.bin
# or
python main.py -c configs/Brain-LGG.yaml --conch_ckpt /path/to/pytorch_model.bin
```

## Data layout

Three inputs per cohort, aligned by slide / sample name.

```
data/<COHORT>/
  omics.csv          # genomics + survival labels
  pt_file/           # {sampleName}.pt  WSI patch features
  reports.csv        # diagnostic reports
```

Edit the paths in `configs/*.yaml` (`gene_dir`, `img_dir`, `diagnostic_reports`).

## Train

From the repository root:

```bash
conda activate surpla
python main.py -c configs/Brain-LGG.yaml --gpu_id 0
```
## Project layout

```
.
├── main.py                 # training entry
├── model.py                # SurPLA_FusionVAENet
├── dataset.py
├── report_feature.py       # frozen CONCH encoder
├── gene_feature.py
├── vision_feature.py
├── feature_fusion.py
├── betaVAE.py
├── utils.py                # survival losses
├── iAUC.py
├── args.py
├── configs/                # YAML configs (edit paths)
├── model/                  # put pytorch_model_CONCH.bin here
```

## License / third-party

CONCH weights and code follow [MahmoodLab/CONCH](https://github.com/mahmoodlab/CONCH). Request access before redistributing the checkpoint.


