# Dist-PU on TEP and DAMADICS dataset
This is a repository for reproducing results in paper "A Positive-Unlabeled Learning Approach for Industrial Anomaly Detection Based on Self-Adaptive Training". This code conducts PU learning experiments on TEP and DAMADICS dataset. This is an auxilary repository for <https://github.com/Sakayi/SatPU>. Download the main repository first.

Original Dist-PU is available on <https://github.com/ray-rui/dist-pu-positive-unlabeled-learning-from-a-label-distribution-perspective>

## Environment
An anaconda evironment is provided.
```bash
conda create --name pytorch_env --file DistPU.yaml
```

# Reproducing results in the paper

## Download dataset
Please refer to <https://github.com/Sakayi/SatPU>

## Conduct experiments and save results

### TEP IDV-1 
```bash
python train_TEP_DAMADICS.py --dataset TEP --TEP-Positive-Ratio 0.8  
python train_TEP_DAMADICS.py --dataset TEP --TEP-Positive-Ratio 0.75 
python train_TEP_DAMADICS.py --dataset TEP --TEP-Positive-Ratio 0.6  
python train_TEP_DAMADICS.py --dataset TEP --TEP-Positive-Ratio 0.4  
python train_TEP_DAMADICS.py --dataset TEP --TEP-Positive-Ratio 0.2  
```

Or run .bat file:
```bash
call ConductExperiments-TEP.bat
```

### DAMADICS Actuator3
```bash
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.8  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.75 --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.6  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.4  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.2  --caseid %%I
```

Or run .bat file:
```bash
call ConductExperiments-DAMADICS.bat
```

## Organize experiment results
Please refer to <https://github.com/Sakayi/SatPU>