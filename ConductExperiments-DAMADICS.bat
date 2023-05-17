for %%I in (11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.8  --caseid %%I
for %%I in (11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.75 --caseid %%I
for %%I in (11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.6  --caseid %%I
for %%I in (11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.4  --caseid %%I
for %%I in (11,12,13,14,15,16,17,18,19,20) do python train_TEP_DAMADICS.py --dataset DAMADICS --TEP-Positive-Ratio 0.2  --caseid %%I