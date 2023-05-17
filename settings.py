# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:45:22 2023

@author: Song Keyu
"""

## Change these to suit your path
FIGURE_DIR                      = r"../SatPU/figures/"
MODEL_DIR                       = r"../SatPU/models/"
TEP_DATA_DIR                    = r"../SatPU/data/TEP"
DAMADICS_DATA_DIR               = r"../SatPU/data/DAMADICS"
TRAINING_PROCESS_SUBFOLDER      = r"training_process"

from os.path import exists,join
from os import makedirs

makedirs(FIGURE_DIR)   if not exists(FIGURE_DIR)    else None
makedirs(MODEL_DIR)    if not exists(MODEL_DIR)     else None
makedirs(join(FIGURE_DIR,TRAINING_PROCESS_SUBFOLDER)) if not exists(join(FIGURE_DIR,TRAINING_PROCESS_SUBFOLDER)) else None

if not exists(TEP_DATA_DIR):
    raise FileNotFoundError(TEP_DATA_DIR)
if not exists(DAMADICS_DATA_DIR):
    raise FileNotFoundError(DAMADICS_DATA_DIR)
