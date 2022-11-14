# Combine L1B results into single file

import os
import sys
sys.path.append(os.environ['OCO_CONFIG_DIR'])

import numpy
import h5py
import refractor_uq

expt_dir = os.path.realpath("./")
scene_file = os.path.join(expt_dir,"lnd_nadir_refractor_expt_l1b_uqscene_202008.h5")
output_dir = os.path.join(expt_dir,"l1b")


# Prepare aggregate
refractor_uq.uq_expt_aggregate_l1b(scene_file,output_dir,discrep=True)


