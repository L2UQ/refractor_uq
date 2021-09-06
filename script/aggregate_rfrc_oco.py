# Aggregate experiment results from ReFRACtor

import os
import sys
sys.path.append(os.environ['OCO_CONFIG_DIR'])

import numpy
import h5py
import refractor_uq

expt_dir = os.path.realpath("/home/jonhobbs/Documents/Research/ReFRACtor")
scene_file = os.path.join(expt_dir,"lnd_nadir_refractor_expt_l1b_uqscene.h5")
agg_file = os.path.join(expt_dir,"lnd_nadir_201508_refractor_aggregate_retrieval.h5")
output_dir = os.path.join(expt_dir,"output")


# Prepare aggregate
refractor_uq.uq_expt_aggregate_l2(scene_file,agg_file,output_dir)

