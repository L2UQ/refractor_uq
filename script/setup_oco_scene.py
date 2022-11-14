# Setup scene file for OCO experiment

import os
import sys
sys.path.append(os.environ['OCO_CONFIG_DIR'])

import numpy
import h5py
import refractor_uq

ref_sounding_id = "2020082319555502"
l1b_file = "oco2_L1bScND_32686a_200823_B10206r_210506204351.h5"
met_file = "oco2_L2MetND_32686a_200823_B10206r_210506064119.h5"
l1b_fields = 'oco2_l1b_fields.csv'
met_fields = 'oco2_met_fields.csv'

if os.path.exists(l1b_file):
    print('L1B present')
else:
    print('L1B not found')

if os.path.exists(met_file):
    print('Met present')
else:
    print('Met not found')

sdspt = refractor_uq.oco2_sounding_idx_match(int(ref_sounding_id),l1b_file)
print(sdspt)

l1bflds = refractor_uq.process_field_list(l1b_fields)
l1bflds.bytes = l1bflds.bytes.astype(int)
l1bflds.dims = l1bflds.dims.astype(int)
print(l1bflds)

metflds = refractor_uq.process_field_list(met_fields)
metflds.bytes = metflds.bytes.astype(int)
metflds.dims = metflds.dims.astype(int)
print(metflds)

# Simulation setup
nsim = 5000
sim_sdg_hdr = int(ref_sounding_id[0:10])
print(sim_sdg_hdr)
sim_l1b = 'lnd_nadir_refractor_expt_l1b_uqscene_202008.h5' 

refractor_uq.setup_uq_l1b(sim_l1b,l1bflds,metflds,sdspt,l1b_file,met_file,sim_sdg_hdr,nsim, \
                          save_noise=True,discrep=True)




