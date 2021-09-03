# Forward model runs for UQ experiment 
# This version saves Level 1B output for an individual sounding, to be aggregated later 
# Command line argument gives the sounding index to process (zero-based)
#    python fwdrun_oco_save_sing.py jobnum

import os
import sys
sys.path.append(os.environ['OCO_CONFIG_DIR'])

from refractor.factory import process_config
from refractor import framework as rf
from retrieval_config import retrieval_config_definition

from numpy import random
from netCDF4 import Dataset
import numpy
import h5py

stidx = int(sys.argv[1])
jbidx = stidx

sdbsln = int(5332)
rdsd = int(jbidx * 1e4) + sdbsln
random.seed(rdsd)

ref_sounding_id = "2015080820071304"
sim_sdg_hdr = int(ref_sounding_id[0:10])
expt_sounding_id = str(int(sim_sdg_hdr * 1e6 + jbidx + 1))
sim_l1b = 'lnd_nadir_refractor_expt_l1b_uqscene.h5'
print(expt_sounding_id)

if os.path.exists(sim_l1b):
    print('Scene present')
else:
    print('Scene not found')

# Read in state vectors
stvcs = 'lnd_nadir_201508_refractor_state_vectors.h5'
f = h5py.File(stvcs,'r')
xin = f['true_state_vector'][jbidx,:]
f.close()

config_def = retrieval_config_definition(sim_l1b, sim_l1b, expt_sounding_id)
config_inst = process_config(config_def)

fm = config_inst['forward_model']
sv = config_inst['retrieval']['state_vector']

# Update state vector 
aprorig = config_inst['retrieval']['initial_guess']
nstate = aprorig.shape[0]
sv.update_state(xin)

# Spectral Windows
spec_win_range = config_inst['spec_win'].range_array
orig_spec_win_value = spec_win_range.value.copy()
new_spec_win_value = spec_win_range.value.copy()

orig_bad_samp_mask = config_inst['spec_win'].bad_sample_mask.copy()
new_bad_samp_mask = config_inst['spec_win'].bad_sample_mask.copy()

for spec_idx in range(int(config_inst['common']['num_channels'])):
    numpixel = config_inst['instrument'].pixel_spectral_domain(spec_idx).data.shape[0]
    new_spec_win_value[spec_idx, 0, 0] = 1
    new_spec_win_value[spec_idx, 0, 1] = numpixel  

spec_win_range.value = new_spec_win_value
config_inst['spec_win'].range_array = spec_win_range
config_inst['spec_win'].bad_sample_mask = new_bad_samp_mask

for spec_idx in range(int(config_inst['common']['num_channels'])):
    rdspc = fm.radiance(spec_idx,True)
    l1bunc = config_inst['input']['l1b'].noise_model.uncertainty(spec_idx,rdspc.spectral_range.data)
    noisified_radiance = random.normal(rdspc.spectral_range.data, l1bunc )  

    if spec_idx == 0:
        rad_out = numpy.zeros((int(config_inst['common']['num_channels']),rdspc.spectral_range.data.shape[0]),dtype=numpy.float64)
        fm_out = numpy.zeros((int(config_inst['common']['num_channels']),rdspc.spectral_range.data.shape[0]),dtype=numpy.float64)
        print(rad_out.shape)
        rad_out[spec_idx,:] = noisified_radiance

        fm_out[spec_idx,:] = rdspc.spectral_range.data
        lamsb = fm.spectral_grid.low_resolution_grid(spec_idx).data
    else:
        rad_out[spec_idx,:] = noisified_radiance
        fm_out[spec_idx,:] = rdspc.spectral_range.data

print(sv)
# Output
l1b_data_dir = 'l1b'
l1b_out = '%s/l1b_%s.h5' % (l1b_data_dir,expt_sounding_id)

fout = h5py.File(l1b_out,'w')

varo2 = fout.create_dataset('SoundingMeasurements/radiance_o2',data=rad_out[0,:])
varwk = fout.create_dataset('SoundingMeasurements/radiance_weak_co2',data=rad_out[1,:])
varst = fout.create_dataset('SoundingMeasurements/radiance_strong_co2',data=rad_out[2,:])

varo2tr = fout.create_dataset('SoundingMeasurements/noiseless_radiance_o2',data=fm_out[0,:])
varwktr = fout.create_dataset('SoundingMeasurements/noiseless_radiance_weak_co2',data=fm_out[1,:])
varsttr = fout.create_dataset('SoundingMeasurements/noiseless_radiance_strong_co2',data=fm_out[2,:])

varx = fout.create_dataset('StateVector/true_state_vector',data=xin)

fout.close()   


