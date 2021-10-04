#!/opt/local/depot/python/3.6.4/bin/python3

# Setup simulation experiment L1b/Met 
# Based on load_example_config from oco repository

import os
import sys

sys.path.append(os.environ['OCO_CONFIG_DIR'])

from refractor.factory import process_config
from refractor import framework as rf
from retrieval_config import retrieval_config_definition

from pprint import pprint
from numpy import random
from netCDF4 import Dataset
import numpy
import h5py
import pandas
import refractor_uq

pwts = numpy.array( [1.0/38.0, 1.0/19.0, 1.0/19.0, 1.0/19.0, 1.0/19.0, \
                     1.0/19.0, 1.0/19.0, 1.0/19.0, 1.0/19.0, 1.0/19.0, \
                     1.0/19.0, 1.0/19.0, 1.0/19.0, 1.0/19.0, 1.0/19.0, \
                     1.0/19.0, 1.0/19.0, 1.0/19.0, 1.0/19.0, 1.0/38.0] )
     

case_id = 'co2'

# Read in the case data
csvfl = 'land_state_%s.csv' % (case_id)
svdt = pandas.read_csv(csvfl, dtype={'SVName':str, 'SVValue':float}, encoding='utf-8-sig')
nst = svdt.shape[0]

if case_id == 'sza':
    ref_sounding_id = "2015080820151174"
else:
    ref_sounding_id = "2015080820071304"

l1b_file = "oco2_L1bScND_05862a_150808_B10003r_191117045059.h5"
met_file = "oco2_L2MetND_05862a_150808_B10003r_191117035013.h5"

if os.path.exists(l1b_file):
    print('L1B present')
else:
    print('L1B not found')

if os.path.exists(met_file):
    print('Met present')
else:
    print('Met not found')

config_def = retrieval_config_definition(l1b_file, met_file, ref_sounding_id)
config_inst = process_config(config_def)

#pprint(config_inst, indent = 2)
#pprint(config_inst['retrieval']['retrieval_components'])
sv = config_inst['retrieval']['state_vector']
sv.update_state(svdt['SVValue'])
xco2 = numpy.dot(pwts,svdt['SVValue'][0:20])

#for n, v in zip(sv.state_vector_name, sv.state):
#    print("%0.6f %s" (v,n))


# Spectral windows
spec_win_range = config_inst['spec_win'].range_array
orig_spec_win_value = spec_win_range.value.copy()
new_spec_win_value = spec_win_range.value.copy()

orig_bad_samp_mask = config_inst['spec_win'].bad_sample_mask.copy()
new_bad_samp_mask = config_inst['spec_win'].bad_sample_mask.copy()

for spec_idx in range(int(config_inst['common']['num_channels'])):
    numpixel = config_inst['instrument'].pixel_spectral_domain(spec_idx).data.shape[0]
    new_spec_win_value[spec_idx, 0, 0] = 0
    new_spec_win_value[spec_idx, 0, 1] = numpixel + 1
new_bad_samp_mask[:,:] = False

spec_win_range.value = new_spec_win_value
config_inst['spec_win'].range_array = spec_win_range
config_inst['spec_win'].bad_sample_mask = new_bad_samp_mask

# Setup output 
h5out = 'land_fmout_%s.h5' % (case_id)
fout = h5py.File(h5out,'w')

# Forward model run
bnds = ['o2','weak_co2','strong_co2']
fm = config_inst['forward_model']
for spec_idx in range(int(config_inst['common']['num_channels'])):
    rdspc = fm.radiance(spec_idx,True)
    wvln = rdspc.spectral_domain.data
    l1bunc = config_inst['input']['l1b'].noise_model.uncertainty(spec_idx,rdspc.spectral_range.data)
    print(wvln.shape)
    print(wvln[100:105])

    flflt  = numpy.array([-9.999e6],dtype=numpy.float32)

    rdnm = '/SoundingMeasurements/noiseless_radiance_%s' % (bnds[spec_idx])
    dtrd  = fout.create_dataset(rdnm,data=rdspc.spectral_range.data)
    dtrd.attrs['missing_value'] = flflt
    dtrd.attrs['_FillValue'] = flflt

    sdnm = '/SoundingMeasurements/radiance_uncert_%s' % (bnds[spec_idx])
    dtrd  = fout.create_dataset(sdnm,data=l1bunc)
    dtrd.attrs['missing_value'] = flflt
    dtrd.attrs['_FillValue'] = flflt

    wvnm = '/SoundingMeasurements/wavelength_%s' % (bnds[spec_idx])
    dtrd  = fout.create_dataset(wvnm,data=wvln)
    dtrd.attrs['missing_value'] = flflt
    dtrd.attrs['_FillValue'] = flflt

dtsv = fout.create_dataset('/StateVector/true_state_vector',data=svdt['SVValue'].values)
dtsv.attrs['missing_value'] = flflt
dtsv.attrs['_FillValue'] = flflt
    
dtsc = fout.create_dataset('/StateVector/xco2',data=xco2)
dtsc.attrs['missing_value'] = flflt
dtsc.attrs['_FillValue'] = flflt
    
dt = h5py.special_dtype(vlen=str)
danm = fout.create_dataset('/StateVector/state_vector_names',(nst,),dtype=dt)
danm[...] = svdt['SVName'].values 

dtsc = fout.create_dataset('/SoundingGeometry/solar_zenith_angle',data=config_inst['input']['l1b'].solar_zenith(0).value)
dtsc.attrs['missing_value'] = flflt
dtsc.attrs['_FillValue'] = flflt
    
fout.close()

print(1.0e6 * xco2)
print(config_inst['input']['l1b'].solar_zenith(0).value)
