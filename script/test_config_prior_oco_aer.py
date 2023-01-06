# Setup simulation experiment L1b/Met 
# Based on load_example_config from oco repository
# Incorporate MERRA aerosol types

import os
import sys

sys.path.append(os.environ['OCO_CONFIG_DIR'])
from refractor.factory import process_config
from refractor import framework as rf
from retrieval_config import retrieval_config_definition
import refractor.factory.creator as creator

from pprint import pprint
import numpy
import h5py
import pandas
import refractor_uq

aerosol_prop_file = os.path.join(os.environ["REFRACTOR_INPUTS"], "l2_aerosol_combined.h5")
ref_sounding_id = "2020082319555502" 
l1b_file = "oco2_L1bScND_32686a_200823_B10206r_210506204351.h5"
met_file = "oco2_L2MetND_32686a_200823_B10206r_210506064119.h5" 

if os.path.exists(l1b_file):
    print('L1B present')
else:
    print('L1B not found')

if os.path.exists(met_file):
    print('Met present')
else:
    print('Met not found')

# Setup config, enable MERRA aerosols
merra_aer_list = ["DU","SS","BC","OC","SO"]
config_def = retrieval_config_definition(l1b_file, met_file, ref_sounding_id)
for j in range(len(merra_aer_list)):
    curtyp = merra_aer_list[j]
    config_def['atmosphere']['aerosol'][curtyp] = {
                    'creator': creator.aerosol.AerosolDefinition,
                    'extinction': {
                        'creator': creator.aerosol.AerosolShapeGaussian,
                        'value': numpy.array([-4.38203, 1, 0.2]),
                    },
                    'properties': {
                        'creator': creator.aerosol.AerosolPropertyHdf,
                        'filename': aerosol_prop_file,
                        'prop_name': curtyp,
                    },
                }
 
# Select types
config_def['atmosphere']['aerosol']['aerosols'] = [ "DU", "SO", "water", "ice" ]

config_inst = process_config(config_def)

sv = config_inst['retrieval']['state_vector']
pprint(sv)
print(sv.state_vector_name)
print(sv.state)
print(sv.state.shape)

# Forward model run
fm = config_inst['forward_model']
for spec_idx in range(int(config_inst['common']['num_channels'])):
    rdspc = fm.radiance(spec_idx,True)
    wvln = rdspc.spectral_domain.data
    print(wvln.shape)
    print(wvln[100:105])

# Write state to CSV
# Data frame with state vector
cfrm = pandas.DataFrame({'SVName': sv.state_vector_name, 'SVValue': sv.state})
csvfl = 'land_state_%s.csv' % (ref_sounding_id)
cfrm.to_csv(csvfl,index=False,encoding='utf-8')

