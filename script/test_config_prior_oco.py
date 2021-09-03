# Setup simulation experiment L1b/Met 
# Based on load_example_config from oco repository

import os
import sys

sys.path.append(os.environ['OCO_CONFIG_DIR'])
from refractor.factory import process_config
from refractor import framework as rf
from retrieval_config import retrieval_config_definition

from pprint import pprint
import numpy
import h5py
import pandas
import refractor_uq

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

