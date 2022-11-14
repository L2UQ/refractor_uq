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

ref_sounding_id = "2020082319555502"
l1b_file = "oco2_L1bScND_32686a_200823_B10206r_210506204351.h5"
met_file = "oco2_L2MetND_32686a_200823_B10206r_210506064119.h5"
l2_file = "oco2_L2DiaND_32686a_200823_B10206r_210905171547.h5"

# Obtain V10 a priori
f = h5py.File(l2_file,'r')
l2apr = f['/RetrievedStateVector/state_vector_apriori'][sdspt,:]
o2rfl = f['/BRDFResults/brdf_reflectance_o2'][sdspt]
wkrfl = f['/BRDFResults/brdf_reflectance_weak_co2'][sdspt]
strfl = f['/BRDFResults/brdf_reflectance_strong_co2'][sdspt]
f.close()


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

# Update prior to V10 values
svout = numpy.zeros(sv.state.shape, sv.state.dtype)
svout[0:20] = l2apr[0:20]     # CO2
svout[20:29] = l2apr[20:29]   # Met, 2 aerosols
svout[29:32] = l2apr[32:35]   
svout[32:35] = l2apr[29:32]   # Swap of water, ice cloud
svout[35] = o2rfl             # BRDF reflectance for Lambertian albedo
svout[37] = wkrfl             # These can be optional, ReFRACtor makes an internal continuum estimate
svout[39] = strfl
svout[41:47] = l2apr[47:53]   # Dispersion, linear
svout[56] = l2apr[62]         # SIF offset

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

