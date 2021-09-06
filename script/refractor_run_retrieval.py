# Run retrieval with simulation 
# Read from UQ example L1B

import os
import sys
sys.path.append(os.environ['OCO_CONFIG_DIR'])

from refractor.factory import process_config
from refractor import framework as rf
from retrieval_config import retrieval_config_definition

from pprint import pprint
import pandas
import numpy
import h5py
import refractor_uq

ref_sounding_id = "2015080820071304"
sim_sdg_hdr = int(ref_sounding_id[0:10])
sim_l1b = 'lnd_nadir_refractor_expt_l1b_uqscene.h5'

sidx = int(sys.argv[1]) 
l1b_file = sim_l1b
met_file = sim_l1b

# Sounding ID
f = h5py.File(sim_l1b,'r')
sounding_id = f['/SoundingGeometry/sounding_id'][sidx,0]
f.close()

if os.path.exists(l1b_file):
    print('L1B present')
else:
    print('L1B not found')

if os.path.exists(met_file):
    print('Met present')
else:
    print('Met not found')

# Read in apriori
csvfl = 'land_state_%s.csv' % (ref_sounding_id)
svdt = pandas.read_csv(csvfl, dtype={'SVName':str, 'SVValue':float}, encoding='utf-8-sig')
nst = svdt.shape[0]
xin = svdt['SVValue']

print(sounding_id)
sndtxt = '%d' % (sounding_id)
config_def = retrieval_config_definition(l1b_file, met_file, sndtxt)
config_inst = process_config(config_def)


fm = config_inst['forward_model']
sv = config_inst['retrieval']['state_vector']
pprint(sv)
print(config_inst['retrieval']['initial_guess'].shape)

# Update initial guess, apriori 
aprorig = config_inst['retrieval']['initial_guess']
config_inst['retrieval']['initial_guess'] = xin
pprint(config_inst['retrieval']['initial_guess'])
nstate = aprorig.shape[0]
config_inst['retrieval']['a_priori'] = xin
pprint(config_inst['retrieval']['a_priori'])


# Execute retrieval
solver = config_inst['retrieval']['solver']
pprint(solver)

solver.solve()
print(solver.status_str)
if ('SUCCESS' in solver.status_str):
    oflg = 1
else:
    oflg = 3

pprint(solver)
print(solver.cost_at_accepted_points)


for n, v in zip(sv.state_vector_name, sv.state):
    print("%.6e %s" % (v,n))

## Spectral information
print(config_inst['input']['l1b'].radiance(0).data.shape)
l1brd = config_inst['input']['l1b'].radiance(0).data

# Currently only spectral windows are defined, no bad samples
smpmsk = config_inst['spec_win'].bad_sample_mask
pprint(smpmsk)
print(smpmsk.shape)
spcwin = config_inst['spec_win'].range_array
pprint(spcwin)
print(spcwin.value)
print(spcwin.value.shape)
# Construct concatenated Level 1B, wavelengths, spectral samples, and modeled radiances, rad uncert

asq = numpy.arange(smpmsk.shape[1])
for j in range(3):
    j0 = int(j)
    rad = fm.radiance(j0,True)
    arg = numpy.arange(spcwin.value[j,0,0]-1,spcwin.value[j,0,1],dtype=numpy.int32)
    #print(arg[0:5])
    l1bsb = config_inst['input']['l1b'].radiance(j0).data[arg]
    l1bunc = config_inst['input']['l1b'].noise_model.uncertainty(j,config_inst['input']['l1b'].radiance(j0).data)[arg]
    lamsb = fm.spectral_grid.low_resolution_grid(j0).data
    yhat = rad.spectral_range.data
    print(l1bsb.shape)
    print(l1bunc.shape)
    if j == 0:
        l1bout = numpy.zeros( (arg.shape[0],), dtype=l1bsb.dtype)
        l1bout[:] = l1bsb
        lamout = numpy.zeros( (arg.shape[0],), dtype=lamsb.dtype)
        lamout[:] = lamsb
        yhatout = numpy.zeros( (arg.shape[0],), dtype=yhat.dtype)
        yhatout[:] = yhat
        rduncout = numpy.zeros( (arg.shape[0],), dtype=l1bunc.dtype)
        rduncout[:] = l1bunc
        print(' A-Band Lambda[200] (python)')
        lspt = int(200 - spcwin.value[j,0,0] + 1)
        print(lamsb[lspt])
         
    else:
        l1bout = numpy.append(l1bout,l1bsb)
        lamout = numpy.append(lamout,lamsb)
        yhatout = numpy.append(yhatout,yhat)
        rduncout = numpy.append(rduncout,l1bunc)

# Output
l2fl = '%s/output/l2_refractor_%d.h5' % (os.getcwd(),sounding_id)
refractor_uq.retrieval_l2_output(l2fl,sounding_id,sv.state,l1bout,yhatout,lamout,\
                                 rduncout,xin,oflg=oflg,niter=solver.num_accepted_steps)


