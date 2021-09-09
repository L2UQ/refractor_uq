# Construct a fixed state experiment 

import sys
import os
import pandas
import csv
import numpy
import h5py
import refractor_uq
from numpy import random, ndarray, linalg

def cov2cor(cvmt):
    d = 1.0 / numpy.sqrt(cvmt.diagonal())
    d1 = numpy.diag(d)
    t1 = numpy.dot(d1,cvmt)
    crmt = numpy.dot(t1,d1)
    return crmt

def unpackcov(pckmat,nelm):
    # Unpack a vectorized lower-triangle of a covariance matrix
    x0 = 1 + numpy.zeros((nelm,nelm))
    xpck = numpy.triu(x0)
    x2 = ndarray.flatten(xpck)
    x2[x2 == 1.0] = pckmat
    x2.shape = (nelm,nelm)
    diagsv = numpy.diagonal(x2)
    x2l = numpy.tril(numpy.transpose(x2),-1)
    xout = x2l + x2
    return xout

rdsd = 231112 
print(rdsd)
random.seed(rdsd)

ref_sounding_id = "2015080820071304"
l1b_file = "oco2_L1bScND_05862a_150808_B10003r_191117045059.h5"
met_file = "oco2_L2MetND_05862a_150808_B10003r_191117035013.h5"

# Start with prior from example
csvfl = 'land_state_%s.csv' % (ref_sounding_id)
svdt = pandas.read_csv(csvfl, dtype={'SVName':str, 'SVValue':float}, encoding='utf-8-sig')
nst = svdt.shape[0]

# Unimodal covariance
bslnfl = 'land_baseline_marginal_distribution_refractor.h5'
f = h5py.File(bslnfl)
bscv = f['/marginal_covariance_matrix'][:,:]
f.close()

tsmp = 5000
prmn = svdt['SVValue']
stnms = svdt['SVName']

# Linear dispersion from L1B
sdspt = refractor_uq.oco2_sounding_idx_match(int(ref_sounding_id),l1b_file)
print(sdspt)
f = h5py.File(l1b_file)
dspcf = f['/InstrumentHeader/dispersion_coef_samp'][:,sdspt[1],:]
f.close()
print(dspcf)

# Set linear dispersion to calib settings, fluorescence to zero
#prmn[56:58] = 0.0
prmn[42] = dspcf[0,1]
prmn[44] = dspcf[1,1]
prmn[46] = dspcf[2,1]

print(stnms)
print(prmn)

# State Order for ReFRACtor
# CO2
# H2O 
# PSfc
# Temp
# Aerosol
# Lambertian
# Dispersion
# EOF
# Fluorescence

# Read in ReFRACtor scale factors
df = pandas.read_csv("land_state_scaling_refractor.csv", \
                     dtype = {'fp_name':str, 'sv_scale':float, 'fpsim_position':int, 'surr_position':int, \
                              'prior_adjust':str, 'pow_trans':int, 'alt_source':str, 'alt_prior':str, \
                              'alt_prior_unc':float, 'surr_name':str, 'scene_group':str })
dffp = df.loc[(df['fpsim_position'] >= 0)]
prmn = prmn * dffp['sv_scale'].values

# Set up states
s1 = numpy.sqrt(numpy.diagonal(bscv))
crmt = cov2cor(bscv)
sdmt = numpy.diag(numpy.sqrt(bscv.diagonal()))

nst = prmn.shape[0]
print(nst)
dtall = numpy.zeros((tsmp,nst),dtype=numpy.float)
dtz = random.multivariate_normal(numpy.zeros((nst,)),crmt,size=tsmp)
dttmp = numpy.tile(prmn,(tsmp,1)) + numpy.dot(dtz,sdmt)
dtall[:,:] = dttmp[:,:]

sim_sdg_hdr = int(ref_sounding_id[0:10])
sdgid = sim_sdg_hdr * 1e6 + numpy.arange(tsmp,dtype=numpy.int64) + 1

# ReFRACtor state
nstrf = dffp.shape[0]
dtrfc = numpy.zeros((tsmp,nstrf),dtype=numpy.float)

for j in range(nstrf):
    fidx = dffp['fpsim_position'].values[j]
    print(fidx)
    dtrfc[:,j] = dtall[:,fidx] / dffp['sv_scale'].values[j]

# Fix linear dispersion
dtrfc[:,42] = dspcf[0,1]
dtrfc[:,44] = dspcf[1,1]
dtrfc[:,46] = dspcf[2,1]

# No EOF effect
dtrfc[:,47:56] = 0.0

print(prmn)
outstr = 'lnd_nadir_201508_refractor_state_vectors.h5'
f = h5py.File(outstr,'w')
stvr = f.create_dataset('true_state_vector',data=dtrfc)
sdvr = f.create_dataset('sounding_id',data=sdgid)
f.close()

