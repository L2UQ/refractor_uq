# Summarize ReFRACtor radiances for baseline forward model run

import numpy
import h5py
from numpy import ndarray

import matplotlib as mpl
mpl.use('agg')
import pylab
from matplotlib import pyplot
from matplotlib import colors
import matplotlib.ticker as mticker

# Sample indicators
clchc = ["#77B0F1","#81BB4E","#E991A1"]
clchcalp = ["#77B0F156","#81BB4E56","#E991A156"]
smpmn = [0, 1016, 2032]
smpmx = [1016, 2032, 3048]


bnms = [r'O$_2$ A-Band',r'Weak CO$_2$',r'Strong CO$_2$']
rdnms = ['o2','weak_co2','strong_co2']

# Control case radiances and wavelengths
rd_ctl = numpy.zeros((3048,))
unc_ctl = numpy.zeros((3048,))
wvln = numpy.zeros((3048,))

ptyp = 'psfc'
tinfo = 'Baseline'

ctlfl = 'land_fmout_control.h5'
f = h5py.File(ctlfl)
for j in range(3):
    sst0 = smpmn[j]
    sfn0 = smpmx[j]
    vrnm = '/SoundingMeasurements/noiseless_radiance_%s' % (rdnms[j])
    rd_ctl[sst0:sfn0] = f[vrnm][:] * 1.0e-20
    vrnm = '/SoundingMeasurements/radiance_uncert_%s' % (rdnms[j])
    unc_ctl[sst0:sfn0] = f[vrnm][:] * 1.0e-20
    wvnm = '/SoundingMeasurements/wavelength_%s' % (rdnms[j])
    wvln[sst0:sfn0] = f[wvnm][:]
f.close()

fig = pyplot.figure(figsize=(15,5))
for bnd in range(3):
    smpsq = numpy.arange(smpmn[bnd],smpmx[bnd])
    smpidx = smpsq - 1016 * bnd

    p1spt = bnd + 1
    p1 = pyplot.subplot(1,3,p1spt)
    pl1, = p1.plot(wvln[smpsq],rd_ctl[smpsq],ls='-',color=clchc[bnd])
    p1.xaxis.grid(color='#898989',linestyle='dotted')
    p1.yaxis.grid(color='#898989',linestyle='dotted')
    p1.set_xlabel(r'Wavelength [$\mu$m]',size=11)
    p1.set_ylabel('Radiance x 1e20',size=11)
    if bnd == 0:
        p1.set_xticks([0.758,0.760,0.762,0.764,0.766,0.768,0.770])
        p1.set_xticklabels(['','0.760','','0.764','','0.768',''])
    for lb in p1.xaxis.get_ticklabels():
        lb.set_fontsize(11)
    for lb in p1.yaxis.get_ticklabels():
        lb.set_fontsize(11)
    tstr = r'%s OCO-2 Radiance' % (bnms[bnd])
    pyplot.title(tstr)


fig.subplots_adjust(bottom=0.1,top=0.9,left=0.05,right=0.95, \
                    hspace=0.35,wspace=0.35)
fig.suptitle(tinfo,fontsize=14)
fnm = 'Refractor_OCO2_FMBaseline.png'
pyplot.savefig(fnm)
pyplot.close()
