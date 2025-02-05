# Add MERRA components to retrieval covariance 

import os
import h5py

covariance_file = os.path.join(os.environ['OCO_CONFIG_DIR'], "retrieval_covariance.h5")
merra_aer_list = ["DU","SS","BC","OC","SO"]

if os.path.isfile(covariance_file):
    f = h5py.File(covariance_file,'r+')
    aercv = f['/aerosol_extinction/gaussian_log/merra'][:,:]
    for k in range(len(merra_aer_list)):
        cvnm = '/aerosol_extinction/gaussian_log/%s' % (merra_aer_list[k])
        if cvnm in f:
            s1 = '%s found' % (cvnm)
            f[cvnm][...] = aercv
        else:
            s1 = '%s not found' % (cvnm)
            f.create_dataset(cvnm,data=aercv)
        print(s1)
    f.close()
    print(aercv)
else:
    print('Covariance file not found')

