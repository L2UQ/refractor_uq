# Module supporting UQ experiment setup and execution for ReFRACtor

import numpy
import h5py
import csv
import pandas
import os
import datetime
from numpy import ndarray, random, linalg

def oco2_sounding_idx_match_l2(sounding_id, l2_file):
    '''Match an OCO-2 sounding ID in a specified Level 2 file
       Return the sounding index location for sounding ID
       This information is used in other routines to extract L2 fields 
       for this reference sounding
    '''

    f = h5py.File(l2_file)
    sdgs = f['RetrievalHeader/sounding_id'][:]
    f.close()

    nsdg = sdgs.shape[0]
  
    sdrp = numpy.arange(nsdg)

    sdflt = sdrp.flatten()    
    idflt = sdgs.flatten()

    sdout = sdflt[idflt == sounding_id]

    if sdout.shape[0] > 0:
        idxout = sdout[0]
    else:
        idxout = None

    return idxout   
 
def oco2_sounding_idx_match(sounding_id, l1b_file):
    '''Match an OCO-2 sounding ID in a specified Level 1B file
       Return the sounding and footprint index location for sounding ID
       This information is used in other routines to extract L1B fields 
       for this reference sounding
    '''

    f = h5py.File(l1b_file)
    sdgs = f['/SoundingGeometry/sounding_id'][:,:]
    f.close()

    nsdg = sdgs.shape[0]
    nftp = sdgs.shape[1]
  
    fprp = numpy.tile(numpy.arange(nftp),(nsdg,1))
    sdrp = numpy.transpose(numpy.tile(numpy.arange(nsdg),(nftp,1)))

    fpflt = fprp.flatten()
    sdflt = sdrp.flatten()    
    idflt = sdgs.flatten()

    fpout = fpflt[idflt == sounding_id]
    sdout = sdflt[idflt == sounding_id]

    if ( (sdout.shape[0] > 0) and (fpout.shape[0] > 0) ):
        idxout = numpy.array([sdout[0],fpout[0]])
    else:
        idxout = None

    return idxout    

def process_field_list(csvfl):
    '''Read in a list of HDF5 fields and assemble into a data frame 
       Typically this is a collection of Level 1B fields 
    '''

    f = open(csvfl)
    csv_f = csv.reader(f)
    ctr = 0
    rlst = []
    for rw in csv_f:
        if ctr == 0:
            cnms = rw
        else:
            rlst.append(rw)
        ctr = ctr + 1
    df = pandas.DataFrame(rlst,columns = cnms)    

    return(df)

def setup_uq_l1b(uq_l1b_file,l1b_fields,met_fields,ref_idx,ref_l1b,ref_met,sdg_hdr,nsdg,save_noise=False,discrep=False, nst=59):
    '''Generate a UQ Level 1B file and fill with reference fields
         uq_l1b_file:  Name of output L1B file
         l1b_fields:   Data frame with level 1B fields
         met_fields:   Data frame with meteoroology fields
         ref_idx:      Reference sounding index in reference file
         ref_l1b:      Reference Level 1B file
         ref_met:      Reference meteorology file
         sdg_hdr:      Sounding ID header for output  
         nsdg:         Number of soundings for UQ
         save_noise:   Option to save noise standard deviations
         discrep:      Option to simulate/save model discrepancy
         nst:          Dimension of state vector
    '''

    # Above arguments could become a structure/dictionary
    # SoundingGeometry/sounding_id

    # Setup UQ L1B, soundings, placeholders for radiances
    sdsq = numpy.arange(1,nsdg+1,dtype=numpy.int64) 
    sdout = sdg_hdr*1e6 + sdsq
    sdout.shape = (nsdg,1)
    print(sdout[15:18])
    fout = h5py.File(uq_l1b_file,'w')
    dfsdg = fout.create_dataset('SoundingGeometry/sounding_id',data=sdout)
   
    flflt = numpy.array([-9.999e6],dtype=numpy.float32)
    dtrd = numpy.zeros((nsdg,1,1016),dtype=numpy.float32) + flflt

    dfrd = fout.create_dataset('SoundingMeasurements/radiance_o2',data=dtrd)
    dfrd.attrs['missing_value'] = flflt
    dfrd.attrs['_FillValue'] = flflt
    dfrd = fout.create_dataset('SoundingMeasurements/radiance_weak_co2',data=dtrd)
    dfrd.attrs['missing_value'] = flflt
    dfrd.attrs['_FillValue'] = flflt
    dfrd = fout.create_dataset('SoundingMeasurements/radiance_strong_co2',data=dtrd)
    dfrd.attrs['missing_value'] = flflt
    dfrd.attrs['_FillValue'] = flflt
 
    dfrd = fout.create_dataset('SoundingMeasurements/noiseless_radiance_o2',data=dtrd)
    dfrd.attrs['missing_value'] = flflt
    dfrd.attrs['_FillValue'] = flflt
    dfrd = fout.create_dataset('SoundingMeasurements/noiseless_radiance_weak_co2',data=dtrd)
    dfrd.attrs['missing_value'] = flflt
    dfrd.attrs['_FillValue'] = flflt
    dfrd = fout.create_dataset('SoundingMeasurements/noiseless_radiance_strong_co2',data=dtrd)
    dfrd.attrs['missing_value'] = flflt
    dfrd.attrs['_FillValue'] = flflt
 
    if save_noise:
        dfrd = fout.create_dataset('SoundingMeasurements/radiance_errsd_o2',data=dtrd)
        dfrd.attrs['missing_value'] = flflt
        dfrd.attrs['_FillValue'] = flflt
        dfrd = fout.create_dataset('SoundingMeasurements/radiance_errsd_weak_co2',data=dtrd)
        dfrd.attrs['missing_value'] = flflt
        dfrd.attrs['_FillValue'] = flflt
        dfrd = fout.create_dataset('SoundingMeasurements/radiance_errsd_strong_co2',data=dtrd)
        dfrd.attrs['missing_value'] = flflt
        dfrd.attrs['_FillValue'] = flflt
        
    if discrep:
        dfrd = fout.create_dataset('SoundingMeasurements/radiance_offset_o2',data=dtrd)
        dfrd.attrs['missing_value'] = flflt
        dfrd.attrs['_FillValue'] = flflt
        dfrd = fout.create_dataset('SoundingMeasurements/radiance_offset_weak_co2',data=dtrd)
        dfrd.attrs['missing_value'] = flflt
        dfrd.attrs['_FillValue'] = flflt
        dfrd = fout.create_dataset('SoundingMeasurements/radiance_offset_strong_co2',data=dtrd)
        dfrd.attrs['missing_value'] = flflt
        dfrd.attrs['_FillValue'] = flflt
        
    dtst = numpy.zeros((nsdg,nst),dtype=numpy.float32) + flflt
    dfst = fout.create_dataset('StateVector/true_state_vector',data=dtst)
    dfst.attrs['missing_value'] = flflt
    dfst.attrs['_FillValue'] = flflt

    # Pull in reference data
    frf = h5py.File(ref_l1b,'r')
    for index, row in l1b_fields.iterrows():
        if ( (row['dims'] == 3) and (row['sdgspec'] == 'Yes')):
            l1bdt = frf[row['variable']][ref_idx[0],ref_idx[1],:]
            l1bout = numpy.tile(l1bdt,(nsdg,1,1)) 
            print(row['variable'])
            print(l1bout.dtype)
            print(l1bout.shape)
            print(l1bout[0:2,0,:])
            dfl13 = fout.create_dataset(row['variable'],data=l1bout)
        elif ((row['dims'] == 2) and (row['sdgspec'] == 'Yes') ):
            l1bdt = frf[row['variable']][ref_idx[0],ref_idx[1]]
            l1bout = numpy.tile(l1bdt,(nsdg,1)) 
            dfl13 = fout.create_dataset(row['variable'],data=l1bout)
        elif ((row['dims'] == 4) and (row['sdgspec'] == 'Yes') ):
            l1bdt = frf[row['variable']][ref_idx[0],ref_idx[1],:,:]
            l1bout = numpy.tile(l1bdt,(nsdg,1,1))
            l1bout.shape = (nsdg,1,l1bdt.shape[0],l1bdt.shape[1]) 
            dfl13 = fout.create_dataset(row['variable'],data=l1bout)
        elif ((row['dims'] == 3) and (row['sdgspec'] == 'No') ):
            l1bdt = frf[row['variable']][:,ref_idx[1],:]
            l1bout = numpy.tile(l1bdt,(1,1)) 
            l1bout.shape = (l1bdt.shape[0],1,l1bdt.shape[1])
            if (row['variable'] == '/InstrumentHeader/bad_sample_list'):
                l1bout[:,:,:] = 0
            dfl13 = fout.create_dataset(row['variable'],data=l1bout)
        elif ((row['dims'] == 4) and (row['sdgspec'] == 'No') ):
            l1bdt = frf[row['variable']][:,ref_idx[1],:,:]
            l1bout = numpy.tile(l1bdt,(1,1)) 
            l1bout.shape = (l1bdt.shape[0],1,l1bdt.shape[1],l1bdt.shape[2])
            dfl13 = fout.create_dataset(row['variable'],data=l1bout)
    frf.close()

    fmt = h5py.File(ref_met,'r')
    for index, row in met_fields.iterrows():
        if row['dims'] == 3:
            metdt = fmt[row['variable']][ref_idx[0],ref_idx[1],:]
            metout = numpy.tile(metdt,(nsdg,1,1)) 
            dfl13 = fout.create_dataset(row['variable'],data=metout)
        elif row['dims'] == 4:
            metdt = fmt[row['variable']][ref_idx[0],ref_idx[1],:,:]
            metout = numpy.tile(metdt,(nsdg,1,1,1)) 
            dfl13 = fout.create_dataset(row['variable'],data=metout)
        elif row['dims'] == 2:
            metdt = fmt[row['variable']][ref_idx[0],ref_idx[1]]
            metout = numpy.tile(metdt,(nsdg,1)) 
            dfl13 = fout.create_dataset(row['variable'],data=metout)
        # Aerosol metadata
        dt = h5py.special_dtype(vlen=str)
        if ( (row['dims'] == 1) and (row['type'] == 'char')):
            chrdt = fmt[row['variable']][:]
            chrdcd = numpy.strings.decode(chrdt,encoding='ascii')
            chrlst = chrdcd.tolist()
            nchr = chrdt.shape[0]
            strfmt = 'S%d' % (row['bytes'])
            print(chrlst)
            print(strfmt)
            
            tid = h5py.h5t.FORTRAN_S1.copy()
            tid.set_size(row['bytes'])
            tid.set_strpad(h5py.h5t.STR_NULLTERM)
            c3dtyp = h5py.Datatype(tid)

            dt1 = fout.create_dataset(row['variable'], shape=(nchr,), dtype=c3dtyp, data=chrlst)
    fmt.close()

    fout.close()

    return

def retrieval_l2_output(uq_l2_file,sounding_id,ret_state,meas_rad,mod_rad,wvln=None,rdunc=None,prior=None,oflg=None,niter=None):
    '''Generate a UQ Level 2 single sounding output file
       This is a bare bones refractor result that can be combined with other 
       standard OCO-2 diagnostics later 
         uq_l2_file:   Name of output L2 file
         sounding_id:  Sounding ID
         ret_state:    Retrieved state vector
         meas_rad:     Measured radiance
         mod_rad:      Modeled radiance
         wvln:         Wavelengths
         rad_unc:      Radiance uncertainty
         prior:        Prior mean
         oflg:         Convergence/outcome flag
         niter:        Number of iterations
    '''

    # Also a priori and cov

    fout = h5py.File(uq_l2_file,'w')

    sdgarr = numpy.array(sounding_id,dtype=numpy.int64)
    dfsdg = fout.create_dataset('RetrievalHeader/sounding_id',data=sdgarr)
   
    flflt = numpy.array([-9.999e6],dtype=numpy.float32)
    flshr = numpy.array([-99],dtype=numpy.int16)
 
    #dtst = numpy.zeros((nsdg,nst),dtype=numpy.float32) + flflt
    dfst = fout.create_dataset('RetrievalResults/retrieved_state_vector',data=ret_state)
    dfst.attrs['missing_value'] = flflt
    dfst.attrs['_FillValue'] = flflt

    dfrd = fout.create_dataset('SpectralParameters/measured_radiance',data=meas_rad)
    dfrd.attrs['missing_value'] = flflt
    dfrd.attrs['_FillValue'] = flflt

    dfyh = fout.create_dataset('SpectralParameters/modeled_radiance',data=mod_rad)
    dfyh.attrs['missing_value'] = flflt
    dfyh.attrs['_FillValue'] = flflt

    if wvln is not None:
        dflm = fout.create_dataset('SpectralParameters/wavelength',data=wvln)
        dflm.attrs['missing_value'] = flflt
        dflm.attrs['_FillValue'] = flflt

    if rdunc is not None:
        dflm = fout.create_dataset('SpectralParameters/measured_radiance_uncert',data=rdunc)
        dflm.attrs['missing_value'] = flflt
        dflm.attrs['_FillValue'] = flflt

    if prior is not None:
        dfst = fout.create_dataset('RetrievalResults/state_vector_apriori',data=ret_state)
        dfst.attrs['missing_value'] = flflt
        dfst.attrs['_FillValue'] = flflt

    if oflg is not None:
        oarr = numpy.array([oflg],dtype=numpy.int16)
        dfof = fout.create_dataset('RetrievalResults/outcome_flag',data=oarr)
        dfof.attrs['missing_value'] = flshr
        dfof.attrs['_FillValue'] = flshr

    if niter is not None:
        itarr = numpy.array([niter],dtype=numpy.int16)
        dfitr = fout.create_dataset('RetrievalResults/number_iterations',data=itarr)
        dfitr.attrs['missing_value'] = flshr
        dfitr.attrs['_FillValue'] = flshr

    fout.close()

    return

def oco2_mapping_list(sounding_id, map_dirs, search_list):
    '''Assemble the collection of supporting data files for sounding ID
       This information is used in other routines  
    '''
  
    yrmndy = int(numpy.floor(sounding_id * 1e-8))
    sdyr = int(yrmndy / 1e4)
    rmdr = int(yrmndy % 1e4)
    sdmn = int(rmdr / 1e2)
    sddy = int(rmdr % 1e2)
    crdy = datetime.datetime(sdyr,sdmn,sddy) 
    stdy = datetime.datetime(sdyr,sdmn,sddy) 

    # Some hour info
    hrtm = (sounding_id * 1e-8) - yrmndy

    dctout = {}
    # Get the collection
    kylst = list(map_dirs.keys())
    for k in range(len(map_dirs.keys() )):
        crky = kylst[k]
        sdfd = -1
        dctr = 0
        while ( (sdfd < 0) and (dctr < 2) ):
            if ( (dctr == 1) and (hrtm < 0.5) ):
                crdy = stdy + datetime.timedelta(days=-1)
            elif ( (dctr == 1) and (hrtm >= 0.5) ):
                crdy = stdy + datetime.timedelta(days=1)
            else:
                crdy = stdy + datetime.timedelta(days=0)
            sspt = -1
            j = 0
            while ( (sdfd < 0) and (j < len(search_list)) ):
                drchk = '%s/%04d/%02d/%02d/%s' % (search_list[j],crdy.year,crdy.month,crdy.day,map_dirs[crky])
                if (os.path.isdir(drchk)):
                    fllst = os.listdir(drchk)
                    q = 0
                    while ( (sdfd < 0) and (q < len(fllst)) ):
                        if (".h5" in fllst[q]):
                            flh5 = '%s/%s' % (drchk,fllst[q])
                            if ( (map_dirs[crky] == 'L1bSc') or (map_dirs[crky] == 'L2Met') \
                                 or (map_dirs[crky] == 'L2ABP') or (map_dirs[crky] == 'L2IDP') or (map_dirs[crky] == 'L2CPr') ): 
                                l1bout = oco2_sounding_idx_match(sounding_id,flh5)
                                if l1bout is not None:
                                    sdfd = l1bout[0]
                                    dctout[crky] = flh5
                            elif ( (map_dirs[crky] == 'L2Dia') or (map_dirs[crky] == 'L2Std') ): 
                                l2out = oco2_sounding_idx_match_l2(sounding_id,flh5)
                                if l2out is not None:
                                    sdfd = l2out
                                    dctout[crky] = flh5
                        q = q + 1
                j = j + 1
            dctr = dctr + 1

    return dctout

def qts_mapping_list(sounding_id, map_dirs, search_list):
    '''Assemble the collection of supporting data files for sounding ID
       This routine uses QTS or other test data instead of operational products
       This information is used in other routines 
       
       sounding_id    - OCO-2/3 sounding ID to match
       map_dirs       - Data product dictionaries for mapping 
       search_list    - List of directories to search 
    '''
 
    yrmndy = int(numpy.floor(sounding_id * 1e-8))
    sdyr = int(yrmndy / 1e4)
    rmdr = int(yrmndy % 1e4)
    sdmn = int(rmdr / 1e2)
    sddy = int(rmdr % 1e2)
    crdy = datetime.datetime(sdyr,sdmn,sddy) 
    stdy = datetime.datetime(sdyr,sdmn,sddy) 

    # Some hour info
    hrtm = (sounding_id * 1e-8) - yrmndy

    # Here the products are in ad hoc directories, instead of by date
    # Loop through product types and directory list

    dctout = {}
    # Get the collection
    kylst = list(map_dirs.keys())
    for k in range(len(map_dirs.keys() )):
        crky = kylst[k]
        sdfd = -1
        j = 0
        while ( (sdfd < 0) and (j < len(search_list)) ):
            drchk = search_list[j] 
            if (os.path.isdir(drchk)):
                fllst = os.listdir(drchk)
                q = 0
                while ( (sdfd < 0) and (q < len(fllst)) ):
                    if (".h5" in fllst[q]):
                        flh5 = '%s/%s' % (drchk,fllst[q])
                        if ( (map_dirs[crky] == 'L1bSc') or (map_dirs[crky] == 'L2Met') \
                            or (map_dirs[crky] == 'L2ABP') or (map_dirs[crky] == 'L2IDP') or (map_dirs[crky] == 'L2CPr') ): 
                            l1bout = oco2_sounding_idx_match(sounding_id,flh5)
                            if l1bout is not None:
                                sdfd = l1bout[0]
                                dctout[crky] = flh5
                        elif ( (map_dirs[crky] == 'L2Dia') or (map_dirs[crky] == 'L2Std') ): 
                            l2out = oco2_sounding_idx_match_l2(sounding_id,flh5)
                            if l2out is not None:
                                sdfd = l2out
                                dctout[crky] = flh5
                    q = q + 1
            j = j + 1

    return dctout


def uq_expt_aggregate_l2(expt_scene_file,expt_agg_file,output_dir):
    '''Generate a OCO-2/3 UQ experiment aggregate output file 
         expt_scene_file:     Name of experiment scene file 
         expt_agg_file:       Name of experiment aggregate output file (OUTPUT)
         output_dir:          Location of individual retrieval output files
    '''

    # Get true states and sounding IDs from L1B file
    f = h5py.File(expt_scene_file,'r')
    sounding_id = f['/SoundingGeometry/sounding_id'][:,0]
    xtrue = f['/StateVector/true_state_vector'][:,:]
    f.close()

    nsnd = sounding_id.shape[0]
    nstate = xtrue.shape[1]
    flflt = numpy.array([-9.999e6],dtype=numpy.float32)
    flshrt = numpy.array([-99],dtype=numpy.int16)

    rtxco2 = numpy.zeros((nsnd,),dtype=numpy.float32) + flflt[0] 
    xhat = numpy.zeros((nsnd,nstate),dtype=numpy.float32) + flflt[0]
    chisqarr = numpy.zeros((nsnd,3),dtype=numpy.float32) + flflt[0]
    itrarr = numpy.zeros((nsnd,),dtype=numpy.int16) + flshrt[0]
    oflgarr = numpy.zeros((nsnd,),dtype=numpy.int16) + flshrt[0]

    pwts = numpy.zeros( (20,)) + 1.0 / 19.0
    pwts[0] = 1.0 / 38.0
    pwts[19] = 1.0 / 38.0
    print(numpy.sum(pwts))

    trxco2 = numpy.dot(xtrue[:,0:20],pwts)

    lmmn = [0.0, 1.0, 1.9]
    lmmx = [1.0, 1.9, 3.0]

    nsdout = 0
    for j in range(nsnd):
        l2nm = '%s/l2_refractor_%d.h5' % (output_dir,sounding_id[j])
        if (os.path.isfile(l2nm)):
            str1 = 'L2 Exists'
            f = h5py.File(l2nm,'r')
            xrtr = f['/RetrievalResults/retrieved_state_vector'][:]
            yobs = f['/SpectralParameters/measured_radiance'][:]
            yhat = f['/SpectralParameters/modeled_radiance'][:]
            yunc = f['/SpectralParameters/measured_radiance_uncert'][:]
            lam = f['/SpectralParameters/wavelength'][:]
            itr = f['/RetrievalResults/number_iterations'][:]
            oflg = f['/RetrievalResults/outcome_flag'][:]
            f.close()
            xhat[j,:] = xrtr[:]
            rtxco2[j] = numpy.dot(xrtr[0:20],pwts)
            itrarr[j] = itr[0]
            oflgarr[j] = oflg[0]
            spcsq = numpy.arange(yobs.shape[0])
            if nsdout == 0:
                nlam = yobs.shape[0]
                yobsout = numpy.zeros((nsnd,nlam),dtype=numpy.float32)
                yhatout = numpy.zeros((nsnd,nlam),dtype=numpy.float32)
                yuncout = numpy.zeros((nsnd,nlam),dtype=numpy.float32)
                lamout = numpy.zeros((nsnd,nlam),dtype=numpy.float32)
            yobsout[j,:] = yobs[:]
            yhatout[j,:] = yhat[:] 
            yuncout[j,:] = yunc[:]
            lamout[j,:] = lam[:]
            for bnd in range(3):
                sqsb = spcsq[(lam > lmmn[bnd]) & (lam <= lmmx[bnd])]
                nrmrsd = (yobs[sqsb] - yhat[sqsb]) / yunc[sqsb]
                print(nrmrsd.shape)
                print(nrmrsd[0:5])
                chisqarr[j,bnd] = numpy.mean(nrmrsd*nrmrsd)
            # Also spectral parameters
            nsdout = nsdout + 1
        else:
            str2 = 'No retrieval output for %d ' % (sounding_id[j])
            print(str2)
            print(l2nm)


    # Output groups: RetrievalResults, SpectralParameters


    fout = h5py.File(expt_agg_file,'w')

    #sdgarr = numpy.array(sounding_id,dtype=numpy.int64)
    dfsdg = fout.create_dataset('RetrievalHeader/sounding_id',data=sounding_id)
   
    # Retrieval Results
    dfst = fout.create_dataset('RetrievalResults/retrieved_state_vector',data=xhat)
    dfst.attrs['missing_value'] = flflt
    dfst.attrs['_FillValue'] = flflt

    dfst = fout.create_dataset('RetrievalResults/retrieved_xco2',data=rtxco2)
    dfst.attrs['missing_value'] = flflt
    dfst.attrs['_FillValue'] = flflt

    dfst = fout.create_dataset('RetrievalResults/number_iterations',data=itrarr)
    dfst.attrs['missing_value'] = flshrt
    dfst.attrs['_FillValue'] = flshrt

    dfst = fout.create_dataset('RetrievalResults/outcome_flag',data=oflgarr)
    dfst.attrs['missing_value'] = flshrt
    dfst.attrs['_FillValue'] = flshrt

    # True States
    dfst = fout.create_dataset('StateVector/true_state_vector',data=xtrue)
    dfst.attrs['missing_value'] = flflt
    dfst.attrs['_FillValue'] = flflt

    dfst = fout.create_dataset('StateVector/true_xco2',data=trxco2)
    dfst.attrs['missing_value'] = flflt
    dfst.attrs['_FillValue'] = flflt

    # Spectral Parameters
    dfo2 = fout.create_dataset('SpectralParameters/reduced_chi_squared_o2',data=chisqarr[:,0])
    dfo2.attrs['missing_value'] = flflt
    dfo2.attrs['_FillValue'] = flflt

    dfwk = fout.create_dataset('SpectralParameters/reduced_chi_squared_weak_co2',data=chisqarr[:,1])
    dfwk.attrs['missing_value'] = flflt
    dfwk.attrs['_FillValue'] = flflt

    dfst = fout.create_dataset('SpectralParameters/reduced_chi_squared_strong_co2',data=chisqarr[:,2])
    dfst.attrs['missing_value'] = flflt
    dfst.attrs['_FillValue'] = flflt

    if nsdout > 0:
        drad = fout.create_dataset('SpectralParameters/measured_radiance',data=yobsout)
        drad.attrs['missing_value'] = flflt
        drad.attrs['_FillValue'] = flflt

        drad = fout.create_dataset('SpectralParameters/measured_radiance_unc',data=yuncout)
        drad.attrs['missing_value'] = flflt
        drad.attrs['_FillValue'] = flflt

        drad = fout.create_dataset('SpectralParameters/modeled_radiance',data=yhatout)
        drad.attrs['missing_value'] = flflt
        drad.attrs['_FillValue'] = flflt

        drad = fout.create_dataset('SpectralParameters/wavelength',data=lamout)
        drad.attrs['missing_value'] = flflt
        drad.attrs['_FillValue'] = flflt

    fout.close()

    return

def cov2cor(cvmt):
    dgvc = numpy.diagonal(cvmt)
    dgsd = numpy.zeros(dgvc.shape,dtype=dgvc.dtype)
    dgsd[:] = dgvc[:]
    dgsd[dgsd == 0.0] = 1.0e-12
    d = 1.0 / numpy.sqrt(dgsd)
    d1 = numpy.diag(d)
    t1 = numpy.dot(d1,cvmt)
    crmt = numpy.dot(t1,d1)
    return crmt

def unpackcov(pckmat,nelm):
    # Unpack a vectorized lower-triangle of a covariance matrix
    x0 = 1 + numpy.zeros((nelm,nelm))
    xpck = numpy.triu(x0)
    x2 = xpck.flatten()
    x2[x2 == 1.0] = pckmat
    x2.shape = (nelm,nelm)
    diagsv = numpy.diagonal(x2)
    x2l = numpy.tril(numpy.transpose(x2),-1)
    xout = x2l + x2
    return xout

def setup_uq_expt_scene(scene_config,state_info_file,moderr=None,sdvl = 255115):
    '''Generate a UQ experiment scene file 
         scene_config:    Name of config CSV file 
         state_info_file: Name of state vector config CSV file
    '''

    random.seed(sdvl)
    if os.path.exists(scene_config):
        f = open(scene_config)
        csv_f = csv.reader(f)
        ctr = 0
        rlst = []
        for rw in csv_f:
            rlst.append(rw)
            if rw[0] == 'fp_scene_file':
                if ( (moderr) or (moderr is None) ):
                    outfile = rw[1]
            elif ( (rw[0] == 'fp_scene_file_noerr') and (not moderr) ):
                outfile = rw[1]
            elif rw[0] == 'marginal_variable_file':
                mrgfile = rw[1]
            elif rw[0] == 'prior_variable_file':
                prfile = rw[1]
            elif rw[0] == 'samp_per_process':
                tsmp = int(rw[1])
            elif rw[0] == 'fp_state_vector_file':
                svcfile = rw[1]
            elif rw[0] == 'fp_discrepancy_file':
                mdcfile = rw[1]
            elif rw[0] == 'fp_parameter_file':
                parfile = rw[1]
            elif rw[0] == 'surface_type':
                sfctyp = rw[1]
            elif rw[0] == 'acquisition_mode':
                aqmd = rw[1]
            ctr = ctr + 1
        f.close()

        if ( (aqmd == 'nadir') or (aqmd == 'Nadir') ):
            aqmdout = 'Sample Nadir'
        else:
            aqmdout = 'Sample Glint'
        amdstr = numpy.empty((1,),"|O")
        amdstr[0] = aqmdout

        # Parameters
        prmlst = ['specific_humidity_profile_ecmwf','surface_pressure_ecmwf','temperature_profile_ecmwf', \
                  'vector_pressure_levels_ecmwf','windspeed_u_ecmwf','windspeed_v_ecmwf', \
                  'footprint_altitude','footprint_azimuth','footprint_latitude','footprint_longitude', \
                  'footprint_solar_azimuth','footprint_solar_zenith','footprint_stokes_coefficients', \
                  'footprint_time_tai93','footprint_zenith', \
                  'sounding_land_fraction','sounding_relative_velocity', \
                  'sounding_latitude', 'sounding_longitude', \
                  'sounding_solar_distance','sounding_solar_relative_velocity','bad_sample_list', \
                  'dispersion_coef_samp','ils_delta_lambda','ils_relative_response','snr_coef']
        pgplst = ['ECMWF','ECMWF','ECMWF','ECMWF','ECMWF','ECMWF', \
                  'FootprintGeometry','FootprintGeometry','FootprintGeometry','FootprintGeometry', \
                  'FootprintGeometry','FootprintGeometry','FootprintGeometry','FootprintGeometry','FootprintGeometry', \
                  'SoundingGeometry','SoundingGeometry','SoundingGeometry','SoundingGeometry', \
                  'SoundingGeometry','SoundingGeometry','InstrumentHeader', \
                  'InstrumentHeader','InstrumentHeader','InstrumentHeader','InstrumentHeader']
        prmfrm = pandas.DataFrame({'HDFGroup':pgplst,'component':prmlst})

        f = open(parfile)
        csv_f = csv.reader(f)
        ctr = 0
        rlst = []
        for rw in csv_f:
            if ctr == 0:
                cnms = rw
            else:
                rlst.append(rw)
            ctr = ctr + 1
        f.close()
        pardf = pandas.DataFrame(rlst,columns = cnms)

        parmrg = pandas.merge(pardf,prmfrm,on="component")
        flflt = numpy.array([-9.999e6],dtype=numpy.float32)

        fout = h5py.File(outfile,'w')

        for i in range(parmrg.shape[0]):
            frd = h5py.File(parmrg['file_name'].values[i],'r')
            print(parmrg['variable_name'].values[i])
            tmpvr = frd.get(parmrg['variable_name'].values[i])
            if (parmrg['variable_name'].values[i] == '/InstrumentHeader/bad_sample_list'):
                print(tmpvr)
            tmpvl = numpy.array(tmpvr)
            frd.close()
            nmout = '/%s/%s' % (parmrg['HDFGroup'].values[i],parmrg['component'].values[i])
            dfscn = fout.create_dataset(nmout,data=tmpvl)
        dt = h5py.special_dtype(vlen=str)
        da2 = fout.create_dataset('/Metadata/AcquisitionMode',(1,),dtype=dt)
        da2[...] = amdstr 

        # State Vector
        f = open(svcfile)
        csv_f = csv.reader(f)
        ctr = 0
        rlst = []
        for rw in csv_f:
            if ctr == 0:
                cnms = rw
            else:
                rlst.append(rw)
            ctr = ctr + 1
        f.close()
        svcdf = pandas.DataFrame(rlst,columns = cnms)

        f = open(state_info_file)
        csv_f = csv.reader(f)
        ctr = 0
        rlst = []
        for rw in csv_f:
            if ctr == 0:
                cnms = rw
            else:
                rlst.append(rw)
            ctr = ctr + 1
        f.close()
        stindf = pandas.DataFrame(rlst,columns = cnms)

        stindf['fpsim_position'] = stindf['fpsim_position'].astype('int32')
        stindf['sv_scale'] = stindf['sv_scale'].astype('float32')
        stindf['pow_trans'] = stindf['pow_trans'].astype('int32')
        sbsq = stindf[(stindf['fpsim_position'] >= 0) & (stindf['scene_group'] != 'NA')]

        smsqfp = sbsq['fpsim_position'].tolist()

        svlfl = svcdf[svcdf['component'] == 'state_vector_values']['file_name'].values[0]
        svlvr = svcdf[svcdf['component'] == 'state_vector_values']['variable_name'].values[0]
        frd = h5py.File(svlfl,'r')
        tmpvr = frd.get(svlvr)
        tmpvl = numpy.array(tmpvr)
        frd.close()
        vcout = numpy.transpose(tmpvl[:,smsqfp])
        # Rescale
        for p in range(vcout.shape[0]):
            vcout[p,:] = vcout[p,:] / sbsq['sv_scale'].values[p]
            if (sbsq['pow_trans'].values[p] == 0):
                vcout[p,:] = numpy.exp(vcout[p,:])
        dfscn = fout.create_dataset('/StateVector/sampled_state_vectors',data=vcout)

        svlfl = svcdf[svcdf['component'] == 'sounding_id']['file_name'].values[0]
        svlvr = svcdf[svcdf['component'] == 'sounding_id']['variable_name'].values[0]
        frd = h5py.File(svlfl,'r')
        tmpvr = frd.get(svlvr)
        tmpvl = numpy.array(tmpvr)
        frd.close()
        dfscn = fout.create_dataset('/SoundingGeometry/sounding_id',data=tmpvl)

        svlfl = svcdf[svcdf['component'] == 'state_vector_names']['file_name'].values[0]
        svlvr = svcdf[svcdf['component'] == 'state_vector_names']['variable_name'].values[0]
        frd = h5py.File(svlfl,'r')
        tmpvr = frd[svlvr][smsqfp] 
        frd.close()
        for p in range(vcout.shape[0]):
            if ('BRDF' in tmpvr[p]): 
                tmpvr[p] = sbsq['surr_name'].values[p] 
        dfscn = fout.create_dataset('/StateVector/state_vector_names',data=tmpvr)

        # Prior
        if sfctyp == 'land':
            prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                      '/Ground/Albedo','/Instrument/Dispersion','/Fluorescence']
            prcmps =   [20, 1, 1, 1, 6, 3, 3, 6, 3, 2]
            prngrp = [ 1, 1, 1, 1, 2, 1, 1, 3, 1, 1]
        elif sfctyp == 'ocean':
            prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                      '/Ground/Windspeed', '/Ground/Albedo','/Instrument/Dispersion']
            prcmps =   [20, 1, 1, 1, 6, 3, 3, 1, 6, 3]
            prngrp = [ 1, 1, 1, 1, 2, 1, 1, 1, 3, 1]


        f = open(prfile)
        csv_f = csv.reader(f)
        ctr = 0
        rlst = []
        for rw in csv_f:
            if ctr == 0:
                cnms = rw
            else:
                rlst.append(rw)
            ctr = ctr + 1
        f.close()
        prdf = pandas.DataFrame(rlst,columns = cnms)

        aprfl = prdf[prdf['component'] == 'state_vector']['file_name'].values[0]
        aprvr = prdf[prdf['component'] == 'state_vector']['mean_variable_name'].values[0]
        aprcv = prdf[prdf['component'] == 'state_vector']['covariance_variable_name'].values[0]
        frd = h5py.File(aprfl,'r')
        tmpvr = frd.get(aprvr)
        tmpvl = numpy.array(tmpvr)
        tmpcv = frd.get(aprcv)
        cvvl = numpy.array(tmpcv)
        frd.close()

        vcapr = tmpvl[smsqfp] 
        aprfl = numpy.zeros((vcapr.shape[0],tsmp),dtype=vcapr.dtype)
        cvmt = unpackcov(cvvl,tmpvl.shape[0])
        cvsb = numpy.zeros((vcapr.shape[0],vcapr.shape[0]),dtype=numpy.float32)
        for c0 in range(len(smsqfp)):
            for c1 in range(len(smsqfp)):
                cvsb[c0,c1] = cvmt[smsqfp[c0],smsqfp[c1]]
        cvscl = numpy.zeros(cvsb.shape,dtype=cvsb.dtype)

        # Rescale
        for p in range(vcapr.shape[0]):
            aprfl[p,:] = vcapr[p] / sbsq['sv_scale'].values[p]
            if (sbsq['pow_trans'].values[p] == 0):
                aprfl[p,:] = numpy.exp(aprfl[p,:])
            for t in range(vcapr.shape[0]):
                cvscl[p,t] = cvsb[p,t] / (sbsq['sv_scale'].values[p] * sbsq['sv_scale'].values[t])
        dfscn = fout.create_dataset('/StateVector/a_priori',data=aprfl)

        print(cvscl[16:20,16:20])

        prunc = numpy.sqrt(numpy.diagonal(cvscl))
        dfscn = fout.create_dataset('/StateVector/a_priori_uncert',data=prunc)

        # Assign by group
        sqgrpstr = sbsq['scene_group'].values
        nstpr = vcapr.shape[0]
        gsq = numpy.arange(nstpr)

        for q in range(len(prgrps)):
            print(prgrps[q])
            gsb = gsq[sqgrpstr == prgrps[q]]

            if prngrp[q] == 1:
                aprtmp = numpy.zeros((prcmps[q],tsmp),dtype=aprfl.dtype)
                aprtmp[:,:] = aprfl[gsb,:]
                vrnmpr = '%s/a_priori' % (prgrps[q])
                if (aprtmp.shape[0] == 1):
                    aprtmp = aprtmp.flatten()
                dfscn = fout.create_dataset(vrnmpr,data=aprtmp)

                cvout = numpy.zeros((prcmps[q],prcmps[q]),dtype=cvscl.dtype)
                for c0 in range(gsb.shape[0]):
                    for c1 in range(gsb.shape[0]):
                        cvout[c0,c1] = cvscl[gsb[c0],gsb[c1]]
                vrnpcv = '%s/covariance' % (prgrps[q])
                dfscn = fout.create_dataset(vrnpcv,data=cvout)
                # Windspeed fix
                #if prgrps[q] == '/Ground/Windspeed':
                #    print('Wind speed state adjustment')
                #    wsfrm = sbsq[(sbsq['fp_name'] == 'Ground Coxmunk Windspeed')]
                #    widx = wsfrm['fpsim_position'].values[0]
                #    print(widx)
                #    vcout[widx,:] = aprtmp[:]
                #    svcvr = fout['/StateVector/sampled_state_vectors']
                #    svcvr[...] = vcout 
            else:
                dm0 = prngrp[q]
                dm1 = prcmps[q] / prngrp[q] 
                aprtmp = numpy.zeros((dm0,dm1,tsmp),dtype=aprfl.dtype)
                ct2 = 0
                for b0 in range(dm0):
                    for b1 in range(dm1):
                        aprtmp[b0,b1,:] = aprfl[gsb[ct2],:]
                        ct2 = ct2 + 1
                vrnmpr = '%s/a_priori' % (prgrps[q])
                dfscn = fout.create_dataset(vrnmpr,data=aprtmp)

                # Covariance, fill in blocks
                cvout = numpy.zeros((dm0,dm1,dm1),dtype=cvscl.dtype)
                ct0 = 0
                for b0 in range(dm0):
                    for c0 in range(dm1):
                        for c1 in range(dm1):
                            ct0 = b0 * dm1 + c0
                            ct1 = b0 * dm1 + c1
                            cvout[b0,c0,c1] = cvscl[gsb[ct0],gsb[ct1]]
                vrnpcv = '%s/covariance' % (prgrps[q])
                dfscn = fout.create_dataset(vrnpcv,data=cvout)
                # Lambertian fix
                #if prgrps[q] == '/Ground/Albedo':
                #    print('Lambertian albedo adjustment')
                #    albfx = numpy.array([0.02, 0.0, 0.02, 0.0, 0.02, 0.0])
                #    wsfrm = sbsq[(sbsq['fp_name'] == 'Ground Lambertian A-Band Albedo Parm 1')]
                #    widx = wsfrm['fpsim_position'].values[0]
                #    print(widx)
                #    wsq = numpy.arange(widx,widx+6)
                #    for ta0 in range(tsmp):
                #        vcout[wsq,ta0] = albfx[:]
                #    svcvr = fout['/StateVector/sampled_state_vectors']
                #    svcvr[...] = vcout 
               
        # Discrepancy 
        ndcmp = 30
        nchnl = 3048
        dscrp = numpy.zeros((nchnl,tsmp),dtype=numpy.float32)
        if ( (moderr) or (moderr is None) ):
            scrtot = numpy.zeros((nchnl,tsmp),dtype=numpy.float32)
            dcpfrm = pandas.read_csv(mdcfile, \
                                     dtype = {'component':str, 'file_name':str, \
                                              'variable_name':str, 'index':int}) 
            dscpfl = dcpfrm['file_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_mean']
            mnvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_sd']
            sdvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_basis']
            bsvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_basis_var']
            evvrnm = frmsb['variable_name'].values[0]
            
            print(dscpfl) 
            print(mnvrnm) 
            frd = h5py.File(dscpfl,'r')
            errmn = frd[mnvrnm][:]
            errsd = frd[sdvrnm][:]
            errbs = frd[bsvrnm][:,:]
            errev = frd[evvrnm][:]
            frd.close()
            print('Discrepancy parameters opened')

            chsq = numpy.arange(nchnl)
            efrm = pandas.DataFrame({'CmpSeq':chsq, 'EigenVal':errev})
            efrm = efrm.sort_values(by=['EigenVal'],ascending=[False])
            print(efrm[0:30])

            # Tile mean and std dev
            mnspc = numpy.transpose(numpy.tile(errmn,(5000,1)))
            #print(mnspc.shape)
            #print(mnspc[200:205,500])
            sdspc = numpy.transpose(numpy.tile(errsd,(5000,1)))

            for d1 in range(ndcmp):
                cidx = efrm['CmpSeq'].values[d1]
                scrsd = numpy.sqrt(errev[cidx])
                scrcmp = random.normal(scale=scrsd,size=tsmp)

                scrscz = numpy.outer(errbs[:,d1],scrcmp)
                scrtot = scrtot + scrscz

            print(scrscz.shape)
            dscrp = mnspc + sdspc * scrtot

        dfscn = fout.create_dataset('/SpectralParameters/sampled_radiance_offset',data=dscrp)

        fout.close()

    else:
        msg1 = 'Config not found'
        print(msg1)

    return

def uq_expt_aggregate_l1b(expt_scene_file,output_dir,discrep=False):
    '''Propagate individual forward model radiances to a UQ experiment result output file 
         expt_scene_file:     Name of experiment scene file 
         output_dir:          Location of individual retrieval output files
    '''

    # Get true states and sounding IDs from L1B file
    f = h5py.File(expt_scene_file,'r')
    sounding_id = f['/SoundingGeometry/sounding_id'][:,0]
    rdo2 = f['/SoundingMeasurements/radiance_o2'][:,0,:]
    rdwk = f['/SoundingMeasurements/radiance_weak_co2'][:,0,:]
    rdst = f['/SoundingMeasurements/radiance_strong_co2'][:,0,:]
    fmo2 = f['/SoundingMeasurements/noiseless_radiance_o2'][:,0,:]
    fmwk = f['/SoundingMeasurements/noiseless_radiance_weak_co2'][:,0,:]
    fmst = f['/SoundingMeasurements/noiseless_radiance_strong_co2'][:,0,:]
    xtr = f['/StateVector/true_state_vector'][:,:]
    if discrep:
        ofo2 = f['/SoundingMeasurements/radiance_offset_o2'][:,0,:]
        ofwk = f['/SoundingMeasurements/radiance_offset_weak_co2'][:,0,:]
        ofst = f['/SoundingMeasurements/radiance_offset_strong_co2'][:,0,:]
    f.close()

    nsnd = sounding_id.shape[0]
    #nstate = xtrue.shape[1]
    flflt = numpy.array([-9.999e6],dtype=numpy.float32)

    lmmn = [0.0, 1.0, 1.9]
    lmmx = [1.0, 1.9, 3.0]

    fout = h5py.File(expt_scene_file,'r+')

    for j in range(nsnd):
        l1bnm = '%s/l1b_%d.h5' % (output_dir,sounding_id[j])
        if (os.path.isfile(l1bnm)):
            f = h5py.File(l1bnm,'r')
            srdo2 = f['/SoundingMeasurements/radiance_o2'][:]
            srdwk = f['/SoundingMeasurements/radiance_weak_co2'][:]
            srdst = f['/SoundingMeasurements/radiance_strong_co2'][:]
            sfmo2 = f['/SoundingMeasurements/noiseless_radiance_o2'][:]
            sfmwk = f['/SoundingMeasurements/noiseless_radiance_weak_co2'][:]
            sfmst = f['/SoundingMeasurements/noiseless_radiance_strong_co2'][:]
            sxtr = f['/StateVector/true_state_vector'][:]
            if discrep:
                sofo2 = f['/SoundingMeasurements/radiance_offset_o2'][:]
                sofwk = f['/SoundingMeasurements/radiance_offset_weak_co2'][:]
                sofst = f['/SoundingMeasurements/radiance_offset_strong_co2'][:]
            f.close()

            # Update
            rdo2[j,:] = srdo2[:]
            rdwk[j,:] = srdwk[:]
            rdst[j,:] = srdst[:]
            fmo2[j,:] = sfmo2[:]
            fmwk[j,:] = sfmwk[:]
            fmst[j,:] = sfmst[:]
            if discrep:
                ofo2[j,:] = sofo2[:]
                ofwk[j,:] = sofwk[:]
                ofst[j,:] = sofst[:]
            xtr[j,:] = sxtr[:]
        else:
            str2 = 'No retrieval output for %d ' % (sounding_id[j])
            print(str2)
            print(l1bnm)

    varo2 = fout['/SoundingMeasurements/radiance_o2']
    varo2[:,0,:] = rdo2[:,:]
    varwk = fout['/SoundingMeasurements/radiance_weak_co2']
    varwk[:,0,:] = rdwk[:,:]
    varst = fout['/SoundingMeasurements/radiance_strong_co2']
    varst[:,0,:] = rdst[:,:]

    varo2 = fout['/SoundingMeasurements/noiseless_radiance_o2']
    varo2[:,0,:] = fmo2[:,:]
    varwk = fout['/SoundingMeasurements/noiseless_radiance_weak_co2']
    varwk[:,0,:] = fmwk[:,:]
    varst = fout['/SoundingMeasurements/noiseless_radiance_strong_co2']
    varst[:,0,:] = fmst[:,:]

    if discrep:
        varo2 = fout['/SoundingMeasurements/radiance_offset_o2']
        varo2[:,0,:] = ofo2[:,:]
        varwk = fout['/SoundingMeasurements/radiance_offset_weak_co2']
        varwk[:,0,:] = ofwk[:,:]
        varst = fout['/SoundingMeasurements/radiance_offset_strong_co2']
        varst[:,0,:] = ofst[:,:]

    varx = fout['/StateVector/true_state_vector']
    varx[:,:] = xtr[:,:]

    fout.close()

    return

def quantile_msgdat(vcdat, probs, msgval=-9999.):
    # Compute quantiles with missing data
    if (numpy.amax(probs) <= 1.0):
        prb100 = 100.0 * probs
    else:
        prb100 = probs

    dtsb = vcdat[vcdat != msgval]

    if dtsb.shape[0] > 0:
        qsout = numpy.percentile(dtsb,q=prb100)
    else:
        qsout = numpy.zeros(probs.shape) + msgval
    return qsout

def quantile_msgdat_discrete(vcdat, probs, msgval=-99):
    # Compute quantiles with missing data, discrete version
    if (numpy.amax(probs) <= 1.0):
        prb100 = 100.0 * probs
    else:
        prb100 = probs

    dtsb = vcdat[vcdat != msgval]

    if dtsb.shape[0] > 0:
        qsout = numpy.percentile(dtsb,q=prb100,interpolation='nearest')
    else:
        qsout = numpy.zeros(probs.shape) + msgval
    return qsout

def inverse_cdf_from_obs_fill_msg(vcdat, obsqs, probs, msgval=-9999.):
    # Compute transform from observed quantiles to probabilities via inverse CDF 

    nprb = probs.shape[0]
    ndat = vcdat.shape[0]
    dsq = numpy.arange(ndat)
    vsq = dsq[vcdat != msgval]
    msq = dsq[vcdat == msgval]
    nvld = vsq.shape[0]
    nmsg = msq.shape[0]
    zout = numpy.zeros((ndat,)) + msgval

    if (obsqs.shape != probs.shape):
        print("Input and output quantile lengths must match")
    elif (nvld == 0):
        print("All observations missing, no transformation performed")
    else:
        ptst = numpy.append(0.0,numpy.append(probs,1.0))
        etst = numpy.append(-numpy.inf,numpy.append(obsqs,numpy.inf))
        qsq = numpy.arange(ptst.shape[0])

        # Matrices
        ntst = etst.shape[0]
        dtmt = numpy.tile(vcdat[vsq],(ntst,1))
        etmt = numpy.transpose(numpy.tile(etst,(nvld,1)))

        # Indicators for breakpoints of empirical CDF
        lwind = (etmt < dtmt)
        hiind = (dtmt < etmt)

        smlw = numpy.sum(lwind,axis=0) - 1
        smhi = ntst - numpy.sum(hiind,axis=0)

        # Find probability spot
        prbdif = ptst[smhi] - ptst[smlw]
        pspt = ptst[smlw] + prbdif * random.uniform(size=nvld)
        #print(pspt[505:510])

        zout[vsq] = pspt 

        # Missing
        if nmsg > 0:
            psptm = random.uniform(size=nmsg)
            zout[msq] = psptm

    return zout

def clean_byte_list(btlst):
    clean = [x for x in btlst if x != None]
    strout = b''.join(clean).decode('utf-8')
    return strout

def setup_uq_expt_scene_ref(outfile,scene_config,state_info,state_array,prior_mean,state_names, \
                            sdg_ids,nreps,l1bfrm, metfrm, ref_idx,ref_l1b,ref_met, \
                            moderr=None,sdvl = 255115, mxcmp=None):
    '''Generate a UQ experiment scene file given state vectors and 
       reference sounding information
         outfile:         Name of output scene file
         scene_config:    Name of config CSV file 
         state_info:      State scaling data frame 
         state_array:     Matrix of state vectors
         prior_mean:      Prior mean vector
         state_names:     State vector names
         sdg_ids:         Vector of sounding IDs
         nreps:           Number of state vector replicates (1st dimension of state_array)
         l1bfrm:          Data frame with L1B fields and properties
         metfrm:          Data frame with meteorology fields and properties
         ref_idx:         Sounding index in L1B/Met files
         ref_l1b:         Reference L1B file
         ref_met:         Reference Meteorology file
         moderr:          Logical for sampling model discrepancy
         sdvl:            Random seed
         mxcmp:           Mixture component array
    '''

    tmpsd = sdvl
    if os.path.exists(scene_config):
        f = open(scene_config)
        csv_f = csv.reader(f)
        ctr = 0
        rlst = []
        for rw in csv_f:
            rlst.append(rw)
            if rw[0] == 'fp_discrepancy_file':
                mdcfile = rw[1]
            elif rw[0] == 'surface_type':
                sfctyp = rw[1]
            elif rw[0] == 'acquisition_mode':
                aqmd = rw[1]
            elif rw[0] == 'random_seed':
                tmpsd = int(rw[1])
            ctr = ctr + 1
        f.close()

        random.seed(tmpsd)

        amdstr = state_names[0:2].copy()
        if ( (aqmd == 'nadir') or (aqmd == 'Nadir') ):
            amdstr[0] = 'Sample Nadir'
        else:
            amdstr[0] = 'Sample Glint'

        # Met/L1b parameters from data frames 
        flflt = numpy.array([-9.999e6],dtype=numpy.float32)

        fout = h5py.File(outfile,'w')

        f = h5py.File(ref_l1b)
        for index, row in l1bfrm.iterrows():
            if ( (row['dims'] == 3) and (row['sdgspec'] == 'Yes')):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1],0]
                l1bout = numpy.tile(l1bdt,(1,)) 
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 2) and (row['sdgspec'] == 'Yes') ):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1]]
                l1bout = numpy.tile(l1bdt,(1,))
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 4) and (row['sdgspec'] == 'Yes') ):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1],0,:]
                l1bout = numpy.tile(l1bdt,(1,))
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 3) and (row['sdgspec'] == 'No') ):
                l1bdt = f[row['variable']][:,ref_idx[1],:]
                if (row['variable'] == '/InstrumentHeader/bad_sample_list'):
                    l1bdt[:,:] = 0
                dfl13 = fout.create_dataset(row['variable'],data=l1bdt)
            elif ((row['dims'] == 4) and (row['sdgspec'] == 'No') ):
                l1bdt = f[row['variable']][:,ref_idx[1],:,:]
                dfl13 = fout.create_dataset(row['variable'],data=l1bdt)
        f.close()

        # Met fields
        fmt = h5py.File(ref_met)
        # Get aerosol types
        #aertyp = fmt['/Metadata/CompositeAerosolTypes'][:]
        aertyp = state_names[0:5].copy() 
        aertyp[0] = "DU"
        aertyp[1] = "SS"
        aertyp[2] = "BC"
        aertyp[3] = "OC"
        aertyp[4] = "SO"
        #dt = h5py.special_dtype(vlen=str)
        da1 = fout.create_dataset('/Aerosol/TypeNames',data=aertyp)
        #da1[...] = aertyp

        arindout = numpy.zeros((5,),dtype=numpy.int16) - 1
        for index, row in metfrm.iterrows():
            metnm = row['variable']
            if ('Meteorology' in metnm):
                metnm = metnm.replace('Meteorology','ECMWF')
                metnm = metnm.replace('_met','_ecmwf')
            print(metnm)
            if row['dims'] == 3:
                metdt = fmt[row['variable']][ref_idx[0],ref_idx[1],:]
                dfl13 = fout.create_dataset(metnm,data=metdt)
            elif row['dims'] == 2:
                metdt = fmt[row['variable']][ref_idx[0],ref_idx[1]]
                metout = numpy.tile(metdt,(1,)) 
                dfl13 = fout.create_dataset(metnm,data=metout)
            elif row['dims'] == 4:
                metdt = fmt[row['variable']][ref_idx[0],ref_idx[1],:,:]
                dfl13 = fout.create_dataset(metnm,data=metdt)

            # Set aerosol types
            if metnm == '/Aerosol/composite_aod_sort_index_met':
                for a1 in range(2):
                    arindout[metdt[a1]] = a1
                dfar1 = fout.create_dataset('/Aerosol/TypesUsed',data=arindout)
        fmt.close()

        dt = h5py.special_dtype(vlen=str)
        da2 = fout.create_dataset('/Metadata/AcquisitionMode',data=amdstr[0:1])
        dfl4 = fout.create_dataset('/SoundingGeometry/sounding_id',data=sdg_ids)

        # State Vector
        nstate = prior_mean.shape[0]
        state_info['state_seq'] = numpy.arange(nstate)
        sbsq = state_info[(state_info['fpsim_position'] >= 0) & (pandas.notnull(state_info['scene_group'])) ]
        smsqfp = sbsq['state_seq'].tolist()
        vcout = numpy.transpose(state_array[:,smsqfp])
        nstscn = vcout.shape[0]

        # Rescale
        for p in range(nstscn):
            vcout[p,:] = vcout[p,:] / sbsq['sv_scale'].values[p]
#            if (sbsq['pow_trans'].values[p] == 0):
#                vcout[p,:] = numpy.exp(vcout[p,:])
        dfscn = fout.create_dataset('/StateVector/sampled_state_vectors',data=vcout)

        tmpvr = state_names[smsqfp]
        grndnm = '/Ground/Albedo' 
        for p in range(nstscn):
            if ('BRDF' in str(tmpvr[p],'utf-8')):
                if (pandas.notnull(sbsq['alt_source'].values[p])):
                    tmpvr[p] = sbsq['surr_name'].values[p]
                else:
                    grndnm = '/Ground/Brdf' 
            elif ( ('CoxMunk' in str(tmpvr[p],'utf-8')) and ('Scaling' in str(tmpvr[p],'utf-8')) ):
                if (pandas.notnull(sbsq['alt_source'].values[p])):
                    tmpvr[p] = sbsq['surr_name'].values[p]
                else:
                    grndnm = '/Ground/Coxmunk_Scaled' 
        dfscn = fout.create_dataset('/StateVector/state_vector_names',data=tmpvr)

        if (mxcmp is not None):
            dfscn = fout.create_dataset('/StateVector/mixture_component',data=mxcmp)

        # Prior
        prstfl = '/groups/algorithm/l2_fp/builds/git_master/build/input/oco/input/l2_oco_static_input.h5'
        if sfctyp == 'land':
            prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                      '/Aerosol/ST/Gaussian/Log',grndnm,'/Instrument/Dispersion','/Fluorescence']
            prcmps =   [20, 1, 1, 1, 6, 3, 3, 3, 6, 3, 2]
            prngrp = [ 1, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1]
        elif sfctyp == 'ocean':
            prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                      '/Aerosol/ST/Gaussian/Log','/Ground/Windspeed', grndnm,'/Instrument/Dispersion']
            prcmps =   [20, 1, 1, 1, 6, 3, 3, 3, 1, 6, 3]
            prngrp = [ 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1]


        vcapr = prior_mean[smsqfp]
        aprfl = numpy.zeros((vcapr.shape[0],nreps),dtype=vcapr.dtype)
        for p in range(nstscn):
            aprfl[p,:] = vcapr[p]

        # Rescaling not needed
        dfscn = fout.create_dataset('/StateVector/a_priori',data=aprfl)

#        print(cvscl[16:20,16:20])

#        prunc = numpy.sqrt(numpy.diagonal(cvscl))
#        dfscn = fout.create_dataset('/StateVector/a_priori_uncert',data=prunc)

        # Assign by group
        sqgrpstr = sbsq['scene_group'].values
        nstpr = vcapr.shape[0]
        gsq = numpy.arange(nstpr)

        fcv = h5py.File(prstfl,'r')
        for q in range(len(prgrps)):
            print(prgrps[q])
            gsb = gsq[sqgrpstr == prgrps[q]]

            if prngrp[q] == 1:
                aprtmp = numpy.zeros((prcmps[q],nreps),dtype=aprfl.dtype)
                aprtmp[:,:] = aprfl[gsb,:]
                vrnmpr = '%s/a_priori' % (prgrps[q])
                if (aprtmp.shape[0] == 1):
                    aprtmp = aprtmp.flatten()
                dfscn = fout.create_dataset(vrnmpr,data=aprtmp)

                vrnpcv = '%s/covariance' % (prgrps[q])
                if (vrnpcv in fcv):
                    cvtmp = fcv[vrnpcv][:,:]
                    if prgrps[q] == '/Instrument/Dispersion':
                        cvout = numpy.zeros((3,3),dtype=numpy.float64)
                        for b0 in range(3):
                            cvout[b0,b0] = cvtmp[b0,0,0]
                    else:
                        cvout = cvtmp
                else:
                    cvout = numpy.zeros((prcmps[q],prcmps[q]),dtype=numpy.float64) + 1.0
                dfscn = fout.create_dataset(vrnpcv,data=cvout)
            else:
                dm0 = int(prngrp[q])
                dm1 = int(prcmps[q] / prngrp[q]) 
                if prgrps[q] == '/Ground/Brdf':
                    prgrd = fcv['/Ground/Brdf/a_priori'][:,:] 
                    aprtmp = numpy.zeros((dm0,prgrd.shape[1],nreps),dtype=aprfl.dtype)
                    nfx = prgrd.shape[1] - dm1
                    # Fixed coefficients are first, variable weights at end
                    for b0 in range(dm0):
                        for gd2 in range(nfx):
                            aprtmp[b0,gd2,:] = prgrd[b0,gd2]
                else:
                    aprtmp = numpy.zeros((dm0,dm1,nreps),dtype=aprfl.dtype)
                    nfx = 0
                ct2 = 0
                for b0 in range(dm0):
                    for b1 in range(dm1):
                        aprtmp[b0,b1+nfx,:] = aprfl[gsb[ct2],:]
                        ct2 = ct2 + 1
                vrnmpr = '%s/a_priori' % (prgrps[q])
                dfscn = fout.create_dataset(vrnmpr,data=aprtmp)

                # Covariance, fill in blocks
                #prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                #      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                #      '/Ground/Albedo','/Instrument/Dispersion','/Fluorescence']
                vrnpcv = '%s/covariance' % (prgrps[q])
                if (vrnpcv in fcv):
                    cvtmp = fcv[vrnpcv][:,:]
                    if prgrps[q] == '/Aerosol/Merra/Gaussian/Log':
                        cvout = numpy.tile(cvtmp,(2,1,1))
                    elif ( (prgrps[q] == '/Ground/Albedo') and (sfctyp == 'ocean') ):
                        vrocn = '/Ground/Coxmunk_Albedo_Quadratic/covariance' 
                        cvout = fcv[vrocn][:,0:2,0:2]
                    else:
                        cvout = cvtmp
                else:
                    cvout = numpy.zeros((prcmps[q],prcmps[q]),dtype=numpy.float64) + 1.0
                dfscn = fout.create_dataset(vrnpcv,data=cvout)
        fcv.close()               

        # Discrepancy 
        ndcmp = 30
        nchnl = 3048
        dscrp = numpy.zeros((nchnl,nreps),dtype=numpy.float32)
        if ( (moderr) or (moderr is None) ):
            scrtot = numpy.zeros((nchnl,nreps),dtype=numpy.float32)
            dcpfrm = pandas.read_csv(mdcfile, \
                                     dtype = {'component':str, 'file_name':str, \
                                              'variable_name':str, 'index':int}) 
            dscpfl = dcpfrm['file_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_mean']
            mnvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_sd']
            sdvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_basis']
            bsvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_basis_var']
            evvrnm = frmsb['variable_name'].values[0]
            
            print(dscpfl) 
            print(mnvrnm) 
            frd = h5py.File(dscpfl,'r')
            errmn = frd[mnvrnm][:]
            errsd = frd[sdvrnm][:]
            errbs = frd[bsvrnm][:,:]
            errev = frd[evvrnm][:]
            frd.close()
            print('Discrepancy parameters opened')

            chsq = numpy.arange(nchnl)
            efrm = pandas.DataFrame({'CmpSeq':chsq, 'EigenVal':errev})
            efrm = efrm.sort_values(by=['EigenVal'],ascending=[False])
            print(efrm[0:30])

            # Tile mean and std dev
            mnspc = numpy.transpose(numpy.tile(errmn,(nreps,1)))
            sdspc = numpy.transpose(numpy.tile(errsd,(nreps,1)))

            for d1 in range(ndcmp):
                cidx = efrm['CmpSeq'].values[d1]
                scrsd = numpy.sqrt(errev[cidx])
                scrcmp = random.normal(scale=scrsd,size=nreps)

                scrscz = numpy.outer(errbs[:,d1],scrcmp)
                scrtot = scrtot + scrscz

            print(scrscz.shape)
            dscrp = mnspc + sdspc * scrtot

        dfscn = fout.create_dataset('/SpectralParameters/sampled_radiance_offset',data=dscrp)

        fout.close()

    else:
        msg1 = 'Config not found'
        print(msg1)

    return

def setup_uq_expt_scene_ref_fixdel(outfile,scene_config,state_info,state_array,prior_mean,state_names, \
                                   sdg_ids,nreps,l1bfrm, metfrm, ref_idx,ref_l1b,ref_met, \
                                   modscl=1.0, sdvl = 255115, mxcmp=None):
    '''Generate a UQ experiment scene file given state vectors and 
       reference sounding information
       This version injects fixed model discrepancy
         outfile:         Name of output scene file
         scene_config:    Name of config CSV file 
         state_info:      State scaling data frame 
         state_array:     Matrix of state vectors
         prior_mean:      Prior mean vector
         state_names:     State vector names
         sdg_ids:         Vector of sounding IDs
         nreps:           Number of state vector replicates (1st dimension of state_array)
         l1bfrm:          Data frame with L1B fields and properties
         metfrm:          Data frame with meteorology fields and properties
         ref_idx:         Sounding index in L1B/Met files
         ref_l1b:         Reference L1B file
         ref_met:         Reference Meteorology file
         modscl:          Scale factor for discrepancy mean 
         sdvl:            Random seed
         mxcmp:           Mixture component array
    '''

    tmpsd = sdvl
    if os.path.exists(scene_config):
        f = open(scene_config)
        csv_f = csv.reader(f)
        ctr = 0
        rlst = []
        for rw in csv_f:
            rlst.append(rw)
            if rw[0] == 'marginal_variable_file':
                mrgfile = rw[1]
            elif rw[0] == 'fp_state_vector_file':
                svcfile = rw[1]
            elif rw[0] == 'fp_discrepancy_file':
                mdcfile = rw[1]
            elif rw[0] == 'surface_type':
                sfctyp = rw[1]
            elif rw[0] == 'acquisition_mode':
                aqmd = rw[1]
            elif rw[0] == 'random_seed':
                tmpsd = int(rw[1])
            ctr = ctr + 1
        f.close()

        random.seed(tmpsd)

        if ( (aqmd == 'nadir') or (aqmd == 'Nadir') ):
            aqmdout = 'Sample Nadir'
        else:
            aqmdout = 'Sample Glint'
        amdstr = numpy.empty((1,),"|O")
        amdstr[0] = aqmdout

        # Met/L1b parameters from data frames 
        flflt = numpy.array([-9.999e6],dtype=numpy.float32)

        fout = h5py.File(outfile,'w')

        f = h5py.File(ref_l1b)
        for index, row in l1bfrm.iterrows():
            if ( (row['dims'] == 3) and (row['sdgspec'] == 'Yes')):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1],0]
                l1bout = numpy.tile(l1bdt,(1,)) 
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 2) and (row['sdgspec'] == 'Yes') ):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1]]
                l1bout = numpy.tile(l1bdt,(1,))
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 4) and (row['sdgspec'] == 'Yes') ):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1],0,:]
                l1bout = numpy.tile(l1bdt,(1,))
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 3) and (row['sdgspec'] == 'No') ):
                l1bdt = f[row['variable']][:,ref_idx[1],:]
                if (row['variable'] == '/InstrumentHeader/bad_sample_list'):
                    l1bdt[:,:] = 0
                dfl13 = fout.create_dataset(row['variable'],data=l1bdt)
            elif ((row['dims'] == 4) and (row['sdgspec'] == 'No') ):
                l1bdt = f[row['variable']][:,ref_idx[1],:,:]
                dfl13 = fout.create_dataset(row['variable'],data=l1bdt)
        f.close()

        # Met fields
        fmt = h5py.File(ref_met)
        for index, row in metfrm.iterrows():
            metnm = row['variable']
            metnm = metnm.replace('Meteorology','ECMWF')
            metnm = metnm.replace('_met','_ecmwf')
            print(metnm)
            if row['dims'] == 3:
                metdt = fmt[row['variable']][ref_idx[0],ref_idx[1],:]
                dfl13 = fout.create_dataset(metnm,data=metdt)
            elif row['dims'] == 2:
                metdt = fmt[row['variable']][ref_idx[0],ref_idx[1]]
                metout = numpy.tile(metdt,(1,)) 
                dfl13 = fout.create_dataset(metnm,data=metout)
        fmt.close()

        dt = h5py.special_dtype(vlen=str)
        da2 = fout.create_dataset('/Metadata/AcquisitionMode',(1,),dtype=dt)
        da2[...] = amdstr 
        dfl4 = fout.create_dataset('/SoundingGeometry/sounding_id',data=sdg_ids)

        # State Vector
        nstate = prior_mean.shape[0]
        state_info['state_seq'] = numpy.arange(nstate)
        sbsq = state_info[(state_info['fpsim_position'] >= 0) & (pandas.notnull(state_info['scene_group'])) ]
        smsqfp = sbsq['state_seq'].tolist()
        print(smsqfp)
        print(state_info['scene_group'].values[63])
        vcout = numpy.transpose(state_array[:,smsqfp])
        nstscn = vcout.shape[0]

        # Rescale
        for p in range(nstscn):
            vcout[p,:] = vcout[p,:] / sbsq['sv_scale'].values[p]
#            if (sbsq['pow_trans'].values[p] == 0):
#                vcout[p,:] = numpy.exp(vcout[p,:])
        dfscn = fout.create_dataset('/StateVector/sampled_state_vectors',data=vcout)

        tmpvr = state_names[smsqfp] 
        for p in range(nstscn):
            if ('BRDF' in tmpvr[p]): 
                tmpvr[p] = sbsq['surr_name'].values[p] 
        dfscn = fout.create_dataset('/StateVector/state_vector_names',data=tmpvr)

        if (mxcmp is not None):
            dfscn = fout.create_dataset('/StateVector/mixture_component',data=mxcmp)

        # Prior
        prstfl = '/groups/algorithm/l2_fp/builds/git_master/build/input/oco/input/l2_oco_static_input.h5'
        if sfctyp == 'land':
            prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                      '/Ground/Albedo','/Instrument/Dispersion','/Fluorescence']
            prcmps =   [20, 1, 1, 1, 6, 3, 3, 6, 3, 2]
            prngrp = [ 1, 1, 1, 1, 2, 1, 1, 3, 1, 1]
        elif sfctyp == 'ocean':
            prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                      '/Ground/Windspeed', '/Ground/Albedo','/Instrument/Dispersion']
            prcmps =   [20, 1, 1, 1, 6, 3, 3, 1, 6, 3]
            prngrp = [ 1, 1, 1, 1, 2, 1, 1, 1, 3, 1]


        vcapr = prior_mean[smsqfp]
        aprfl = numpy.zeros((vcapr.shape[0],nreps),dtype=vcapr.dtype)
        for p in range(nstscn):
            aprfl[p,:] = vcapr[p]

        # Rescaling not needed
        dfscn = fout.create_dataset('/StateVector/a_priori',data=aprfl)

#        print(cvscl[16:20,16:20])

#        prunc = numpy.sqrt(numpy.diagonal(cvscl))
#        dfscn = fout.create_dataset('/StateVector/a_priori_uncert',data=prunc)

        # Assign by group
        sqgrpstr = sbsq['scene_group'].values
        nstpr = vcapr.shape[0]
        gsq = numpy.arange(nstpr)

        fcv = h5py.File(prstfl,'r')
        for q in range(len(prgrps)):
            print(prgrps[q])
            gsb = gsq[sqgrpstr == prgrps[q]]

            if prngrp[q] == 1:
                aprtmp = numpy.zeros((prcmps[q],nreps),dtype=aprfl.dtype)
                aprtmp[:,:] = aprfl[gsb,:]
                vrnmpr = '%s/a_priori' % (prgrps[q])
                if (aprtmp.shape[0] == 1):
                    aprtmp = aprtmp.flatten()
                dfscn = fout.create_dataset(vrnmpr,data=aprtmp)

                vrnpcv = '%s/covariance' % (prgrps[q])
                if (vrnpcv in fcv):
                    cvtmp = fcv[vrnpcv][:,:]
                    if prgrps[q] == '/Instrument/Dispersion':
                        cvout = numpy.zeros((3,3),dtype=numpy.float64)
                        for b0 in range(3):
                            cvout[b0,b0] = cvtmp[b0,0,0]
                    else:
                        cvout = cvtmp
                else:
                    cvout = numpy.zeros((prcmps[q],prcmps[q]),dtype=numpy.float64) + 1.0
                dfscn = fout.create_dataset(vrnpcv,data=cvout)
            else:
                dm0 = prngrp[q]
                dm1 = prcmps[q] / prngrp[q] 
                aprtmp = numpy.zeros((dm0,dm1,nreps),dtype=aprfl.dtype)
                ct2 = 0
                for b0 in range(dm0):
                    for b1 in range(dm1):
                        aprtmp[b0,b1,:] = aprfl[gsb[ct2],:]
                        ct2 = ct2 + 1
                vrnmpr = '%s/a_priori' % (prgrps[q])
                dfscn = fout.create_dataset(vrnmpr,data=aprtmp)

                # Covariance, fill in blocks
                #prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                #      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                #      '/Ground/Albedo','/Instrument/Dispersion','/Fluorescence']
                vrnpcv = '%s/covariance' % (prgrps[q])
                if (vrnpcv in fcv):
                    cvtmp = fcv[vrnpcv][:,:]
                    if prgrps[q] == '/Aerosol/Merra/Gaussian/Log':
                        cvout = numpy.tile(cvtmp,(2,1,1))
                    else:
                        cvout = cvtmp
                else:
                    cvout = numpy.zeros((prcmps[q],prcmps[q]),dtype=numpy.float64) + 1.0
                dfscn = fout.create_dataset(vrnpcv,data=cvout)
        fcv.close()               

        # Discrepancy 
        ndcmp = 30
        nchnl = 3048
        dscrp = numpy.zeros((nchnl,nreps),dtype=numpy.float32)
        scrtot = numpy.zeros((nchnl,nreps),dtype=numpy.float32)
        dcpfrm = pandas.read_csv(mdcfile, \
                                     dtype = {'component':str, 'file_name':str, \
                                              'variable_name':str, 'index':int}) 
        dscpfl = dcpfrm['file_name'].values[0]
        frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_mean']
        mnvrnm = frmsb['variable_name'].values[0]
        frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_sd']
        sdvrnm = frmsb['variable_name'].values[0]
            
        frd = h5py.File(dscpfl,'r')
        errmn = frd[mnvrnm][:]
        errsd = frd[sdvrnm][:]
        frd.close()

        sd01 = numpy.zeros(errsd.shape,dtype=errsd.dtype)
        sd01[errsd > 0.0] = 1.0

        mnspc = numpy.transpose(numpy.tile(modscl*errmn*sd01,(nreps,1)))
        dscrp = mnspc 

        dfscn = fout.create_dataset('/SpectralParameters/sampled_radiance_offset',data=dscrp)

        fout.close()

    else:
        msg1 = 'Config not found'
        print(msg1)

    return

def setup_uq_expt_scene_ref_bc(outfile,scene_config,state_info,state_array,prior_mean,state_names, \
                               sdg_ids,nreps,l1bfrm, metfrm, ref_idx,ref_l1b,ref_met,err_scl, \
                               moderr=None, fpidx=None, nrpdscrp=None, sdvl = 255115, mxcmp=None):
    '''Generate a UQ experiment scene file given state vectors and 
       reference sounding information, 
       discrepancy based on bias correction
         outfile:         Name of output scene file
         scene_config:    Name of config CSV file 
         state_info:      State scaling data frame 
         state_array:     Matrix of state vectors
         prior_mean:      Prior mean vector
         state_names:     State vector names
         sdg_ids:         Vector of sounding IDs
         nreps:           Number of state vector replicates (1st dimension of state_array)
         l1bfrm:          Data frame with L1B fields and properties
         metfrm:          Data frame with meteorology fields and properties
         ref_idx:         Sounding index in L1B/Met files
         ref_l1b:         Reference L1B file
         ref_met:         Reference Meteorology file
         err_scl:         Array of scale factors for discrepancy (e.g. channel noise)
         moderr:          Logical for sampling model discrepancy
         fpidx:           Optional footprint index (0-based, footprint number minus 1)
         nrrpdscrp:       Number of replicates for discrepancy, defaults to nrep
         sdvl:            Random seed
         mxcmp:           Mixture component array
    '''

    tmpsd = sdvl
    if os.path.exists(scene_config):
        f = open(scene_config)
        csv_f = csv.reader(f)
        ctr = 0
        rlst = []
        for rw in csv_f:
            rlst.append(rw)
            if rw[0] == 'fp_discrepancy_file':
                mdcfile = rw[1]
            elif rw[0] == 'surface_type':
                sfctyp = rw[1]
            elif rw[0] == 'acquisition_mode':
                aqmd = rw[1]
            elif rw[0] == 'random_seed':
                tmpsd = int(rw[1])
            ctr = ctr + 1
        f.close()

        random.seed(tmpsd)

        amdstr = state_names[0:2].copy()
        if ( (aqmd == 'nadir') or (aqmd == 'Nadir') ):
            amdstr[0] = 'Sample Nadir'
        else:
            amdstr[0] = 'Sample Glint'

        # Met/L1b parameters from data frames 
        flflt = numpy.array([-9.999e6],dtype=numpy.float32)

        fout = h5py.File(outfile,'w')

        f = h5py.File(ref_l1b)
        for index, row in l1bfrm.iterrows():
            if ( (row['dims'] == 3) and (row['sdgspec'] == 'Yes')):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1],0]
                l1bout = numpy.tile(l1bdt,(1,)) 
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 2) and (row['sdgspec'] == 'Yes') ):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1]]
                l1bout = numpy.tile(l1bdt,(1,))
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 4) and (row['sdgspec'] == 'Yes') ):
                l1bdt = f[row['variable']][ref_idx[0],ref_idx[1],0,:]
                l1bout = numpy.tile(l1bdt,(1,))
                dfl13 = fout.create_dataset(row['variable'],data=l1bout)
            elif ((row['dims'] == 3) and (row['sdgspec'] == 'No') ):
                l1bdt = f[row['variable']][:,ref_idx[1],:]
                if (row['variable'] == '/InstrumentHeader/bad_sample_list'):
                    l1bdt[:,:] = 0
                dfl13 = fout.create_dataset(row['variable'],data=l1bdt)
            elif ((row['dims'] == 4) and (row['sdgspec'] == 'No') ):
                l1bdt = f[row['variable']][:,ref_idx[1],:,:]
                dfl13 = fout.create_dataset(row['variable'],data=l1bdt)
        f.close()

        # Met fields
        fmt = h5py.File(ref_met)
        # Get aerosol types
        #aertyp = fmt['/Metadata/CompositeAerosolTypes'][:]
        aertyp = state_names[0:5].copy() 
        aertyp[0] = "DU"
        aertyp[1] = "SS"
        aertyp[2] = "BC"
        aertyp[3] = "OC"
        aertyp[4] = "SO"
        #dt = h5py.special_dtype(vlen=str)
        da1 = fout.create_dataset('/Aerosol/TypeNames',data=aertyp)
        #da1[...] = aertyp

        arindout = numpy.zeros((5,),dtype=numpy.int16) - 1
        for index, row in metfrm.iterrows():
            metnm = row['variable']
            if ('Meteorology' in metnm):
                metnm = metnm.replace('Meteorology','ECMWF')
                metnm = metnm.replace('_met','_ecmwf')
            print(metnm)
            if row['dims'] == 3:
                metdt = fmt[row['variable']][ref_idx[0],ref_idx[1],:]
                dfl13 = fout.create_dataset(metnm,data=metdt)
            elif row['dims'] == 2:
                metdt = fmt[row['variable']][ref_idx[0],ref_idx[1]]
                metout = numpy.tile(metdt,(1,)) 
                dfl13 = fout.create_dataset(metnm,data=metout)
            elif row['dims'] == 4:
                metdt = fmt[row['variable']][ref_idx[0],ref_idx[1],:,:]
                dfl13 = fout.create_dataset(metnm,data=metdt)

            # Set aerosol types
            if metnm == '/Aerosol/composite_aod_sort_index_met':
                for a1 in range(2):
                    arindout[metdt[a1]] = a1
                dfar1 = fout.create_dataset('/Aerosol/TypesUsed',data=arindout)
        fmt.close()

        dt = h5py.special_dtype(vlen=str)
        da2 = fout.create_dataset('/Metadata/AcquisitionMode',data=amdstr[0:1])
        dfl4 = fout.create_dataset('/SoundingGeometry/sounding_id',data=sdg_ids)

        # State Vector
        nstate = prior_mean.shape[0]
        state_info['state_seq'] = numpy.arange(nstate)
        sbsq = state_info[(state_info['fpsim_position'] >= 0) & (pandas.notnull(state_info['scene_group'])) ]
        smsqfp = sbsq['state_seq'].tolist()
        vcout = numpy.transpose(state_array[:,smsqfp])
        nstscn = vcout.shape[0]

        # Rescale
        for p in range(nstscn):
            vcout[p,:] = vcout[p,:] / sbsq['sv_scale'].values[p]
#            if (sbsq['pow_trans'].values[p] == 0):
#                vcout[p,:] = numpy.exp(vcout[p,:])
        dfscn = fout.create_dataset('/StateVector/sampled_state_vectors',data=vcout)

        tmpvr = state_names[smsqfp] 
        grndnm = '/Ground/Albedo' 
        for p in range(nstscn):
            if ('BRDF' in str(tmpvr[p],'utf-8')):
                if (pandas.notnull(sbsq['alt_source'].values[p])):
                    tmpvr[p] = sbsq['surr_name'].values[p]
                else:
                    grndnm = '/Ground/Brdf'
            elif ( ('CoxMunk' in str(tmpvr[p],'utf-8')) and ('Scaling' in str(tmpvr[p],'utf-8'))  ): 
                if (pandas.notnull(sbsq['alt_source'].values[p])):
                    tmpvr[p] = sbsq['surr_name'].values[p]
                else:
                    grndnm = '/Ground/Coxmunk_Scaled'
        if (mxcmp is not None):
            dfscn = fout.create_dataset('/StateVector/mixture_component',data=mxcmp)
        dfscn = fout.create_dataset('/StateVector/state_vector_names',data=tmpvr)

        # Prior
        prstfl = '/home/jhobbs/L2FPConfig/l2_oco_static_input.h5'
        if sfctyp == 'land':
            prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                      '/Aerosol/ST/Gaussian/Log',grndnm,'/Instrument/Dispersion','/Fluorescence']
            prcmps =   [20, 1, 1, 1, 6, 3, 3, 3, 6, 3, 2]
            prngrp = [ 1, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1]
        elif sfctyp == 'ocean':
            prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                      '/Aerosol/ST/Gaussian/Log','/Ground/Windspeed', grndnm,'/Instrument/Dispersion']
            prcmps =   [20, 1, 1, 1, 6, 3, 3, 3, 1, 6, 3]
            prngrp = [ 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1]


        vcapr = prior_mean[smsqfp]
        aprfl = numpy.zeros((vcapr.shape[0],nreps),dtype=vcapr.dtype)
        for p in range(nstscn):
            aprfl[p,:] = vcapr[p]

        # Rescaling not needed
        dfscn = fout.create_dataset('/StateVector/a_priori',data=aprfl)

        # Assign by group
        sqgrpstr = sbsq['scene_group'].values
        nstpr = vcapr.shape[0]
        gsq = numpy.arange(nstpr)

        fcv = h5py.File(prstfl,'r')
        for q in range(len(prgrps)):
            print(prgrps[q])
            gsb = gsq[sqgrpstr == prgrps[q]]

            if prngrp[q] == 1:
                aprtmp = numpy.zeros((prcmps[q],nreps),dtype=aprfl.dtype)
                aprtmp[:,:] = aprfl[gsb,:]
                vrnmpr = '%s/a_priori' % (prgrps[q])
                if (aprtmp.shape[0] == 1):
                    aprtmp = aprtmp.flatten()
                dfscn = fout.create_dataset(vrnmpr,data=aprtmp)

                vrnpcv = '%s/covariance' % (prgrps[q])
                if (vrnpcv in fcv):
                    cvtmp = fcv[vrnpcv][:,:]
                    if prgrps[q] == '/Instrument/Dispersion':
                        cvout = numpy.zeros((3,3),dtype=numpy.float64)
                        for b0 in range(3):
                            cvout[b0,b0] = cvtmp[b0,0,0]
                    elif prgrps[q] == '/Ground/Windspeed':
                        # Version 10 prior cov
                        cvout = numpy.zeros((1,1),dtype=numpy.float64) + 40.0
                    else:
                        cvout = cvtmp
                elif prgrps[q] == '/Aerosol/ST/Gaussian/Log':
                     # From oco_base_config.lua
                     cvout = numpy.zeros((3,3),dtype=numpy.float64)
                     cvout[0,0] = 3.24
                     cvout[1,1] = 1.0e-8
                     cvout[2,2] = 1.0e-4
                else:
                    cvout = numpy.zeros((prcmps[q],prcmps[q]),dtype=numpy.float64) + 1.0
                dfscn = fout.create_dataset(vrnpcv,data=cvout)
            else:
                dm0 = prngrp[q]
                dm1 = int(prcmps[q] / prngrp[q]) 
                if prgrps[q] == '/Ground/Brdf':
                    prgrd = fcv['/Ground/Brdf/a_priori'][:,:] 
                    aprtmp = numpy.zeros((dm0,prgrd.shape[1],nreps),dtype=aprfl.dtype)
                    nfx = prgrd.shape[1] - dm1
                    # Fixed coefficients are first, variable weights at end
                    for b0 in range(dm0):
                        for gd2 in range(nfx):
                            aprtmp[b0,gd2,:] = prgrd[b0,gd2]
                else:
                    aprtmp = numpy.zeros((dm0,dm1,nreps),dtype=aprfl.dtype)
                    nfx = 0
                ct2 = 0
                for b0 in range(dm0):
                    for b1 in range(dm1):
                        aprtmp[b0,b1+nfx,:] = aprfl[gsb[ct2],:]
                        ct2 = ct2 + 1
                vrnmpr = '%s/a_priori' % (prgrps[q])
                dfscn = fout.create_dataset(vrnmpr,data=aprtmp)

                # Covariance, fill in blocks
                #prgrps = ['/Gas/CO2','/Gas/H2O_Scaling_factor','/Surface_Pressure','/Temperature/Offset', \
                #      '/Aerosol/Merra/Gaussian/Log','/Aerosol/Ice/Gaussian/Log','/Aerosol/Water/Gaussian/Log', \
                #      '/Ground/Albedo','/Instrument/Dispersion','/Fluorescence']
                vrnpcv = '%s/covariance' % (prgrps[q])
                if (vrnpcv in fcv):
                    cvtmp = fcv[vrnpcv][:,:]
                    if prgrps[q] == '/Aerosol/Merra/Gaussian/Log':
                        cvout = numpy.tile(cvtmp,(2,1,1))
                    elif ( (prgrps[q] == '/Ground/Albedo') and (sfctyp == 'ocean') ):
                        vrocn = '/Ground/Coxmunk_Albedo_Quadratic/covariance' 
                        cvout = fcv[vrocn][:,0:2,0:2]
                        #cvout[:,1,1] = 1.0 
                    else:
                        cvout = cvtmp
                else:
                    cvout = numpy.zeros((prcmps[q],prcmps[q]),dtype=numpy.float64) + 1.0
                dfscn = fout.create_dataset(vrnpcv,data=cvout)
        fcv.close()               

        # Discrepancy 
        nchnl = 3048
        dscrp = numpy.zeros((nchnl,nreps),dtype=numpy.float32)
        if ( (moderr) or (moderr is None) ):
            scrtot = numpy.zeros((nchnl,nreps),dtype=numpy.float32)
            dcpfrm = pandas.read_csv(mdcfile, \
                                     dtype = {'component':str, 'file_name':str, \
                                              'variable_name':str, 'index':int}) 
            dscpfl = dcpfrm['file_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_mean']
            mnvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_sd']
            sdvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_basis']
            bsvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_basis_var']
            evvrnm = frmsb['variable_name'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_start_index']
            stidx = frmsb['index'].values[0]
            frmsb = dcpfrm.loc[dcpfrm['component'] == 'discrepancy_end_index']
            fnidx = frmsb['index'].values[0]
            
            if fpidx is None: 
                frd = h5py.File(dscpfl,'r')
                errmn = frd[mnvrnm][:]
                errsd = frd[sdvrnm][:]
                errbs = frd[bsvrnm][stidx:fnidx,:]
                errev = frd[evvrnm][stidx:fnidx]
                msgarr = frd[mnvrnm].attrs['missing_value'][:]
                frd.close()
            elif fpidx >= 0:
                frd = h5py.File(dscpfl,'r')
                errmn = frd[mnvrnm][fpidx,:]
                errsd = frd[sdvrnm][fpidx,:]
                errbs = frd[bsvrnm][fpidx,stidx:fnidx,:]
                errev = frd[evvrnm][fpidx,stidx:fnidx]
                msgarr = frd[mnvrnm].attrs['missing_value'][:]
                frd.close()

            errmn[errmn == msgarr[0]] = 0.0
            

            # Tile mean and scale factor
            mnspc = numpy.transpose(numpy.tile(errmn,(nreps,1)))
            sdspc = numpy.transpose(numpy.tile(err_scl,(nreps,1)))

            ndcmp = fnidx-stidx
            for d1 in range(ndcmp):
                scrsd = numpy.sqrt(errev[d1])
                scrtmp = random.normal(scale=scrsd,size=nrpdscrp)
                ntile = nreps / nrpdscrp
                scrcmp = numpy.repeat(scrtmp, ntile)

                bstmp = errbs[d1,:]
                bstmp[bstmp == msgarr[0]] = 0.0
                scrscz = numpy.outer(bstmp,scrcmp)
                scrtot = scrtot + scrscz

            print(scrscz.shape)
            dscrp = sdspc * (mnspc + scrtot)

        dfscn = fout.create_dataset('/SpectralParameters/sampled_radiance_offset',data=dscrp)

        fout.close()

    else:
        msg1 = 'Config not found'
        print(msg1)

    return

