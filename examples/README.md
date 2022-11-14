### examples

Example 1: Forward Model Perturbations

The script `fmexamp_perturb.py` can be run with multiple perturbations to the state vector and viewing geometry, using the `case_id` variable.
For each case, the corresponding data file (`land_state_case_id.csv`) is read and the forward model is evaluated. The cases available are 

* `aer`: The aerosol optical depth (AOD) is perturbed
* `albedo`: The albedo in each spectral band is perturbed
* `co2`: The XCO2 is perturbed
* `psfc`: The surface pressure is perturbed
* `sza`: The solar zenith angle is perturbed

*** 

Example 2: Retrievals With Model Discrepancy

This retrieval simulation experiment incorporates forward model discrepancy to introduce realistic imperfect knowledge into the observing system simulation. The OCO-2 reference sounding for this example occurs over Texas during August 2020. The reference sounding ID is

```
ref_sounding_id = "2020082319555502"
```

1. Supporting OCO-2 operational data products can be downloaded from the [GES DISC](https://disc.gsfc.nasa.gov/)
    - Accessing publicly available OCO-2 products from the [GES DISC](https://disc.gsfc.nasa.gov/) requires registration for a free NASA Earthdata acocunt.
    - Additional steps and general instructions for downloading/subsetting products can be found at the [data access page](https://disc.gsfc.nasa.gov/data-access)
    - The list of supporting data are in the repository as `/metadata/OCO2_V10_2020082319555502.txt` and can be downloaded with `wget` with the command

            wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies -i OCO2_V10_2020082319555502.txt  
    The products total about 2.4 GB in size.  
    GES DISC data access is gradually being moved to the [Earthdata Cloud](https://disc.gsfc.nasa.gov/information/documents?title=Migrating%20to%20the%20Cloud) as of 2022-23, so data access capabilities will be subject to change.
2. The general experiment workflow in the repository [README](../README.md) can be followed. The `ref_sounding_id` can be changed in scripts as noted above.
    - In `setup_oco_scene_discrep.py`, the call to `setup_uq_l1b` should use the `discrep=True` option
3. Incorporate model discrepancy
    - At step 4 of the procedure (execute individual forward runs), use the `fwdrun_oco_save_sing_discrep.py` script
    - The discrepancy distribution parameters are found in the supporting file `lnd_nadir_202008_reg02_combined_radiance_parameters_bc_fp8.h5`
    - In `combine_l1b_oco.py`, the call to `uq_expt_aggregate_l1b` should use the `discrep=True` option
4. Retrievals and aggregation can be run in the same fashion as other experiments

An [example notebook](../visualization/refrac_discrep_summary.ipynb) with retrieval results is available in the `visualization` directory
