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

This retrieval simulation experiment incorporates forward model discrepancy to introduce realistic imperfect knowledge into the observing system simulation. The OCO-2 reference sounding for this example occurs over Texas during August 2020. 

1. Supporting OCO-2 operational data products can be downloaded from the [GES DISC](https://disc.gsfc.nasa.gov/)
    - Accessing publicly available AIRS products from the [GES DISC](https://disc.gsfc.nasa.gov/) requires registration for a free NASA Earthdata acocunt.
    - Additional steps and general instructions for downloading/subsetting products can be found at the [data access page](https://disc.gsfc.nasa.gov/data-access)
    - The list of supporting data are in the repository as `/metadata/OCO2_V10_2020082319555502.txt` and can be downloaded with `wget` with the command

            wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies -i OCO2_V10_2020082319555502.txt  
    The products total about 350 MB in size.
