### examples

Example 1: Forward Model Perturbations

The script `fmexamp_perturb.py` can be run with multiple perturbations to the state vector and viewing geometry, using the `case_id` variable.
For each case, the corresponding data file (`land_state_case_id.csv`) is read and the forward model is evaluated. The cases available are 

* `aer`: The aerosol optical depth (AOD) is perturbed
* `albedo`: The albedo in each spectral band is perturbed
* `co2`: The XCO2 is perturbed
* `psfc`: The surface pressure is perturbed
* `sza`: The solar zenith angle is perturbed
