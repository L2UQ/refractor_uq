# refractor_uq
Uncertainty quantification software for use with ReFRACtor

This software adds capability for simulation-based uncertainty quantification (UQ) for remote sensing retrievals carried with [ReFRACtor](https://refractor.github.io/documentation/), the Reusable Framework for Retrieval of Atmospheric Composition.

Procedure

1. Extract prior mean and test forward model
    - Script: `test_config_prior_oco.py`
    - Output: `land_state_2015080820071304.csv`
2. Simulate state true state vectors
    - Script: `sim_oco_unimodal.py`
    - Output: `lnd_nadir_201508_refractor_state_vectors.h5`
    - Supporting files:
3. Setup scene files
    - Script: `setup_oco_scene.py`
    - Output: `lnd_nadir_refractor_expt_l1b_uqscene.h5`
4. Execute individual forward runs to produce radiances
    - Create a directory called `l1b` for radiances: `mkdir l1b`
    - Script: `fwdrun_oco_save_sing.py`, requires a command line argument (number) with the sounding index:  
    `python fwdrun_oco_save_sing.py ##`
    - Optional cluster job array: `run_oco_fwd.sh`
    - Output: `l1b/l1b_####.h5`
5. Combine L1B results back into scene file
    - Script: `combine_l1b_oco.py`
    - Output: Radiances saved to `lnd_nadir_refractor_expt_l1b_uqscene.h5`
6. Run retrievals
    - Create a directory called `output` for retrieval results: `mkdir output`
    - Script: `refractor_run_retrieval`, requires a command line argument (number) with the sounding index:  
    `python refractor_run_retrieval.py ##`
    - Output: `output/l2_####.h5`
7. Aggregate retrievals 

