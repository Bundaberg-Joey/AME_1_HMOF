# AMI_1
This is code for data preparation and AI part of the single test Bayesian Optimization version of the Autonomous Materials Explorer.
Part of a collaboration between Maths and Chem Eng at the University of Bath.

## BAYESIAN OPTIMIZATION `BOGP.py`:
This file contains the `prospector` class which does all the ML and decision making.
It uses a dense GP model from [the GPy modeule](https://github.com/SheffieldML/GPy) along with our own custom code for sparse inference, sampling and Bayesian optimization.

The prospector class is initialised with a feature matrix `X` and contains the below methods:
* `fit`
* `predict`
* `samples`
* `pick_next`

## Simulated Screenings
Rather than use full MonteCarlo simulations when developing the AMI, simuated screenings are performed where reference data sets are used with already known values.
This cuts down the simulation time from full simulation to a quick index search in numpy.

The classes used to simulate the screening are located in `AmiSimTools.py` and the script to perform the screening is `Simulated_AMI_Screen.py`.

### `Ami_Sim_Tools`
Contains classes for the loading of simulation data and the execution of thos simulated screenings with the AMI / other such model.

### `Simluated_AMI_Screen`
A Python script using the classes listed in the sim tools file.

## Data Preparation `Data_Prep`:
### HMOF DATA IMPORT: 
* `import_chem_and_ph_HMOF.py` 
* `import_excell_data_HMOF.py`
* `combine_excell_and_chem_ph_HMOF.py`

These fies are used to import the HMOF data and prepare it for use in the screening. They need access to various data that is not hosted on the git or anywhere online at the moment.

### HMOF DATA:
* `HMOFDATA.pkl` 

Contains all of the feature matrices and target values for HMOF screening

### HMOF FEATURE COMPARRISON:
* `compare_features_HMOF.py`
* `RRMSE_HMOF.npy`

These files are for testing the accuracy of predictions using different datasets in HMOFDATA.pky 
