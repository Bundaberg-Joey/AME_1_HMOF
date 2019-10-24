# AME1
This is code for data preparation and AI part of the single test Bayesian Optimization version of the Autonomous Materials Explorer. Part of a collaboration between Maths and Chem Eng at the University of Bath.

BAYESIAN OPTIMIZATION:

BOGP.py

This file contains the prospector class which does all the ML and decision making. It uses a dense GP model from https://github.com/SheffieldML/GPy along with our own custom code for sparse inference, sampling and Bayesian optimization.

HMOF DATA IMPORT:

import_chem_and_ph_HMOF.py import_excell_data_HMOF.py and combine_excell_and_chem_ph_HMOF.py

These fies are used to import the HMOF data and prepare it for use in the screening. They need access to various data that is not hosted on the git or anywhere online at the moment.

HMOF DATA:

HMOFDATA.pkl 

contains all of the feature matrices and target values for HMOF screening

HMOF FEATURE COMPARRISON:

compare_features_HMOF.py RRMSE_HMOF.npy

These files are for testing the accuracy of predictions using different datasets in HMOFDATA.pky 

SIMULATED AME SCREENING:

simulated_screed_HMOF.py

This runs a single simulated AME screen and makes some nice plots on the way