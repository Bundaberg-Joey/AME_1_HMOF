#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:25:49 2020

@author: Hook
"""

import argparse

import BOGP
from AmiSimTools.DataTriage import DataTriageCSV
from AmiSimTools.SimScreen import SimulatedScreenerSerial, SimulatedScreenerParallel


data_location = r'../Scaled_HCOF_F2.csv'

sim_data = DataTriageCSV.load_from_path(data_location)

sim_pscreen=SimulatedScreenerParallel(sim_data,1,40,10,10,15)

ami = BOGP.prospector(X=sim_data.X, acquisition_function='Thompson')

sim_pscreen.perform_screening(ami)


