#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 20:39:44 2019

combies

@author: Hook
"""
import pickle

with open('DATA_FROM_EXCELL_TABLE.pkl', 'rb') as f:
    EXCELL = pickle.load(f)
    
with open('CHEM_AND_PH_DATA.pkl', 'rb') as f:
    CPH = pickle.load(f)
#
#with open('DATA_FROM_EXCELL_TABLE.pkl', 'wb') as f:
#    pickle.dump({'APIs':APIs,'physical':physical,'adsoptions':adsoptions,'X_Physical':X_Physical,'Y_Adsorbtion':Y_Adsorbtion,'YAPI_True':YAPI_True,'YAPI_Fixed':YAPI_Fixed}, f)
#
#with open('CHEM_AND_PH_DATA.pkl', 'wb') as f:
#    pickle.dump({'ATOMS':ATOMS,'X_Chemical':ELEMENT_COUNTS_NEW_scaled,'X_Topological':PH_MATRIX_scaled_reduced}, f)
#

HMOFDATA={**EXCELL,**CPH}

with open('HMOFDATA.pkl', 'wb') as f:
    pickle.dump(HMOFDATA, f)
