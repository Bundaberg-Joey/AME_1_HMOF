#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 09:55:26 2019

IMPORTS ALL DATA FROM EXCELL FILES

PHYSICAL FEATURES
ADSORPTIONS
APIs
FIXED APIs

@author: Hook
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

table_with_sep = pd.read_csv("table_cCGA_nCGA_mCGA_selC4ads_saV_saG_mP_dP_vF.csv") 

physical_1=['Volumetric Surface Area (m^2/cm^3)',
       'Gravimetric Surface Area (m^2/g)', 'Max Pore Diameter (Angs)',
       'Dom Pore Diameter (Angs)', 'Void Fraction']

adsoptions=['CO2 adsorbed at 0.01 bar (cm^3 at STP/g)',
       'CO2 adsorbed at 0.05 bar (cm^3 at STP/g)',
       'CO2 adsorbed at 0.1 bar (cm^3 at STP/g)',
       'CO2 adsorbed at 0.5 bar (cm^3 at STP/g)',
       'CO2 adsorbed at 2.5 bar (cm^3 at STP/g)',
       'N2 adsorbed at 0.09 bar (cm^3 at STP/g)',
       'N2 adsorbed at 0.9 bar (cm^3 at STP/g)',
       'CH4 adsorbed at 0.05 bar (cm^3 at STP/g)',
       'CH4 adsorbed at 0.5 bar (cm^3 at STP/g)',
       'CH4 adsorbed at 0.9 bar (cm^3 at STP/g)',
       'CH4 adsorbed at 2.5 bar (cm^3 at STP/g)',
       'CH4 adsorbed at 4.5 bar (cm^3 at STP/g)']

plt.plot(table_with_sep['CO2 adsorbed at 0.01 bar (cm^3 at STP/g)'],table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)'],'.')
plt.plot([0,200],[0,200])
plt.show()

co2_dc=table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']-table_with_sep['CO2 adsorbed at 0.01 bar (cm^3 at STP/g)']
co2_dc[co2_dc<=0]=0

plt.plot(table_with_sep['N2 adsorbed at 0.9 bar (cm^3 at STP/g)'],table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)'],'.')
plt.show()

numerator=np.array(np.multiply(co2_dc,table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']))
denominator=np.array(table_with_sep['N2 adsorbed at 0.9 bar (cm^3 at STP/g)'])
plt.loglog(denominator,numerator,'.')
plt.xlabel('N2 contribution')
plt.ylabel('CO2 contribution')
plt.show()

API_true=numerator/denominator
top=np.argsort(API_true)[-100:]
plt.loglog(denominator,numerator,'.')
plt.loglog(denominator[top],numerator[top],'.')
plt.xlabel('N2 contribution')
plt.ylabel('CO2 contribution')
plt.show()

API=np.log(1+numerator/(denominator+1))
top=np.argsort(API)[-100:]
plt.loglog(denominator,numerator,'.')
plt.loglog(denominator[top],numerator[top],'.')
plt.xlabel('N2 contribution')
plt.ylabel('CO2 contribution')
plt.show()

top=np.argsort(API)[-100:]
plt.scatter(np.log10(denominator),np.log10(numerator),2,API)
plt.plot(np.log10(denominator[top]),np.log10(numerator[top]),'.')
plt.xlabel('log N2 contribution')
plt.ylabel('log CO2 contribution')
plt.show()

APIs=['Selectivity, ads',
       'DC, CO2', 'API', 'CO2 uptake, ads', 'Regenerability',
       'Selectivity, des', 'DC, N2', 'sorbent selection parameter']

table_with_sep['Selectivity, ads']=9*table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']/table_with_sep['N2 adsorbed at 0.9 bar (cm^3 at STP/g)']
table_with_sep['DC, CO2']=table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']-table_with_sep['CO2 adsorbed at 0.01 bar (cm^3 at STP/g)']
table_with_sep['API']=table_with_sep['Selectivity, ads']*table_with_sep['DC, CO2']
table_with_sep['CO2 uptake, ads']=table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']
table_with_sep['Regenerability']=table_with_sep['DC, CO2']/table_with_sep['CO2 uptake, ads']
table_with_sep['Selectivity, des']=9*table_with_sep['CO2 adsorbed at 0.05 bar (cm^3 at STP/g)']/table_with_sep['N2 adsorbed at 0.09 bar (cm^3 at STP/g)']
table_with_sep['DC, N2']=table_with_sep['N2 adsorbed at 0.9 bar (cm^3 at STP/g)']-table_with_sep['N2 adsorbed at 0.09 bar (cm^3 at STP/g)']
table_with_sep['sorbent selection parameter']=(table_with_sep['Selectivity, ads']**2*table_with_sep['DC, CO2'])/(table_with_sep['DC, N2']*table_with_sep['Selectivity, des'])

HMOF_table = pd.read_csv("hMOF_allData_March25_2013.csv") 

""" quick double check that ordering of MOFs is the same between tables """
plt.plot(HMOF_table['Void Fraction'],table_with_sep['Void Fraction'],'.')
plt.show()

physical_2=['Dom. Pore (ang.)', 'Max. Pore (ang.)',
       'Void Fraction', 'Surf. Area (m2/g)',
       'Vol. Surf. Area', 'Density']

physical=[physical_1]+['Density']

X_Physical=np.hstack((np.array(table_with_sep[physical_1]),np.array(HMOF_table['Density']).reshape(-1,1)))
Y_Adsorbtion=np.array(table_with_sep[adsoptions])
YAPI_True=np.array(table_with_sep[APIs])

""" now get fixed APIs by adding confidence bound onto all adsoptions """
""" plot histograms of negative DC values to get an idea of error magnitude """

plt.hist(table_with_sep['DC, CO2'][table_with_sep['DC, CO2']<0])
plt.show()
plt.hist(table_with_sep['DC, N2'][table_with_sep['DC, N2']<0])
plt.show()

CO2eps=1
N2eps=1

""" confidence bound on denominator """
table_with_sep['Selectivity, ads']=9*table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']/(table_with_sep['N2 adsorbed at 0.9 bar (cm^3 at STP/g)']+N2eps)
table_with_sep['DC, CO2']=table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']-table_with_sep['CO2 adsorbed at 0.01 bar (cm^3 at STP/g)']
""" don't allow negative DC """
table_with_sep['DC, CO2'][table_with_sep['DC, CO2']<CO2eps]=CO2eps
table_with_sep['API']=table_with_sep['Selectivity, ads']*table_with_sep['DC, CO2']
table_with_sep['CO2 uptake, ads']=table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']
table_with_sep['Regenerability']=table_with_sep['DC, CO2']/(table_with_sep['CO2 uptake, ads']+CO2eps)
table_with_sep['Selectivity, des']=9*table_with_sep['CO2 adsorbed at 0.05 bar (cm^3 at STP/g)']/(table_with_sep['N2 adsorbed at 0.09 bar (cm^3 at STP/g)']+N2eps)
table_with_sep['DC, N2']=table_with_sep['N2 adsorbed at 0.9 bar (cm^3 at STP/g)']-table_with_sep['N2 adsorbed at 0.09 bar (cm^3 at STP/g)']
table_with_sep['DC, N2'][table_with_sep['DC, N2']<N2eps]=N2eps
table_with_sep['sorbent selection parameter']=table_with_sep['Selectivity, ads']**2*table_with_sep['DC, CO2']*table_with_sep['N2 adsorbed at 0.09 bar (cm^3 at STP/g)']/(table_with_sep['DC, N2']*9*(table_with_sep['CO2 adsorbed at 0.05 bar (cm^3 at STP/g)']+CO2eps))

log_cols=['Selectivity, ads','API','Regenerability','Selectivity, des','sorbent selection parameter']
table_with_sep[log_cols]=np.log(1+table_with_sep[log_cols])
YAPI_Fixed=np.array(table_with_sep[APIs])

APInumerator=table_with_sep['DC, CO2']*9*table_with_sep['CO2 adsorbed at 0.1 bar (cm^3 at STP/g)']
APIdenominator=table_with_sep['N2 adsorbed at 0.9 bar (cm^3 at STP/g)']

with open('DATA_FROM_EXCELL_TABLE.pkl', 'wb') as f:
    pickle.dump({'APInumerator':APInumerator,'APIdenominator':APIdenominator,'APIs':APIs,'physical':physical,'adsoptions':adsoptions,'X_Physical':X_Physical,'Y_Adsorbtion':Y_Adsorbtion,'YAPI_True':YAPI_True,'YAPI_Fixed':YAPI_Fixed}, f)




#
#ELEMENT_COUNT_PER_VOLUME_FOR_TABLE=np.load('ELEMENT_COUNT_PER_VOLUME_FOR_TABLE.npy')
#
#summed=np.sum(X_Physical,1)+np.sum(ELEMENT_COUNT_PER_VOLUME_FOR_TABLE,1)+np.sum(Y_Adsorbtion,1)
##good_MOFs=[i for i in range(X_Physical.shape[0]) if np.isnan(summed[i])==False]
#good_MOFs=list(range(X_Physical.shape[0]))
#
#X_Physical=X_Physical[good_MOFs]
#Y_Adsorbtion=Y_Adsorbtion[good_MOFs]
#YAPI_True=YAPI_True[good_MOFs]
#YAPI_Fixed=YAPI_Fixed[good_MOFs]
#X_Chemical=ELEMENT_COUNT_PER_VOLUME_FOR_TABLE[good_MOFs]
#
#with open('DATA_FROM_EXCELL_TABLE.pkl', 'wb') as f:
#    pickle.dump({'APIs':APIs,'physical':physical,'adsoptions':adsoptions,'X_Physical':X_Physical,'Y_Adsorbtion':Y_Adsorbtion,'YAPI_True':YAPI_True,'YAPI_Fixed':YAPI_Fixed}, f)
##
##
#
#""" normalize feature matrix """
#n,d=X_Physical.shape
#for j in range(d):
#    I=np.isnan(X_Physical[:,j])==False
#    X_Physical[:,j]=X_Physical[:,j]-np.mean(X_Physical[I,j])
#    X_Physical[:,j]=X_Physical[:,j]/np.mean(X_Physical[I,j]**2)**0.5
#    plt.hist(X_Physical[I,j])
#    plt.show()
#
#n,d=X_Chemical.shape
#for j in range(d):
#    I=np.isnan(X_Chemical[:,j])==False
#    #X_Chemical[:,j]=np.log(np.mean(X_Chemical[I,j])+X_Chemical[:,j])
#    X_Chemical[:,j]=X_Chemical[:,j]-np.mean(X_Chemical[I,j])
#    X_Chemical[:,j]=X_Chemical[:,j]/np.mean(X_Chemical[I,j]**2)**0.5
#    plt.hist(X_Chemical[I,j])
#    plt.show()
#
#with open('HMOF_DATA.pkl', 'wb') as f:
#    pickle.dump({'API':API,'X_Physical':X_Physical,'X_Chemical_old':X_Chemical_old,'X_Chemical_new':X_Chemical,'Y_Adsorbtion':Y_Adsorbtion,'YAPI_True':YAPI_True,'YAPI_Fixed':YAPI_Fixed}, f)
#
#n,m=YAPI_Fixed.shape
#
#for j in range(m):
#    plt.plot(YAPI_True[:,j],YAPI_Fixed[:,j],'.')
#    plt.title(APIs[j])
#    if j==m-1:
#        plt.xlim([0,1e4])
#    plt.show()
#    
