#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:57:06 2019

reconsile indexing 
import element counts 
get unit cell volumes
import PH

@author: Hook
"""

import numpy as np
import pandas as pd 
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import pickle

df_reference = pd.read_csv('table_cCGA_nCGA_mCGA_selC4ads_saV_saG_mP_dP_vF.csv')

n,d=df_reference.shape

MOF_FILENAMES=sio.loadmat('MOF_NAMES.mat')['MOF_NAMES']

m=len(MOF_FILENAMES)
file_id=np.zeros(m)
for i in range(m):
    t=16
    while MOF_FILENAMES[i][0][0][t]!='_':
        t=t+1
    file_id[i]=int(MOF_FILENAMES[i][0][0][16:t])

ix=np.argsort(file_id)
file_id_sorted=file_id[ix]
MOF_FILENAMES_SORTED=[MOF_FILENAMES[ix[i]] for i in range(m)]

""" note n>m """

pd_id=np.array(df_reference['Hypothetical MOF ID'])

file_in_pd=np.zeros(m)
pd_in_file=np.zeros(n)

i=0
j=0
while i<n and j<m:
    if pd_id[i]==file_id_sorted[j]:
        file_in_pd[j]=i
        pd_in_file[i]=j
        i=i+1
        j=j+1
    else:
        if pd_id[i]<file_id_sorted[j]:
            pd_in_file[i]=np.nan
            i=i+1
        else:
            if pd_id[i]>file_id_sorted[j]:
                file_in_pd[j]=np.nan
                j=j+1

""" check a few examples just to be sure """

for t in range(100):
    i=np.random.randint(n)
    print(pd_id[i]-file_id_sorted[int(pd_in_file[i])])
    print(MOF_FILENAMES_SORTED[int(pd_in_file[i])])
    print(pd_id[i])

for t in range(100):
    i=np.random.randint(m)
    print(pd_id[int(file_in_pd[i])]-file_id_sorted[i])
    print(MOF_FILENAMES_SORTED[i])
    print(pd_id[int(file_in_pd[i])])
    
""" now compare element count matrices """
""" old element count matrix is all wrong! """
""" import all XYZ files and count atoms again """
""" also recalcualte unit cell volumes """

ATOMS=['H','C','N','O','F','Cl','Br','V','Cu','Zn','Zr','I']
natoms=len(ATOMS)
ELEMENT_COUNTS_NEW=np.zeros((n,natoms))
       
for i in range(n):
    if np.mod(i,1000)==0:
        print(i)
    if np.isnan(pd_in_file[i])==False:
        H=sio.loadmat('HMOF_XYZ/'+MOF_FILENAMES_SORTED[int(pd_in_file[i])][0][0])
        A=H['A']
        for j in range(len(A)):
            for k in range(natoms):
                if len(A[j][0])>0:
                    if A[j][0][0]==ATOMS[k]:
                        ELEMENT_COUNTS_NEW[i,k]=ELEMENT_COUNTS_NEW[i,k]+1       

        
""" unit cell calculation """
unit_vols_pd=np.zeros(n)
for i in range(n):
    if np.mod(i,1000)==0:
        print(i)
    if np.isnan(pd_in_file[i])==False:
        H=sio.loadmat('HMOF_XYZ/'+MOF_FILENAMES_SORTED[int(pd_in_file[i])][0][0])
        angles=H['cell_angles'][0]*np.pi/180
        cell_lengths=H['cell_lengths'][0]        
        cx=cell_lengths[2]*np.cos(angles[1])
        cy=cell_lengths[2]*(np.cos(angles[0])-np.cos(angles[1]))*np.cos(angles[2])/np.sin(angles[2])
        cz=cell_lengths[2]*np.sqrt(1-np.cos(angles[1])**2-(np.cos(angles[0])-np.cos(angles[1]))*np.cos(angles[2])/np.sin(angles[2])**2)
        M=np.array([[cell_lengths[0],0,0],[cell_lengths[1]*np.cos(angles[2]),cell_lengths[1]*np.sin(angles[2]),0],[cx,cy,cz]])
        unit_vols_pd[i]=np.linalg.det(M)

""" import PH data """
PH_MATRIX=np.zeros((n,20))
nskip=0
for i in range(n):
    if np.mod(i,1000)==0:
        print(i)
    loaded=False
    try:
        H=sio.loadmat('HMOF_PH_REORDERED/PH_'+str(i))
        loaded=True
    except:
        PH_MATRIX[i,:]=np.nan
        nskip=nskip+1
        print('skippy  '+str(nskip))
    if loaded:
        H1=H['H1']
        for h in H1:
            if h[1]>0:
                h[1]=min(h[1],10)
                persistence=(h[1]-h[0])/h[1]
                PH_MATRIX[i,int(2*h[1]-1)]+=persistence


PH_MATRIX_scaled=np.divide(PH_MATRIX,unit_vols_pd.reshape(-1,1))
I=[i for i in range(n) if np.isnan(np.sum(PH_MATRIX_scaled[i]))==0]

R=[2,4,6,8,10]
l=2 # smoothing factor
W=np.zeros((20,5))
for i in range(20):
    for j in range(5):
        W[i,j]=np.exp(-np.power(i-R[j],2)/(2*l**2))
    W[i,:]=W[i,:]/np.sum(W[i,:])
    
PH_MATRIX_scaled_reduced=np.matmul(PH_MATRIX_scaled,W)  

""" scaled version of the element counts """

ELEMENT_COUNTS_NEW_scaled=np.divide(ELEMENT_COUNTS_NEW,unit_vols_pd.reshape(-1,1))

with open('CHEM_AND_PH_DATA.pkl', 'wb') as f:
    pickle.dump({'ATOMS':ATOMS,'X_Chemical':ELEMENT_COUNTS_NEW_scaled,'X_Topological':PH_MATRIX_scaled_reduced}, f)

