#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:00:09 2023

Script calculating the proportion of national males born in 2015 that were bred from


@author: jilska2
"""

import pandas as pd
import seaborn as sb
sb.set_style('darkgrid')
import numpy as np
import sys as sys
import os as os
import matplotlib.pyplot as plt

import pop_analysis_functions_v2 as paf

plt.close('all')
if __name__ == "__main__":
    if len(sys.argv) == 3:
        filename=sys.argv[1]
        breed=sys.argv[2]
        
        outfolder = os.path.dirname(os.path.abspath(filename))
        ped = pd.read_csv(filename,encoding='ISO-8859-1')
        try:
            ped.columns = ['Breed','Register', 'RegType','OrigCountry', 'ChampTitle', 'KennelPrefix', 'StudBookNo','StudBookNoGenDate', 'IRN','KCreg', 'DogName', 'sex', 'DOB','IRN_sire','KCreg_sire','DogName_sire','IRN_dam','KCreg_dam','DogName_dam','CS','CS_type','AI','Multi_sire']
        except ValueError:
            ped.columns = ['DogGuid','Register', 'RegType','OrigCountry', 'ChampTitle', 'KennelPrefix', 'StudBookNo','StudBookNoGenDate', 'IRN','KCreg', 'DogName', 'sex', 'Breed','DOB','IRN_sire','KCreg_sire','DogName_sire','IRN_dam','KCreg_dam','DogName_dam','CS','CS_type','AI','Multi_sire']
            
            ## Date handling - this may be time consuming, but pandas date handling is more straightforward than using numpy
        ## Fill missing values with 01/01/1900
        try:
            ped['DOB'] = ped['DOB'].fillna('01/01/1900 00:00:00')
        except AttributeError:
            ped['DOB'] = ped['DOB'].fillna('01/01/1900')
        
        ## Convert DOB to datetime        
        ped['DOB'] = pd.to_datetime(ped['DOB'], dayfirst=True, errors='coerce')
        ped['yob'] = ped['DOB'].dt.year
        
        ## Select only dogs born after 2014
        df = ped.loc[ped['yob']>2014]
        
        ## Select national males born in 2015
        males = df.loc[(df['yob']==2015) & (df['Register']=='Breed register') & (df['RegType']!='Importations') & (df['sex']=='Dog')]
        ## Find males used in breeding
        sm = males.loc[males['IRN'].isin(df['IRN_sire'])]
        try:
            print("{} out of {} ({:3.2f}%) national males born in 2015 were used in breeding to date".format(len(sm),len(males),len(sm)/len(males)*100))
        except ZeroDivisionError:
            pass
        print("N_males,{}".format(len(males)))
        print("N_sires,{}".format(len(sm)))
        
    else:
        print("Provide filepath and breed abbreviation")
        
else:
    print("Provide filepath and breed abbreviation")