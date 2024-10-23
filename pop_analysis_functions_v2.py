#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:41:10 2022

Functions for the population analysis. 

Large update on 12/08/2022

@author: jilska2
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('darkgrid')
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
from matplotlib.offsetbox import AnchoredText
import time
import numpy.ma as ma
import math as math
import matplotlib.ticker as ticker
from datetime import date

def pd_to_np(df):
    '''
    Function extracting columns from pandas dataframe into numpy arrays.
    '''
    ## Convert ped to numpy arrays for faster processing
    pid = np.array(df['pid'])
    psire = np.array(df['ps'])
    pdam = np.array(df['pd'])
    yob = np.array(df['yob'])
    sex = np.array(df['sex'])
    dserial = np.array(df['dserial'])
    F = np.array(df['F'])
    
    return pid,psire,pdam,yob,sex,dserial,F

def assign_litters(pid,psire,pdam,dserial):
    '''
    Function assigning litter numbers
    
    '''
    start = time.time()
    litters = np.zeros(len(pid))

    nosire = np.where(psire==0)[0]
    nodam = np.where(pdam==0)[0]
    nodob = np.where(dserial==693962)[0]
    
    ## Find intersection of these:
    t1 = np.union1d(nosire,nodam)
    founds = np.union1d(t1,nodob)
    
    ## For each - assign a unique litter number
    for i in range(founds.shape[0]):
        litters[founds[i]]=i+1
        
    last_lit = max(litters)
    # print("Unique litters assigned to {} animals with unknown parent(s)".format(t1.shape[0]))
    # print("Unique litters assigned to {} animals with unknown DOB".format(nodob.shape[0]))
    # print("Number of litters assigned = ",int(last_lit))
    
    ## Find remaining animals
    r = np.where(litters==0)
    
    ## Create a list of DOB,sire,dam combinations
    mat = np.array([psire,pdam,dserial]).T
    ## From the ones that require assigning (mat[r]), select unique
    qt = np.unique(mat[r],axis=0)
    gh = {}
    ## Create a dictionary where key is sire, dam,dserial tuple, and litter number is value
    for q in qt:
        gh[tuple(q)]=last_lit+1
        last_lit+=1
    # print("Assigning litters to dogs with known parents and DOBs")    
    for i in r[0]:
        k=tuple(mat[i])
        litters[i]=gh[k]
    print("Completed assigning {} litters, across {} dogs".format(int(last_lit),litters.shape[0]))
    end = time.time()
    print("Time lapsed: {:3.2f}".format(end - start))
    print("\n#######################################################")
    
    return litters

def basic_stats(pid,yob,litters):
    '''
    Function returning basic stats relating to years and litters
    Requires vector of pid's, vector of yob's and vector of litters.
    
    Returns nanim, max and min yob, uniq_yob, tborn and nlitters.
    
    NOTE - this version assumes that all breeds' reg starts in 1990. This isn't true for some breeds. 
    Inclusion of 0s for years preceding the first registered dog artificially increases reg coeff later on. 
    Use basic_stats_b which uses the first year that a dog was actually registered.
    '''
    
    ## Set some basic stats
    nanim = len(pid)             
    maxyob=int(max(yob))           
    minyob = int(min(yob))
    ## List of years from 1990 to max-1
    if maxyob==2022:
        uniq_yob = [x for x in range(1990,maxyob,1)]
    else:
        uniq_yob = [x for x in range(1990,maxyob+1,1)]
    nyrs = len(uniq_yob)  

    tborn = np.zeros(nyrs)
    nlitts = np.zeros(nyrs)
    n=0
    ## Loop through years
    for y in uniq_yob:
        ## Find indices based on yob
        x = np.where(yob==y)
        ## enter number of dogs born in that period
        tborn[n]=len(x[0])
        ## Enter number of litters
        nlitts[n] = len(np.unique(litters[x]))
        n+=1
        
    return nanim, maxyob, minyob, uniq_yob, nyrs, tborn,nlitts


def basic_stats_b(pid,yob,litters):
    '''
    Function returning basic stats relating to years and litters
    Requires vector of pid's, vector of yob's and vector of litters.
    
    Returns nanim, max and min yob, uniq_yob, tborn and nlitters.
    
    Updated 11/08/2023 - uniq_yob is now years between the yob for first registered dog and the yob of last registered
    '''
    
    ## Set some basic stats
    nanim = len(pid)             
    maxyob=int(max(yob))           
    minyob = int(min(yob))
    ## List of years from 1990 to max-1
    if maxyob==2022:
        uniq_yob = [x for x in range(minyob,maxyob,1)]
    else:
        uniq_yob = [x for x in range(minyob,maxyob+1,1)]
    nyrs = len(uniq_yob)  

    tborn = np.zeros(nyrs)
    nlitts = np.zeros(nyrs)
    n=0
    ## Loop through years
    for y in uniq_yob:
        ## Find indices based on yob
        x = np.where(yob==y)
        ## enter number of dogs born in that period
        tborn[n]=len(x[0])
        ## Enter number of litters
        nlitts[n] = len(np.unique(litters[x]))
        n+=1
        
    return nanim, maxyob, minyob, uniq_yob, nyrs, tborn,nlitts

def count_tab(df,col):
    '''
    Function returning count of values in a column, and a normalised count - relative frequency, dividing all values by the sum of values.
    Normalised count returns values sorted from the most frequent, so no need to sort.
    '''
    a = df[col].value_counts().reset_index()
    a.columns=['{}'.format(col),'{}_N'.format(col)]
    b = df[col].value_counts(normalize=True).reset_index()
    b.columns=['{}'.format(col),'{}_freq'.format(col)]
    c = pd.merge(a,b,on='{}'.format(col))
    
    return c


def count_reg_type(df):
    '''
    Function printing out counts and frequencies for all combinations of Register and Regtype.
    This is due to the fact that there are incorrect registrations types in incorrect Registries.
    The information has not be clarified by Reg department yet, so producing a table for all, then
    sums can be done in the output file. 
    '''
    a = df[['Register','RegType']].value_counts().reset_index()#.to_csv("reg_type_errors.csv")
    b = df[['Register','RegType']].value_counts(normalize=True).reset_index()
    
    c = pd.merge(a,b,on=['Register','RegType'])
    c.columns = ['Register','RegType','N','freq']
    ## Multiply freq to %
    c['freq']=round(c['freq']*100,2)
    c.sort_values(by=['Register'],inplace=True)
    
    return c

def reg_type_stats(df,outfolder):
    '''
    Function printing out basic statistics based on registration types. 
    Also produces stats for imports. 
    
    '''

    
    regsn = count_reg_type(df)
    
    ## Updated in July 2023 - count only dogs in Breed Registry with Litter reg
    n = regsn.loc[(regsn['Register']=='Breed register') & (regsn['RegType'].str.contains("Litter"))]['N'].sum()
    print("{} dogs in Breed register and litter registrations\n".format(n))
    regsn.to_csv("{}/out_tables/registration_types.csv".format(outfolder), index=False)

    ## Import by country - update July 2023 - only Importations within Breed register included
    imps = df.loc[(df['RegType']=='Importations') & (df['Register']=='Breed register')]
    countries = count_tab(imps,'OrigCountry')
    ## Imports from unknown country
    cu = len(imps[imps['OrigCountry'].isna()])
    print("{} imports from {} countries ({} with country unknown)".format(len(imps),len(countries),cu))
    countries.to_csv("{}/out_tables/imports_by_country.csv".format(outfolder), index=False)
    
    print("Top 3 countries:")
    for i in range(3):      
        try:
            print("{} - {:3.2f}% (n={})".format(countries.iloc[i,0], countries.iloc[i,2]*100, countries.iloc[i,1]))
        except IndexError:
            pass
    return imps

def print_my_summary(model):
    
    a = model.params.reset_index()
    a.columns=['parameter','slope']
    b = model.bse.reset_index()
    b.columns = ['parameter','SE']
    c = model.pvalues.reset_index()
    c.columns = ['parameter','p-value']
    
    t = pd.merge(a,b,on=['parameter'])
    d = pd.merge(t,c,on=['parameter'])
    
    return d

def imp_trends(imps, psire,pdam,outfolder):
    '''
    Function fitting a regression line, and plotting trend of the import numbers by yob
    '''
    print("\nCalculating import trends")
    
    
    ## Find number of imports with known sex
    ks = len(imps.loc[~imps['sex'].isna()])
    print("{} imports with known sex".format(ks))

    
    ## Find male imports - index, 154 male imports
    impm = np.where(imps['sex']=='Dog')
    ## Find male imports who became sires, 72 import sires
    impsx = np.where(np.in1d(imps['pid'],psire))
    ## Calculate proportion
    try:
        print("{}/{} ({:3.2f}%) male imports have been used in breeding".format(len(impsx[0]),len(impm[0]),len(impsx[0])/len(impm[0])*100))
    except ZeroDivisionError:
        print("{}/{} (NA) male imports have been used in breeding".format(len(impsx[0]),len(impm[0])))
    
    ## Find female imports - index
    impf = np.where(imps['sex']=='Bitch')
    ## Find female imports who became dams
    impdx = np.where(np.in1d(imps['pid'],pdam))
    ## Calculate proportion
    try:
        print("{}/{} ({:3.2f}%) female imports have been used in breeding".format(len(impdx[0]),len(impf[0]),len(impdx[0])/len(impf[0])*100))
    except ZeroDivisionError:
        print("{}/{} (NA) female imports have been used in breeding".format(len(impdx[0]),len(impf[0])))

    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    ## Calculate a regression line to see whether there is a significant trend in number of imports over YOB
    ## Imps by yob
    iby = imps[['IRN','yob']].drop_duplicates()
    
    ## Count number of imports per year
    ibyc = iby.groupby('yob').count().reset_index()
    ibyc.to_csv("{}/out_tables/trends_import.csv".format(outfolder),index=False)
    
    ## Fit regression line
    ## Remove imports with unknown yob, and born in the latest year
    # ibyc = ibyc.loc[(ibyc['yob']>1989) & (ibyc['yob']<max(ibyc['yob']))]
    ibyc = ibyc.loc[(ibyc['yob']>1989) & (ibyc['yob']<2022)]
    try:
        imp_yob_r = ols('IRN ~ yob', data=ibyc).fit()
        ibya,ibyb = imp_yob_r.params
        ibyr = print_my_summary(imp_yob_r)
        
        ibyr['p-value']=ibyr['p-value'].astype('float')
        pval = ibyr.loc[ibyr['parameter']=='yob']['p-value'].values[0]
        se = ibyr.loc[ibyr['parameter']=='yob']['SE'].values[0]
        ibyr[['slope','SE']] = ibyr[['slope','SE']].astype('float').round(2)
    
        if ibyb>0:
            dirc = 'increasing'
        else:
            dirc = 'decreasing'
            
        if pval<0.001:
            print("Significant {} trend in number of imports over yob, p<0.001".format(dirc))
        elif pval<0.01:
            print("Significant {} trend in number of imports over yob, p<0.01".format(dirc))
        elif pval<0.05:
            print("Significant {} trend in number of imports over yob, p<0.05".format(dirc))  
        else:
            print("Trend in number of imports over yob is not significant")
            
        print("Slope: {:3.2f}".format(ibyb))
    except ValueError:
        print("Could not calculate regression line")
        
    ## Plot
    fig, ax = plt.subplots()
    c = np.linspace(start=1990,stop=2022,num=32)
    
    ax = sb.lineplot(data=ibyc.loc[ibyc['yob']>1989],x='yob',y='IRN',marker='o',mec='#0000ff',mfc='#9999ff',color='#0000ff')
    plt.gca().set_xlim(left=1989)
    plt.xlabel("Year of birth")
    plt.ylabel("Number of imports")
    try:
        abline = [ibyb*i + ibya for i in c]
        plt.plot(c,abline,color='#000000',linestyle=':',label='regression (m)', linewidth=1)
        anchored_text = AnchoredText("Year effect = {:3.2f}, \np-value = {}".format(ibyb,pval.round(4)), loc=2)
        ax.add_artist(anchored_text)
    except NameError:
        pass
    plt.ylim(0,)
    
    
    plt.savefig("{}/out_plots/trends_import.jpg".format(outfolder), dpi=199)
    print("Plot saved in out_plots/trends_import.jpg")
    
    try:
        return ibyc,ibya,ibyb,se,pval
    except UnboundLocalError:
        return

def champ_stats(df,outfolder):
    '''
    Function preparing counts of dogs with particular championship stats.
    Added count of dogs with StudBook No - these may include OB/AG and FT working dogs. 
    Approximating purpose bred dogs    
    '''
    print("Calculating champion statistics")
    ## Champion Titles
    ## In csv format, missing values are read in as nans
    df.loc[df['ChampTitle']=='nan','ChampTitle']=np.NaN
    ## Correction - Assuming that 'CH FT CH'=='FT CH' and 'CH SH CH==SH CH'
    champs = df.loc[~df['ChampTitle'].isna()]
    
    df['ChampTitle']=df['ChampTitle'].astype('str')
    df.loc[df['ChampTitle']=='CH FT CH','ChampTitle']='FT CH'
    df.loc[df['ChampTitle']=='CH SH CH','ChampTitle']='SH CH'
    temp = count_tab(df,'ChampTitle')
    
    ## Dogs with studbook numbers - show dogs!
    ## Stud book numbers
    sdb = df.loc[~df['StudBookNo'].isna()]
    row = ['StudBookNo',len(sdb),len(sdb)/len(df)]
    temp.loc[len(temp)+1]=row
    temp['%'] = round(temp['ChampTitle_freq']*100,2)
#     print(temp)
    temp.to_csv("{}/out_tables/champ_stats.csv".format(outfolder), index=False)
    print()
    return champs,sdb

def kennel_stats(df,outfolder):
    '''
    Stats per kennel prefix. 
    '''
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    kp = df.loc[~df['KennelPrefix'].isna()]
    print("{} dogs with known Kennel Prefix".format(len(kp)))
    kpc = count_tab(df,'KennelPrefix')
    print("The top kennel prefix - {}, produced {:3.2f}% of dogs (with known kennel prefix)".format(kpc.iloc[0,0],kpc.iloc[0,2]*100))
    
    ## Histogram of kennels - not sure if needed, most likely will be same picture across all breeds...
    sb.histplot(x=kpc['KennelPrefix_N'],kde=False,color='#0000ff',alpha=1)
    plt.xlabel("Dogs per kennel")
    plt.ylabel("Count of kennels")
    plt.savefig("{}/out_plots/kennel_dogsN.jpg".format(outfolder), dpi=199)
    print("Output saved to out_plots/kennel_dogsN.jpg")
    return

def prop_born(yob_counts,y,n):
    '''
    Function to be applied to pandas dataframe row. 

    Parameteres:
    --------------
    
    numpy array with counts
    y - year in question
    n - number to be used as nominator
    
    
    Returns:
    --------------
    proportion born
    '''
    
    ## Find index of value
    yi = np.where(yob_counts[0]==y)[0][0]
    ## find number born
    yb = yob_counts[1][yi]
    ## Calculate born in kennel as proportion of all
    
    return n/yb*100

def top_kennels(df,yob_counts,outfolder):
    '''
    Function presenting top 3 or top 2 kennels per year, from 2000 onward. 
    Prints out a plot.
    '''
    print("\nTop kennels:")
    ## Select recent dogs with known Kennel Prefix
    kp = df.loc[(~df['KennelPrefix'].isna()) & (df['yob']>1999)]
    ## Group by year and prefix
    kpc = kp.groupby(['yob','KennelPrefix']).count().reset_index()
    ## Sort values
    kpc = kpc.sort_values(by=['yob','IRN'])
    ## Select top 3 kennels per year
    tk = kpc.groupby(['yob']).tail(3)
    ## If large number of kennels - reduce to top 2
    if tk['KennelPrefix'].nunique()>10:
        tk = kpc.groupby(['yob']).tail(2)

    tk = tk[['yob','KennelPrefix','IRN']]
    ## Express as percentage of born in that year
    tk['prop']=tk.apply(lambda x: prop_born(yob_counts,x['yob'],x['IRN']), axis=1)
    
    ## Plot
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    # sb.set(rc={'figure.figsize':(6.4,4.8)})
    ax = sb.lineplot(x=tk['yob'],y=tk['prop'],hue=tk['KennelPrefix'],style=tk['KennelPrefix'],markers=True,linewidth=0.8)
    plt.xlabel("Year of birth")
    plt.ylabel("Proportion of all dogs born in that year (%)")
    plt.ylim(0,)
    plt.xlim(1999,)
    plt.title("Top kennels by year")
    # Put the legend out of the figure
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    sb.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig("{}/out_plots/trends_topKennels.jpg".format(outfolder), dpi=199)
    plt.clf()
    print("Plot saved in out_plots/trends_topKennels.jpg")
    return

def FUNC01_ngen(ped,nanim,pid,psire,pdam,outfolder):
    '''
    Function calculating the average number of generations per year of birth. 
    
    Parameters:
    ----------
    ped - original pedigree pandas dataframe
    nanim - number of animals in pedigree
    pid - recoded id for the dogs
    psire - recoded id for sire
    pdam - recoded id for dam
    outfolder
    
    Returns
    ----------
    Saves a plot with mean number of generations of pedigree. 
    
    '''
    print("\nCalculating average number of generations per year of birth")
    ## Vectorised - pandas is 6x slower!!
    ## Create vectors to hold the numbers
    scomp = np.zeros(nanim)
    dcomp = np.zeros(nanim)
    comp = np.zeros(nanim)
    cgen = np.zeros(nanim)

    ## Loop through animals in pedigree
    for i in range(nanim):
        ## Sires: if sire unknown - 1 gen
        if psire[i]==0:
            scomp[i]=1
        else:
            ## Find index of sire
            s = np.where(pid==psire[i])
            ## Enter sire's value
            try:
                scomp[i] = comp[s]
            except ValueError:
                scomp[i] = 1
        ## Same for dams
        if pdam[i]==0:
            dcomp[i]=1
        else:
            d = np.where(pid == pdam[i])
            try:
                dcomp[i] = comp[d]
            except ValueError:
                dcomp[i] = 1
        ## Add the generations behind sire and dam
        comp[i] = scomp[i]+dcomp[i]
        ## Calculate logarithmic mean
        cgen[i] = np.log(comp[i])/np.log(2)
        
    # print("FUNC01 testing done")


    ## Read into a table with pid and yob - this section may be potentially sped up if converted to vector operations. 
    ngen = ped[['pid','yob']].reset_index(drop=True)
    ## Round to 4 decimal places - I was getting somewhat different result when used the higher precision!
    ngen['cgen'] = cgen.round(4)
    ## Group by yob and calculate mea
    ny = ngen.groupby('yob').mean().reset_index()
    ny.columns=['yob','pid','mean_ngen']
    ## Save table to file - note, it only has yobs with actual dogs born!
    #print("Table with yob and mean number of generations saved to ngen_log.csv")
    ny[['yob','mean_ngen']].to_csv("{}/out_tables/ngen_log.csv".format(outfolder),index=False)
    ## Plotting for dogs since 1980 only
    ny2 = ny.loc[ny['yob']>1989]
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    plt.plot(ny2['yob'],ny2['mean_ngen'],'o-',mec='#0000ff',mfc='#9999ff',color='#0000ff',linewidth=0.9)
       
    # plt.gca().set_xlim(left=1989)
    # plt.gca().set_ylim(bottom=1)
    plt.xlim(1989,)
    plt.ylim(1,)
    #plt.yticks(np.arange(1, max(ny2['mean_ngen'])+1, 2.0))
    plt.title("Mean # generations of pedigree by year of birth")
    plt.xlabel("Year of birth")
    plt.ylabel("Mean number of generations of pedigree")
    plt.savefig("{}/out_plots/trends_ngen.jpg".format(outfolder), dpi=199)
    print("Plot saved to trends_ngen.jpg")
#     end = time.time()
#     print("Time lapsed: {:3.2f}".format(end - start))
    # print("\n#######################################################")
    
    return

def FUNC02_census_stats(nyrs,uniq_yob,yob,psire,pdam,litters,outfolder,ofname):
    print("\nCalculating census statistics per year of birth")
    nsires = np.zeros(nyrs)
    ndams = np.zeros(nyrs)
    tborn = np.zeros(nyrs)
    nlitts = np.zeros(nyrs)
    maxperc = np.zeros(nyrs)
    maxn = np.zeros(nyrs)
    medn = np.zeros(nyrs)
    modn = np.zeros(nyrs)
    meann = np.zeros(nyrs)
    sdn = np.zeros(nyrs)
    top50 =np.zeros(nyrs)
    top25 = np.zeros(nyrs)
    top10 = np.zeros(nyrs)
    top5 = np.zeros(nyrs)
    
    g = open("{}/out_tables/{}.csv".format(outfolder,ofname),'w')
    g.write('yob,tborn,nlitters,ndams,nsires,max%,max_n,median_n,mode_n,mean_n,SD_n,top50%,top25%,top10%,top5%\n')
    
    n=0
    ## Loop through years
    for y in uniq_yob:
        ## Find indices based on yob
        x = np.where(yob==y)
        ## enter number of dogs born in that period
        tborn[n]=len(x[0])
        ## Enter number of litters
        nlitts[n] = len(np.unique(litters[x]))
        if len(x[0])>1:
    
            ## Get the counts per sire
            t = np.array(np.unique(psire[x], return_counts=True))
            mask = np.argsort(t[1])[::-1]
            ts = t[:,mask]
            ## Calculate the frequency as a proportion
            perc = np.array([[x/sum(ts[1])*100 for x in ts[1]]])
            ## Save number of sires
            nsires[n]=len(perc[0])
            ndams[n]=len(np.unique(pdam[x]))
    
            ## Save the sire stats
            maxn[n]=max(ts[1])
            maxperc[n] = perc[0][0]
            medn[n] = np.nanmedian(ts[1])
            modn[n] = stats.mode(ts[1])[0][0]
            meann[n] = np.mean(ts[1])
            
            sdn[n] = np.std(ts[1],ddof=1)
            top50[n] = sum(perc[0][:round(len(perc[0])/2)])
            top25[n] = sum(perc[0][:round(len(perc[0])/4)])
            top10[n] = sum(perc[0][:round(len(perc[0])/10)])
            top5[n] = sum(perc[0][:round(len(perc[0])/20)])
        else:
            nsires[n]=np.NaN
            ndams[n]=np.NaN
            maxn[n]= np.NaN
            maxperc[n] = np.NaN
            medn[n] = np.NaN
            modn[n] = np.NaN
            meann[n] = np.NaN
            sdn[n] = np.NaN
            top50[n] = np.NaN
            top25[n] = np.NaN
            top10[n] = np.NaN
            top5[n] = np.NaN
        n+=1
        
    
    
    for i in range(nyrs):
        g.write('{},{},{},{},{},{:3.2f},{},{},{},{:3.2f},{:3.2f},{:3.2f},{:3.2f},{:3.2f},{:3.2f}\n'.format(uniq_yob[i],tborn[i],nlitts[i],ndams[i],nsires[i],maxperc[i],maxn[i],
                                                                                           medn[i],modn[i],meann[i],sdn[i],top50[i],top25[i],top10[i],top5[i]))
        
    g.close()
#     print("Time lapsed: ",end-start)
    print("Output saved in ","out_tables/{}.csv".format(ofname))
    
    return tborn,nlitts

def get_val(q):
    try:
        return round(q[0],3)
    except TypeError:
        return round(q,3)

## Func04 causing issues - separate out of function
def FUNC04_genint(nyrs,outfolder,uniq_yob,yob,pid,psire,pdam,dserial):
    '''
    Tom's comment:

        "This function goes through pedigree by yob, identifies individuals born in that year that go on to become parents. It then iterates through these future breeding individuals and calculates their sire and dam age at time of their birth. Thus, the s/d count, mean and std generation intervals refere to parents of breeding individuals born in yob i."

    Note - the generation interval is also calculated when only one dog was used in breeding. Initially I have removed it, but the gen int is needed later for calculation of Ne.  


    '''
    ## Create arrays to store results
    siret = np.zeros(nyrs)
    damt = np.zeros(nyrs)
    t = np.zeros(nyrs)
    start = time.time()
    counter = 0
    g = open('{}/out_tables/generation_interval_log.csv'.format(outfolder), 'w')
    g.write("yob,sire_n,mean_sire_t,sd_sire_t,dam_n,mean_dam_t,sd_dam_t,gen_int\n")
    
    for year in uniq_yob:
        ## Find dogs born in yob that become sires
        ## Find indices for dogs born in yob
        z1 = np.where(yob==year)
        ## Find intersection between pid born in this year, and sires
        p1 = np.intersect1d(pid[z1],psire)
        ## Find intersection between pid born in this year, and dams
        p2 = np.intersect1d(pid[z1],pdam)
    
        ## Merge sires and dams, and order
        z = np.union1d(p1,p2)
    
        ## If no parents, or only 1 parent
        if z.size == 0:
            ## Print warning - Tom included a verbal warning in the output file, but as I'm writing to csv, this would be clunky
            ## Just keep values empty - it will be obvious as the columns include count of sires and dams. 
            print("None of dogs born in {} became parents".format(year))
            siret[counter]=np.NaN
            damt[counter]=np.NaN
            t[counter]=np.NaN 
            avs = np.NaN
            cs = np.NaN
            sds = np.NaN
            cd = np.NaN
            avd = np.NaN
            sdd = np.NaN
            gi = np.NaN
            
        else:
            if z.size==1:
                print("Only one dog born in {} became a parent".format(year))
            avs = 0 # mean sire interval
            avd = 0 # mean dam interval
            sxs = 0 # sum sire ages
            sxd = 0 # sum dam ages
            sxxs = 0 # var sire ages
            sxxd = 0 # var dam ages
            cs = 0 # sire tally
            cd = 0 # dam tally
            sds = 0 # SD sire ages
            sdd = 0 # SD dam ages
    
            for j in range(z.size):
                ## For each dog in the selected group z, see whether sire is present:
                ## Indexing is slightly different, as recoding is not necessarily ordered as in Tom's routine
                ## Find the index for that dog in the pid list
                dogi = np.where(pid==z[j])
                try:
                    ## Find sire of that dog
                    y = psire[dogi]
                    if y!=0:
                        ## Find index of the sire in pid
                        sirei = np.where(pid==psire[dogi])
                        ## Exclude records where sire was born in 1900, i.e. unknown
                        if yob[sirei][0]>1900:
                            sxs = sxs + (dserial[dogi]-dserial[sirei]) # sum of sire ages
                            sxxs = sxxs + ((dserial[dogi]-dserial[sirei])*(dserial[dogi]-dserial[sirei])) # variance of sire age
                            cs+=1 # sire tally, counting only found sires!
    
                ## If can't find the sire
                except IndexError:
                    pass
                ## Repeat for dams
                try:
                    x = pdam[dogi]
                    if x!=0:
                        ## Exclude records where sire was born in 1900, i.e. unknown
                        dami = np.where(pid==pdam[dogi])
                        if yob[dami][0]>1900:
                            sxd = sxd + (dserial[dogi]-dserial[dami]) # sum of sire ages
                            sxxd = sxxd + ((dserial[dogi]-dserial[dami])*(dserial[dogi]-dserial[dami])) # variance of sire age
                            cd+=1 # sire tally, counting only found sires!
    
                ## If can't find the dam
                except IndexError:
                    pass        
    
            if cs == 0:
                print("No sires for the year", year)
            else:
    #             print("Found {} sires".format(cs))
                avs = sxs/cs
                varss = (sxxs-(sxs*sxs/cs))/cs
                sds = np.sqrt(varss)
    
            if cd == 0:
                print("No dams for the year", year)
            else:
                ## Calculate the stats
                avd = sxd / cd
                vard = (sxxd - (sxd*sxd/cd))/cd
                sdd = np.sqrt(vard)
    
        ## I don't think this is correct! If cs or cd ==0, then this will calculate gi based on the previous year...
        gi = (avs + avd)/2
        siret[counter]=avs
        damt[counter]=avd
        t[counter]=gi
#         print(year,cs,avs,sds,cd,avd,sdd,gi)
        g.write("{},{},{:4.3f},{:4.3f},{},{:4.3f},{:4.3f},{:4.3f}\n".format(year,cs,avs[0],sds[0],cd,avd[0],sdd[0],gi[0]))
        counter+=1
    
    g.close() 
    end = time.time()
    print("Time lapsed: ",end-start)
    
    ## The calculated generation interval is in days. Convert to years.
    ## Later this could actually go in the file? Days not very informative...
    tyrs = [x/365 for x in t]
    
    return tyrs


def FUNC04_genint_ji(nyrs,outfolder,uniq_yob,yob,pid,psire,pdam,dserial):
    '''
    Tom's comment:

        "This function goes through pedigree by yob, identifies individuals born in that year that go on to become parents. It then iterates through these future breeding individuals and calculates their sire and dam age at time of their birth. Thus, the s/d count, mean and std generation intervals refere to parents of breeding individuals born in yob i."

    Note - the generation interval is also calculated when only one dog was used in breeding. Initially I have removed it, but the gen int is needed later for calculation of Ne.  
    
    Edit: this is a new version of this function. Previous version was buggy with small pedigrees. 


    '''
    ## Create arrays to store results
    siret = np.zeros(nyrs)
    damt = np.zeros(nyrs)
    t = np.zeros(nyrs)
       
    start = time.time()
    counter = 0
    g = open('{}/out_tables/generation_interval_log.csv'.format(outfolder), 'w')
    g.write("yob,sire_n,mean_sire_t,sd_sire_t,dam_n,mean_dam_t,sd_dam_t,gen_int\n")
    
    for year in uniq_yob:
        #print("Processing year", year)
        ## Find dogs born in yob that become sires
        ## Find indices for dogs born in yob
        z1 = np.where(yob==year)
        ## Find intersection between pid born in this year, and sires
        p1 = np.intersect1d(pid[z1],psire)
        ## Find intersection between pid born in this year, and dams
        p2 = np.intersect1d(pid[z1],pdam)
        
        ## Merge sires and dams, and order
        z = np.union1d(p1,p2)
    
        ## If no parents, or only 1 parent
        if z.size == 0:
            ## Print warning - Tom included a verbal warning in the output file, but as I'm writing to csv, this would be clunky
            ## Just keep values empty - it will be obvious as the columns include count of sires and dams. 
            print("None of dogs born in {} became parents".format(year))
            siret[counter]=np.NaN
            damt[counter]=np.NaN
            t[counter]=np.NaN 
            g.write("{},{},{},{},{},{},{},{}\n".format(year,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN))
            
        else:
            if z.size==1:
                print("Only one dog born in {} became a parent".format(year))
                
            avs = 0 # mean sire interval
            avd = 0 # mean dam interval
            sxs = 0 # sum sire ages
            sxd = 0 # sum dam ages
            sxxs = 0 # var sire ages
            sxxd = 0 # var dam ages
            cs = 0 # sire tally
            cd = 0 # dam tally
            sds = 0 # SD sire ages
            sdd = 0 # SD dam ages
    
            for j in range(z.size):
                ## For each dog in the selected group z, see whether sire is present:
                ## Indexing is slightly different, as recoding is not necessarily ordered as in Tom's routine
                ## Find the index for that dog in the pid list
                dogi = np.where(pid==z[j])
                try:
                    ## Find sire of that dog
                    y = psire[dogi]
                    if y!=0:
                        ## Find index of the sire in pid
                        sirei = np.where(pid==psire[dogi])
                        ## Exclude records where sire was born in 1900, i.e. unknown
                        if yob[sirei][0]>1900:
                            sxs = sxs + (dserial[dogi]-dserial[sirei]) # sum of sire ages
                            sxxs = sxxs + ((dserial[dogi]-dserial[sirei])*(dserial[dogi]-dserial[sirei])) # variance of sire age
                            cs+=1 # sire tally, counting only found sires!
            
                ## If can't find the sire
                except IndexError:
                    pass
                ## Repeat for dams
                try:
                    x = pdam[dogi]
                    if x!=0:
                        ## Exclude records where sire was born in 1900, i.e. unknown
                        dami = np.where(pid==pdam[dogi])
                        if yob[dami][0]>1900:
                            sxd = sxd + (dserial[dogi]-dserial[dami]) # sum of sire ages
                            sxxd = sxxd + ((dserial[dogi]-dserial[dami])*(dserial[dogi]-dserial[dami])) # variance of sire age
                            cd+=1 # sire tally, counting only found sires!
            
                ## If can't find the dam
                except IndexError:
                    pass        
    
            if cs == 0:
                print("No sires for the year", year)
            else:
    #             print("Found {} sires".format(cs))
                avs = sxs/cs
                varss = (sxxs-(sxs*sxs/cs))/cs
                sds = np.sqrt(varss)
    
            if cd == 0:
                print("No dams for the year", year)
            else:
                ## Calculate the stats
                avd = sxd / cd
                vard = (sxxd - (sxd*sxd/cd))/cd
                sdd = np.sqrt(vard)
    
            ## I don't think this is correct! If cs or cd ==0, then this will calculate gi based on the previous year...
            gi = (avs + avd)/2
            siret[counter]=avs
            damt[counter]=avd
            t[counter]=gi
            try:
                g.write("{},{},{:4.3f},{:4.3f},{},{:4.3f},{:4.3f},{:4.3f}\n".format(year,cs,avs[0],sds[0],cd,avd[0],sdd[0],gi[0]))
            except TypeError:
                try:
                    g.write("{},{},{},{},{},{:4.3f},{:4.3f},{:4.3f}\n".format(year,cs,avs,sds,cd,avd[0],sdd[0],gi[0]))
                except TypeError:
                    try:
                        g.write("{},{},{:4.3f},{:4.3f},{},{},{},{:4.3f}\n".format(year,cs,avs[0],sds[0],cd,avd,sdd,gi[0]))
                    except TypeError:
                        g.write("{},{},{},{},{},{},{},{}\n".format(year,cs,avs,sds,cd,avd,sdd,gi))
    
            
            # g.write("{},{},{:4.3f},{:4.3f},{},{:4.3f},{:4.3f},{:4.3f}\n".format(year,cs,avs[0],sds[0],cd,avd[0],sdd[0],gi[0]))
            counter+=1
    
    g.close() 
    end = time.time()
    print("Time lapsed: ",end-start)
    
    ## The calculated generation interval is in days. Convert to years.
    ## Later this could actually go in the file? Days not very informative...
    tyrs = [x/365 for x in t]
    
    return tyrs
    
def FUNC05_F_by_year(nyrs,uniq_yob,yob,f,outfolder,subset):
    '''
    Mean inbreeding per yob, with, and without 0s. 
    Initially, this was written as a matrix array. But, it's not really necessary, as the number of entries is small - 
    the only large calculation is done to calculate the mean - and that remains as numpy calculation. 
    Change the output to pandas dataframe'
    '''
    print("\nCalculating mean F per year")
    k = pd.DataFrame(columns=['YOB','n_born','Mean_F','n_non0F','Mean_non0F'])

    for y in uniq_yob:
        x = np.where(yob==y)
        n = x[0].size
        fs = f[x[0]]
        fsnon0 = [x for x in fs if x!=0]
        row = [y,n,np.nanmean(fs),len(fsnon0),np.nanmean(fsnon0)]
        k.loc[len(k)+1]=row
        
    k.to_csv('{}/out_tables/F_by_year_{}.csv'.format(outfolder,subset),index=False)
    print("Output saved to out_tables/F_by_year_{}.csv".format(subset))
        
    return k


def FUNC05_F_by_year_plot(k,outfolder,subset):       
    print("\nPlotting F by year")
    print(len(k))
    ## Calculate correlations - slightly different to Matlab, I can still produce r even if there are NaNs. 
    ## The masked invalid is a function from numpy masked array which gets rid of NaNs
    p30 = ma.corrcoef(ma.masked_invalid(k['Mean_F'][-30:]), ma.masked_invalid(k['Mean_non0F'][-30:]))[0][1]
    p20 = ma.corrcoef(ma.masked_invalid(k['Mean_F'][-20:]), ma.masked_invalid(k['Mean_non0F'][-20:]))[0][1]
    p10 = ma.corrcoef(ma.masked_invalid(k['Mean_F'][-10:]), ma.masked_invalid(k['Mean_non0F'][-10:]))[0][1]
        
    ## Scatter plot nzF versus wzF, annotate points, 
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots()
    ax.scatter(k['Mean_F'],k['Mean_non0F'],edgecolor='#0000cc',facecolor='#9999ff',linewidth=0.4)
    
    for i, txt in enumerate(k['YOB'].astype('Int64')):
        ax.annotate(txt, (k['Mean_F'].iloc[i], k['Mean_non0F'].iloc[i]),fontsize=6,textcoords = 'offset points', ha = 'right', va ='bottom',xytext=(-2,4),bbox=dict(boxstyle='square',pad=100, fc='none', ec='none'))
            
        
    ## Set axes limits to be equal - assuming y-axis (nzF) will have higher values
    ax.set_ylim((k['Mean_F'].min(skipna=True)-0.1*k['Mean_F'].min(skipna=True)),k['Mean_non0F'].max(skipna=True)+0.05*k['Mean_non0F'].max(skipna=True))
    ax.set_xlim(ax.get_ylim())
    
    
    ## Add a diagonal
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,color='r',linestyle='--')
    try:
        anchored_text = AnchoredText("Correlation:\n{} to {} = {:4.3f}\n{} to {} = {:4.3f}\n{} to {} = {:4.3f}".format(int(k['YOB'].iloc[-30]),int(k['YOB'].iloc[-1]),p30,int(k['YOB'].iloc[-20]),int(k['YOB'].iloc[-1]),p20,int(k['YOB'].iloc[-10]),int(k['YOB'].iloc[-1]),p10), loc=4)
    except IndexError:
        anchored_text = AnchoredText("Correlation:\n{} to {} = {:4.3f}\n{} to {} = {:4.3f}\n{} to {} = {:4.3f}".format(int(k['YOB'].iloc[0]),int(k['YOB'].iloc[-1]),p30,int(k['YOB'].iloc[10]),int(k['YOB'].iloc[-1]),p20,int(k['YOB'].iloc[20]),int(k['YOB'].iloc[-1]),p10), loc=4)
        
        
    ax.add_artist(anchored_text)
    ax.set_xlabel("Including F=0")
    ax.set_ylabel("Without F=0")
    
    
    plt.savefig("{}/out_plots/ObsF_0s_compare_{}.jpg".format(outfolder,subset), dpi=199)
    print("Figure saved in out_plots/ObsF_0s_compare_{}.jpg".format(subset))
    
    return

def FUNC05_F_by_year_plot_a(k,outfolder,subset):       
    print("\nPlotting F by year")
    current_year = date.today().year
    
    try:
        ## This version of the function puts hard stops on the years. Otherwise, issues arise if some breeds haven't had registrations in given years
        p3 = k.loc[(k['YOB']>=(current_year-31)) & (k['YOB']<current_year)].dropna(subset=['Mean_F','Mean_non0F'])
        p30 = np.corrcoef(p3['Mean_F'],p3['Mean_non0F'])[0][1]
        p2 = k.loc[(k['YOB']>=(current_year-21)) & (k['YOB']<current_year)].dropna(subset=['Mean_F','Mean_non0F'])
        p20 = np.corrcoef(p2['Mean_F'],p2['Mean_non0F'])[0][1]
        p1 = k.loc[(k['YOB']>=(current_year-11)) & (k['YOB']<current_year)].dropna(subset=['Mean_F','Mean_non0F'])
        p10 = np.corrcoef(p1['Mean_F'],p1['Mean_non0F'])[0][1]   
        
            
        ## Scatter plot nzF versus wzF, annotate points, 
        plt.clf()
        plt.rc('figure', figsize=(6.4,4.8))
        fig, ax = plt.subplots()
        ax.scatter(k['Mean_F'],k['Mean_non0F'],edgecolor='#0000cc',facecolor='#9999ff',linewidth=0.4)
        
        for i, txt in enumerate(k['YOB'].astype('Int64')):
            ax.annotate(txt, (k['Mean_F'].iloc[i], k['Mean_non0F'].iloc[i]),fontsize=6,textcoords = 'offset points', ha = 'right', va ='bottom',xytext=(-2,4),bbox=dict(boxstyle='square',pad=100, fc='none', ec='none'))
                
            
        ## Set axes limits to be equal - assuming y-axis (nzF) will have higher values
        ax.set_ylim((k['Mean_F'].min(skipna=True)-0.1*k['Mean_F'].min(skipna=True)),k['Mean_non0F'].max(skipna=True)+0.05*k['Mean_non0F'].max(skipna=True))
        ax.set_xlim(ax.get_ylim())
        
        
        ## Add a diagonal
        ax.plot([0, 1], [0, 1], transform=ax.transAxes,color='r',linestyle='--')
        anchored_text = AnchoredText("Correlation:\n{} to {} = {:4.3f}\n{} to {} = {:4.3f}\n{} to {} = {:4.3f}".format(current_year-31,current_year-1,p30,current_year-21,current_year-1,p20,current_year-11,current_year-1,p10), loc=4)
        # except IndexError:
        #     anchored_text = AnchoredText("Correlation:\n{} to {} = {:4.3f}\n{} to {} = {:4.3f}\n{} to {} = {:4.3f}".format(int(k['YOB'].iloc[0]),int(k['YOB'].iloc[-1]),p30,int(k['YOB'].iloc[10]),int(k['YOB'].iloc[-1]),p20,int(k['YOB'].iloc[20]),int(k['YOB'].iloc[-1]),p10), loc=4)
            
            
        ax.add_artist(anchored_text)
        ax.set_xlabel("Including F=0")
        ax.set_ylabel("Without F=0")
        
        
        plt.savefig("{}/out_plots/ObsF_0s_compare_{}.jpg".format(outfolder,subset), dpi=199)
        print("Figure saved in out_plots/ObsF_0s_compare_{}.jpg".format(subset))
    except ValueError:
        print("Issues with mean F per yob")
    
    return

def FUNC06_obs_exp_F(k,coan,uniq_yob,outfolder,tyrs):
    '''
    Function plotting observed vs expected inbreeding. Note, there is a difference - Tom used to calculate the coancestry using samples of individuals, I am calculating this for all born in a year. 
    '''
    print("\nPlotting observed and expected F")
    ## Add coancestry to k
    k2 = pd.merge(k,coan,on='YOB',how='outer')
    k = k2
    
    ## When there is only one individual born in a year, then group coancestry is super high at 0.5. remove coancestry from such year
    k2.loc[k2['n_born']==1,'group_coancestry']=np.NaN
    
    ## Remove the last year - introduced with group_coancestry, but we want stats to be up to max(yob)-1
    k2= k2.loc[k2['YOB'].isin([float(x) for x in uniq_yob])]
    
    ## Calculate mean gen. interval
    t = np.mean([x for x in tyrs if x>0])

    ## Offset for expected
    years1 = [x+t for x in k['YOB']]
#     years1 = [y+t for y,t in zip(k['YOB'],tyrs)]
    obsF = k['Mean_F']
    expF = k['group_coancestry']
    nzoF = k['Mean_non0F']
    
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots()
    ax.plot(k['YOB'],obsF,'o-',mec='#0000ff',mfc='#9999ff',color='#0000ff',linewidth=0.9, label='Observed inbreeding')
    plt.plot(years1,expF,'r-',label='Expected inbreeding')
    plt.legend()
    plt.gca().set_xlim(left=1989)
    plt.xlabel("Year of birth")
    plt.ylabel("Mean COI")
    plt.savefig("{}/out_plots/ObsF_expF.jpg".format(outfolder), dpi=199)
    
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots()
    ax.plot(k['YOB'],obsF,'o-',mec='#0000ff',mfc='#9999ff',color='#0000ff',linewidth=0.9, label='Observed inbreeding')
    plt.plot(k['YOB'],nzoF,'k:',marker='o',mfc='#888888',mec='#404040',ms=4,label='Obs zero excluded')
    
    plt.plot(years1,expF,'r-',label='Expected inbreeding')
    plt.legend()
    plt.gca().set_xlim(left=1989)
    plt.xlabel("Year of birth")
    plt.ylabel("Mean COI")
    plt.savefig("{}/out_plots/ObsF_expF_non0.jpg".format(outfolder), dpi=199)
    print("Plots saved in out_plots/ObsF_expF.jpg and ObsF_expF_non0.jpg")
    
    return k

def FUNC_Ne(k,uniq_yob,tyrs,outfolder):
    '''
    Function calculating effective population size. 
    Note - allows selection of the value to be used to approximate F (i.e. allows 'Mean_F','group_coancestry' or even 'Mean_non0F')

    '''
    print("\nCalculating effective population size")
    posT = [x for x in tyrs if x>=0]
    meanT = np.mean(posT)
    
    ## Calculate log of F - coancestry
    dF = []
    Ne = []
    for variable in ['Mean_F','group_coancestry']:
        f1 = np.log([1-x for x in k[variable]])
        y = np.array([x for x in range(1,len(k)+1,1)])

        mask = ~np.isnan(f1)
        slope, intercept, r, p, se = stats.linregress(y[mask],f1[mask])

        deltaF = 1-(math.exp(slope))
        Neff = 1/(2*(deltaF*meanT))
        
        dF.append(deltaF)
        Ne.append(Neff)
        
    ## Read into table
    g = pd.DataFrame(columns=['Statistic','Fbased','Rbased'])
    g['Statistic']=["deltaF per year","Mean generation interval","Ne"]
    g['Fbased']=[dF[0],meanT,Ne[0]]
    g['Rbased'] = [dF[1],meanT,Ne[1]]
    g.to_csv("{}/out_tables/Whole_period_dF_t_Ne.csv".format(outfolder), index=False)
    print("Output saved in out_tables/Whole_period_dF_t_Ne.csv")
    
    return g


def FUNC7_sd_stats(yob, uniq_yob,maxyob, tborn,psire,pdam,outfolder):
    ## In Tom's routine, nps is a matrix of variable sizes. Numpy arrays are fixed in size, so I can't do this. 
    ## The size depends on the percentages per sire, so I also can't set it ahead of time. Instead, I create a new pd dataframe for each i and then merhe
    ## them together. Note - the number of bins from numpy histogram is slightly different to matlab, so results are somewhat different.
    ## Create indices for 5 year blocks
    ## In Tom's version, only complete 5-year blocks are printed. If all data is to be used, do version 2
    y1 = [x for x in range(0,len(uniq_yob),5)]  ## TL version with complete 5year blocks
    # y1 = [x for x in range(0,max(y),5)]  ## v2 - all data
    
    ## Dropping the last years if incomplete blocks
    if max(np.array(uniq_yob)[y1])>max(yob)-1:
        y1 = y1[:-1]
    
    nps = pd.DataFrame(columns=['bin'])
    npd = pd.DataFrame(columns=['bin'])

    period=np.zeros((15,len(y1))) ## this will go to 5yearly stats
    
    np.set_printoptions(suppress=True)
    for i in range(len(y1)):
#         print("Running i=",i)
        col = str(y1[i])
        block=[x for x in range(y1[i],y1[i]+5) if x<len(uniq_yob)]
        period[0,i]=np.mean(tborn[block])

        ## Find indices of yob's within the block
        yb = np.where((yob>=uniq_yob[block[0]]) & (yob<=uniq_yob[block[-1]]))
        ## Sire counts - in Matlab, counts are for all sires. Here, only for the sires from the group
        ts1 = np.array(np.unique(psire[yb], return_counts=True))

        if len(ts1[0])==0: ## if there are no sires found
            nps.loc[0,'bin']=0
            nps.loc[0,col]=0
            period[2:8,i]=0
        else:
            ## Update Jul 2023 - remove unknown sires!
            if ts1[0][0]==0:
                ts1=ts1[:,1:]
            mask = np.argsort(-ts1[1])
            ts1 = ts1[:,mask]
            perc = np.array([[x/sum(ts1[1])*100 for x in ts1[1]]])
            ts = np.append(ts1,perc,axis=0)
            ## Put the sire counts into a histogram. TL has produced a large number of bins - but I'm not sure whether this is actually necessary?
            ## It should be possible to just allow for automatic categorisation? 
            ## In any case, the results are not going to be identical to what Matlab produces, as the logic in assigning to bin
            ## is different
            bins=np.arange(0,max(perc[0])+0.01,0.01)
            n1, xout1 = np.histogram(ts[2,:], bins=bins)
            temp = pd.DataFrame(columns=['bin',col])
            temp['bin']=xout1[:-1]
            temp[col]=n1
            nps1 = pd.merge(nps,temp,on='bin',how='outer')
            nps = nps1

            period[1,i] = len(ts1[0]) ## no sires in a block
            period[2,i] = max(ts1[1]) ## max progeny per sire
            period[3,i] = np.mean(ts1[1]) ## mean #prog per sire
            period[4,i] = np.nanmedian(ts1[1]) ## median
            period[5,i] = stats.mode(ts1[1])[0][0] ## mode
            period[6,i] = np.std(ts1[1],ddof=1)
            period[7,i] = stats.skew(ts1[1])

        ## Dam counts - repeat the steps
        td1 = np.array(np.unique(pdam[yb], return_counts=True))

        if len(td1[0])==0: ## if there are no sires found
            npd.loc[0,'bin']=0
            npd.loc[0,col]=0
            period[9:,i]=0
        else:
            ## Update Jul 2023 - remove unknown dam!
            if td1[0][0]==0:
                td1 = td1[:,1:]
            mask = np.argsort(-td1[1])
            td1 = td1[:,mask]
            perc = np.array([[x/sum(td1[1])*100 for x in td1[1]]])
            td = np.append(td1,perc,axis=0)
            ## Put the sire counts into a histogram. TL has produced a large number of bins - but I'm not sure whether this is actually necessary?
            ## It should be possible to just allow for automatic categorisation? 
            ## In any case, the results are not going to be identical to what Matlab produces, as the logic in assigning to bin
            ## is different
            bins=np.arange(0,max(perc[0])+0.01,0.01)
            n2, xout2 = np.histogram(td[2,:], bins=bins)

            temp = pd.DataFrame(columns=['bin',col])
            temp['bin']=xout2[:-1]
            temp[col]=n2
            npd1 = pd.merge(npd,temp,on='bin',how='outer')
            npd = npd1

            period[8,i] = len(td1[0]) ## no sires in a block
            period[9,i] = max(td1[1]) ## max progeny per sire
            period[10,i] = np.mean(td1[1]) ## mean #prog per sire
            period[11,i] = np.nanmedian(td1[1]) ## median
            period[12,i] = stats.mode(td1[1])[0][0] ## mode
            period[13,i] = np.std(td1[1],ddof=1)
            period[14,i] = stats.skew(td1[1])


    nps.columns=['bin']+list(np.array(uniq_yob)[y1])
    npd.columns=['bin']+list(np.array(uniq_yob)[y1])
    np.set_printoptions(suppress=True)
    
    ## Find out why still saving in scientific notation
    a = pd.DataFrame(period)

    q = ['{} - {}'.format(x,x+4) for x in list(np.array(uniq_yob)[y1])]
    if int(q[-1].split('-')[1])>maxyob:
        q[-1]=q[-1].split('-')[0]+'-'
    
    a.columns = q
    a.index=['mean_Nborn','n_sires','max#prog_Sire','mean#prog_Sire','median#prog_Sire','mode#prog_Sire','SD#prog_Sire','skew#prog_Sire',
            'n_dams','max#prog_Dam','mean#prog_Dam','median#prog_Dam','mode#prog_Dam','SD#prog_Dam','skew#prog_Dam']
    a = a.round(2)
    a.to_csv("{}/out_tables/5yearly_stats.csv".format(outfolder))
    
    return y1, nps, npd

def FUNC7_dfNe_stats(y1,uniq_yob,tyrs,k,variable,outfolder):
    
    f1 = np.log([1-x for x in k[variable]])

    period1=np.zeros((3,len(y1))) ## this will go to 5yearly dF and Ne
    yt = np.array([x for x in range(1,len(uniq_yob)+1,1)])
    
    for i in range(len(y1)):
#         print(i)
        block=[x for x in range(y1[i],y1[i]+5) if x<len(uniq_yob)]

        ## EDIT - in Tom's version, the periods are inclusive of both end years, so 1990-1995 includes dogs born in 1990 and 1995. 
        ## I have changed it here, so that the periods are calculated for 4 years only - on the same example inclusive of the 1990, but not 1995.
        ## This could probably be simplified - not sure why it's necessary, could be sorted with the >/< earlier on?
#         try:
#             if min(block)>=max(y1):
#                 block1 = block
#             else:
#                 ## Note - in this version, blocks have 6 years included!!
#                 block1 = block + [max(block)+1]
#         except ValueError:
#             block1=block

        block1 = block
        
        try:            
            if block1[-1]>max(yt):
                ft = f1[min(block1):max(yt)]
            else:
                ft = f1[block1]
                
#             print(i,ft)

            try:
                y2 = yt[block1]
            except IndexError:
                y2 = yt[min(block1):max(yt)]
    
            mask = ~np.isnan(ft)
            slope,intercept,r,p,se = stats.linregress(y2[mask],ft[mask])
            dF = 1-(math.exp(slope))
            try:
                postTb = [x for x in np.array(tyrs)[block1] if x>0]
            except IndexError:
                postTb = [x for x in np.array(tyrs)[min(block1):max(yt)] if x>0]
            meanTb = np.mean(postTb)
            Neb = 1/(2*(dF*meanTb))
            period1[0,i]=dF*meanTb
            period1[1,i]=meanTb
            period1[2,i]=Neb 


        except ValueError:
            pass
            period1[0,i]=-99
            period1[1,i]=-99
            period1[2,i]=-99

    a = pd.DataFrame(period1)
    a.columns = ['{} - {}'.format(x,x+4) for x in list(np.array(uniq_yob)[y1])]
    a.index=["dF_5yr","genInt_5yr","Ne_5yr"]
    a = a.round(4)
    a.to_csv("{}/out_tables/5yearly_dF_Ne_from_{}.csv".format(outfolder,variable))
    
    return


def FUNC07_stem(nps,npd,y1,uniq_yob,outfolder):
    '''
    Function plotting stem plot of parental contributions. Updated in Jul 2023. Doesn't produce the last 
    period (2020+) as in some breeds this resulted in shrinkage of scale so that complete periods could not
    be interpreted correctly. 
    '''
    ## Plotting
    ## Ensure nps and npd are of the same size
    npds = pd.merge(nps,npd,on=['bin'],how='outer',suffixes=['_nps','_npd'])
    npds = npds.fillna(0)
    npds = npds.set_index('bin')
    
    temp = npds.loc[~(npds==0).all(axis=1)].reset_index()
    temp.set_index('bin',inplace=True)
    ## Replace 0s with nans to avoid division by zero error when log will be calculated
    temp.replace(0, np.nan, inplace=True)
    
    ## Find highest values for axis limits, and for xtick labels
    m=np.log(10*math.ceil(0.1*npds.to_numpy().max()))
    xmax = temp.index[-1]
    xtic = [round(x,1) for x in np.arange(0,xmax+(0.05*xmax),xmax/4)]
    
    ## Plot the stem plot
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots(len(y1)-1,2, sharex=True)
    plt.rc('figure', figsize=(10,6))
    sexes=['nps','npd']
    for sex in range(len(sexes)):
        for q in range(len(y1)-1):
            yb = np.array(uniq_yob)[y1][q] 
            markerline, stemlines, baseline = ax[q,sex].stem(temp.index, np.log(temp['{}_{}'.format(yb,sexes[sex])]),use_line_collection=True)
            ## Mask the baseline. 
            # plt.setp(baseline, color='k', linewidth=0.01)
            plt.setp(baseline, linewidth=0.01)
            if sex==0:
                plt.setp(markerline, color='#0000ff', markersize = 3)
                plt.setp(stemlines, color='#0000ff',linewidth = 0.6)
                ax[q,sex].set_ylabel("{}-{}".format(yb,yb+4),rotation=0,fontsize=7)
            else:
                plt.setp(markerline, color='r', markersize = 3)
                plt.setp(stemlines, color='r',linewidth = 0.6)
            ax[q,sex].set_ylim(-0.1,m)
            ## Hide ytick labels without removing grid
            ax[q,sex].tick_params(axis='y', colors=(0,0,0,0))
            
    ax[q,0].set_xlim(0,xmax+0.05*xmax)
    ax[q,0].set_xticks(xtic)
    
    ax[q,1].set_xlim(0,xmax+0.05*xmax)
    ax[q,1].set_xticks(xtic)
            
    # ## Manually change the label of the last entry if incomplete
    # if y1[-1]==y1[-2]+5:
    #     ax[q,0].set_ylabel("{}+".format(yb))        
    ax[q,0].set_xlabel("Progeny as % of all registered dogs")
    ax[q,1].set_xlabel("Progeny as % of all registered dogs")
    plt.savefig("{}/out_plots/Sire&dam_progeny_stem.jpg".format(outfolder), dpi=199)
   
    return

def FUNC07_stem_b(nps,npd,y1,uniq_yob,outfolder):
    ## Plotting
    ## Ensure nps and npd are of the same size
    npds = pd.merge(nps,npd,on=['bin'],how='outer',suffixes=['_nps','_npd'])
    npds = npds.fillna(0)
    npds = npds.set_index('bin')
    
    temp = npds.loc[~(npds==0).all(axis=1)].reset_index()
    temp.set_index('bin',inplace=True)
    m=np.log(10*math.ceil(0.1*temp.to_numpy().max()))
    ## Replace 0s with nans to avoid division by zero error when log will be calculated
    temp.replace(0, np.nan, inplace=True)
    
    ## Find highest values for axis limits, and for xtick labels
    # m=np.log(10*math.ceil(0.1*temp.to_numpy().max()))
    xmax = temp.index[-1]
    xtic = [round(x,1) for x in np.arange(0,xmax+1,xmax/4)]
    
    ## Plot the stem plot
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots(len(y1),2, sharex=True)
    plt.rc('figure', figsize=(10,6))
    sexes=['nps','npd']
    for sex in range(len(sexes)):
        for q in range(len(y1)):
            yb = np.array(uniq_yob)[y1][q] 
            markerline, stemlines, baseline = ax[q,sex].stem(temp.index, np.log(temp['{}_{}'.format(yb,sexes[sex])]),use_line_collection=True)
            ## Mask the baseline. 
            # plt.setp(baseline, color='k', linewidth=0.01)
            plt.setp(baseline, linewidth=0.01)
            if sex==0:
                plt.setp(markerline, color='#0000ff', markersize = 3)
                plt.setp(stemlines, color='#0000ff',linewidth = 0.6)
                ax[q,sex].set_ylabel("{}-{}".format(yb,yb+4),rotation=0,fontsize=7)
            else:
                plt.setp(markerline, color='r', markersize = 3)
                plt.setp(stemlines, color='r',linewidth = 0.6)
            ax[q,sex].set_ylim(-0.1,m)
            ax[q,sex].set_xlim(0,xmax+0.05*xmax)
            ax[q,sex].set_xticks(xtic)
            ## Hide ytick labels without removing grid
            ax[q,sex].tick_params(axis='y', colors=(0,0,0,0))
            
    ## Manually change the label of the last entry if incomplete
    if y1[-1]==y1[-2]+5:
        ax[q,0].set_ylabel("{}+".format(yb))        
    ax[len(y1)-1,0].set_xlabel("Progeny as % of all registered dogs")
    ax[len(y1)-1,1].set_xlabel("Progeny as % of all registered dogs")
    plt.savefig("{}/out_plots/Sire&dam_progeny_stem.jpg".format(outfolder), dpi=199)
   
    return

def reg_volatility(tborn,uniq_yob,yob_counts,outfolder):
    '''
    Function calculating volatility and peaks as defined in Gharlandi et al (2013)
    Also, percentage change in registrations in last complete year as compared against registrations in 2000. 
    
    Dropping years where no dogs were registered.
    '''
    print("\nCalculating registration volatility")
    ## Calculate year by year differences
    d = np.ediff1d(tborn)
    d = [np.abs(x) for x in d]
    w = []
    for i,n in enumerate(d):
        if tborn[i]!=0:
            w.append(np.abs(n/tborn[i]))

    vol = np.nanmean(w)
    g = open("{}/out_tables/registration_volatility.csv".format(outfolder),'w')
    g.write("{},{}\n".format("Volatility",vol))
    
    ## Find peaks - peak defined if it has start and end times, where registrations where <=10% of max
    peak = max(tborn)
    ## Find index of the peak
    pi = np.where(tborn==peak)[0][0]
    ym = uniq_yob[pi]
    prior = tborn[:pi]
    post = tborn[pi:]

    ## Find start of the peak
    st = [x for x in prior if x<=peak/10]
    end = [x for x in post if x<=peak/10]
    ## Find indices of the start and end
    sti = list(np.where(np.in1d(tborn, st))[0])
    endi = list(np.where(np.in1d(tborn,end))[0])
    g.write("{},{}\n".format("Peak in",ym))
    g.write("{},{}\n".format("N at peak",peak))
    
    try:
        ## Find last year before peak starts
        st_st = max(np.take(uniq_yob, sti))
        ## Rate of increase in popularity
        rup = (peak-tborn[np.where(uniq_yob==st_st)[0]])/(ym - st_st)
        g.write("{},{}\n".format("Start of peak",st_st))
        g.write("{},{}\n".format("N at start",tborn[np.where(uniq_yob==st_st)[0]][0]))
        g.write("{},{}\n".format("Rate of increase",rup[0]))
        
    except ValueError:
        print("Couldn't identify start of the peak")
        
    try:        
        ## Find first year after the peak
        end_st = min(np.take(uniq_yob,endi))        
        rdown = (peak-tborn[np.where(uniq_yob==end_st)[0]])/(end_st - ym)
        
        g.write("{},{}\n".format("End of peak",end_st))
        g.write("{},{}\n".format("N at end",tborn[np.where(uniq_yob==end_st)[0]][0]))
        g.write("{},{}\n".format("Rate of decrease",rdown[0]))
        
    except ValueError:
        print("Couldn't identify the end of the peak")
        
    ## Registrations in last complete year in relation to reference point - N dogs registered in 2000
    try:
        ref = yob_counts[1][np.where(yob_counts[0]==2000)][0]
    except IndexError:
        ref = 0
        
    regref = (tborn[-1]/ref)*100
    g.write("{},{}\n".format("Last year",uniq_yob[-1]))
    g.write("N last year,{}".format(tborn[-1]))
    g.write("N in 2000,{}".format(ref))
    g.write("{},{}\n".format("Last year against 2000 (% change)",regref))
        
    g.close()
    print("Output saved in out_tables/registration_volatility.csv")
    return

def FUNC03_reg_trends(tborn,uniq_yob,maxyob,outfolder,variable):
    '''
    Function plotting regression of either number of dogs or number of litters per year of birth. 
    Can be done either for the complete pedigree, or filtered - there are no additional calculations of inbreeding involved. 
    '''
    print("\nCalculating registration trends")
    results = stats.linregress(uniq_yob,tborn)
    a = results[1]
    b = results[0]
    se = results[4]
    cilow = b-(1.96*se)
    cihigh = b+(1.96*se)
    pval = results[3]
    print("Regression of {} on yob = {:3.2f}, with 95% CI of {:3.2f} to {:3.2f}".format(variable,b,cilow,cihigh))
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots()
    
    #c = np.linspace(start=1990,stop=maxyob,num=len(uniq_yob))
    c = np.linspace(start=min(uniq_yob),stop=maxyob,num=len(uniq_yob))
    abline = [b*i + a for i in c]
    
    
    ## Might be better to change to plt.plot, so that I have more freedom with colors, but for now, this messes with anchored text
    plt.plot(uniq_yob,tborn,'o-',mec='#0000ff',mfc='#9999ff',color='#0000ff',linewidth=0.9)
    plt.gca().set_xlim(left=1989)
    plt.xlabel("Year of birth")
    if variable=='tborn':
        plt.ylabel("Number of UK-born dogs")
    else:
        plt.ylabel("Number of UK-born litters")
        
    plt.plot(c,abline,color='#000000',linestyle=':',label='regression (m)', linewidth=1)
    anchored_text = AnchoredText("Year effect = {:3.2f},\n95% CI {:3.2f} to {:3.2f}\np-value = {:3.2f}".format(b,cilow,cihigh,pval), loc=2)
    ax.add_artist(anchored_text)
    plt.savefig("{}/out_plots/trends_{}.jpg".format(outfolder,variable), dpi=199)
    print("Plot saved in out_plots/trends_{}.jpg".format(variable))
    
    return a,b,se,pval

def print_stats(col_array,outfolder=None,ofname=None):
    '''
    Function printing out summmary statistics for a given column. 
    Returns a sample SD (divided by n-1), as opposed to population SD (divided by n, python's default!). 
    '''
    mo = stats.mode(col_array.dropna())[0][0]
    med = np.nanmedian(col_array.dropna())
    mean = np.mean(col_array.dropna())
    std = np.std(col_array.dropna(), ddof=1)
    mn = min(col_array)
    mx = max(col_array)
    
    result = pd.DataFrame(columns=['statistic','value'])
    result.loc[len(result)+1]=['Min',mn]
    result.loc[len(result)+1]=['Max',mx]
    result.loc[len(result)+1]=['Mode',mo]
    result.loc[len(result)+1]=['Median',med]
    result.loc[len(result)+1]=['Mean',mean]
    result.loc[len(result)+1]=['SD',std]

    ## Print to file
    try:
        result.to_csv("{}/out_tables/{}.csv".format(outfolder,ofname), index=False)
    except FileNotFoundError:
        pass
    
    return result


def litter_trends(litters,psire,pdam,dserial,yob,outfolder,ofname):
    '''
    Function producing summary statistics for litter size over time.
    '''
    ## Litter size statistics
    lit_yob = dict(zip(litters,yob))
#     lit_sire = dict(zip(litters,psire))
#     lit_dam = dict(zip(litters,pdam))
    lit_dserial = dict(zip(litters,dserial))
    

    ## Get litter sizes
    l = np.array(np.unique(litters,return_counts=True))
    ldf = pd.DataFrame(l.T,columns=['litter','size'])
    ldf['yob']=ldf['litter'].map(lit_yob)
#     ldf['psire']=ldf['litter'].map(lit_sire)
#     ldf['pdam']=ldf['litter'].map(lit_dam)
    ldf['dserial']=ldf['litter'].map(lit_dserial)
    # ldf = ldf.astype('Int64')

    print("Found {} unique litters".format(len(ldf)))
    print("statistics for litter size")
    print(print_stats(ldf['size'],outfolder,ofname))

    ## Fit regression on yob
    lit_yob_reg = ols('size ~ yob', data=ldf).fit()
    lya,lyb = lit_yob_reg.params
    lyr = print_my_summary(lit_yob_reg)
    
    lyr['p-value']=lyr['p-value'].astype('float').round(3)
    pval = lyr.loc[lyr['parameter']=='yob']['p-value'].values[0]
    se = lyr.loc[lyr['parameter']=='yob']['SE'].values[0]
    lyr[['slope','SE']] = lyr[['slope','SE']].astype('float').round(2)
    
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots()
    #c = np.linspace(start=1990,stop=2022,num=32)
    c = np.linspace(start=min(yob),stop=2022,num=32)
    abline = [lyb*i + lya for i in c]
    ax = sb.lineplot(data=ldf,x='yob',y='size',marker='o',mec='#0000ff',mfc='#9999ff',color='#0000ff')
    plt.gca().set_xlim(left=1989)
    plt.xlabel("Year of birth")
    plt.ylabel("Litter size")
    plt.ylim(1,12)
    plt.plot(c,abline,color='#000000',linestyle=':',label='regression (m)', linewidth=1)
    anchored_text = AnchoredText("Year effect = {:3.2f}, \np-value = {}".format(lyb,pval), loc=3)
    ax.add_artist(anchored_text)
    plt.savefig("{}/out_plots/trends_litter_size_{}.jpg".format(outfolder,ofname), dpi=199)
    
    return ldf,lya,lyb,se,pval


def litt_dist(ldf,outfolder,ofname):
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    ax = sb.histplot(ldf['size'],bins=int(max(ldf['size'])),color='#0000ff',discrete=True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.ylabel("Count of litters")
    plt.xlabel("Litter size")
    plt.xlim(0.1,)
    plt.savefig("{}/out_plots/litter_size_dist_{}.jpg".format(outfolder,ofname), dpi=199)

    return
def COI_litter_b(ldf,litters,F,outfolder,ofname):
    '''
    Function plotting litter size as a function of COI - but after removing records where few litters at 
    given (rounded) COI were reported. This is to remove the most extreme COI values which could affect the regression coefficient. 
    Plot shows all litters.
    '''
    
    ## Add inbreeding to litter stats
    lit_F = dict(zip(litters, F))
    ldf['F']=ldf['litter'].map(lit_F)
    ldf['F'] = ldf['F']*100
    
    ## For plotting, round fvalues, otherwise plot messy
    ldf['F_round']=ldf['F'].round(0)
    
    ## Regression will be fitted only when at least 3 litters with a given F_round were recorded
    a = ldf.groupby(['F_round']).count()
    b = a.loc[a['litter']<3].index
    
    ## Drop the rare F values
    ldf2 = ldf.loc[~ldf['F_round'].isin(b)]  
    if len(ldf2)>0:
        ## Extract the extreme values - will be plotted separately in grey
        ldf3 = ldf.loc[ldf['F_round'].isin(b)]
        print("Regression calculated for COI values for which at least 3 litters were recorded")
        
        print("Dropped {} litters at rare COI values".format(len(ldf)-len(ldf2)))
    
        ## Fit regression line - on the estimates as they are (6 decimals)
        lit_f_reg = ols('size ~ F', data=ldf2).fit()
        lfa,lfb = lit_f_reg.params
        lfr = print_my_summary(lit_f_reg)
        
        lfr['p-value']=lfr['p-value'].astype('float').round(3)
        pval = lfr.loc[lfr['parameter']=='F']['p-value'].values[0]
        se = lfr.loc[lfr['parameter']=='F']['SE'].values[0]
        lfr[['slope','SE']] = lfr[['slope','SE']].astype('float').round(2)
        
        ## Slope shows the change given 1% change in F. 
        ## This is too minute, so calculate change induced by 10% increase in F to be printed on plot
        f10 = lfb*10
        if f10>0:
            fb='+{:2.1f}'.format(f10)
        else:
            fb='{:2.1f}'.format(f10)
    
        ## Plot
        plt.clf()
        plt.rc('figure', figsize=(6.4,4.8))
        fig, ax = plt.subplots()
        c = np.linspace(start=0,stop=max(ldf2['F_round']),num=50)
        abline = [lfb*i + lfa for i in c]
        ax = sb.lineplot(data=ldf2,x='F_round',y='size',marker='o',mec='#0000ff',mfc='#9999ff',color='#0000ff')
    #     ax = sb.lineplot(data=ldf3,x='F_round',y='size',marker='o',mec='#A0A0A0',mfc='#E0E0E0',color='#A0A0A0',linestyle='--',markersize=5,linewidth=0.8)
        ax = sb.lineplot(data=ldf3,x='F_round',y='size',marker='o',mec='#000000',mfc='#E0E0E0',color='#000000',linestyle='--',markersize=5,linewidth=0.8)
        
        #plt.gca().set_xlim(left=1989)
        plt.gca().set_ylim(bottom=0)
        plt.xlabel("Coefficient of Inbreeding (%)")
        plt.ylabel("Litter size")
        plt.plot(c,abline,color='#000000',linestyle=':',label='regression (m)', linewidth=1)
        anchored_text = AnchoredText("10% change in COI = {} puppies, \np-value = {}".format(fb,pval), loc=3)
        ax.add_artist(anchored_text)
        plt.savefig("{}/out_plots/COI_litter_size_{}.jpg".format(outfolder,ofname), dpi=199)
        
        return ldf, lfa,lfb,se,pval
    
    else:
        print("Not enough data to calculate the regression, no plot produced.")
        
        return ldf, 'NA','NA','NA','NA'
    
    
    

def Fcat_litter(ldf,outfolder,ofname):
    '''
    Function returning box plot of litter sizes per COI category
    '''
    ## Numbers of litters in categories of F values
    ldf['Fcat'] = pd.cut(ldf['F'],bins=[-0.01,5,15,25,35,100],labels=['0 - 5%','6 - 15%','16 - 25%','26 - 35%','>35%'])
    ## Boxplot - the box is marked by 25th to 75th percentile, with line in the middle showing MEDIAN!!. Whiskers extend to show range. Star marks MEAN.
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    sb.boxplot(x='Fcat',y='size',data=ldf,showmeans=True,meanprops={"marker":"*",
                           "markerfacecolor":"#404040", 
                           "markeredgecolor":"#404040",
                          "markersize":"5"})
    plt.ylim([0,max(ldf['size'])+2])
    plt.xlabel("COI category")
    plt.ylabel("Litter size")
    plt.savefig("{}/out_plots/COI_litterSize_box_{}.jpg".format(outfolder,ofname),dpi=199)
    
    return ldf

def COI_split(ldf,imps,psire,pdam,litters,label1,label2,outfolder):
    '''
    Function plotting litter COI based on a provided category
    '''
    ## Inbreeding per litter, with and without litters after imported parents
    ## Find litters where at least one parent is an import
    impsarr = np.array(imps['pid'])
    ## Find indices of sires if sires are imports
    sire_impx = np.where(np.in1d(psire,impsarr))
    dam_impx = np.where(np.in1d(pdam,impsarr))

    ## Either sire or dam - still indices!
    pimpx = np.union1d(sire_impx,dam_impx)
    #print("Found {} dogs after at least one imported parent".format(len(pimpx)))

    ## Find the litter data
    imp_lit = litters[pimpx]

    if len(imp_lit)>0:
        ## Plotting
        plt.clf()
        fig, ax = plt.subplots()

        ldf_ni = ldf.loc[~ldf['litter'].isin(imp_lit)]
        ldf_i = ldf.loc[ldf['litter'].isin(imp_lit)]
        sb.lineplot(data=ldf_i,x='yob',y='F_round',label='{} litters'.format(label1),marker='o',mec='#0000ff',mfc='#9999ff',color='#0000ff')
        sb.lineplot(data=ldf_ni, x='yob', y='F_round', label='{} litters'.format(label2),marker='o',mec='#cc3399',mfc='#ebadd6',color='#cc3399')
    #     sb.lineplot(data=ldf,x='yob',y='F_round',label='All litters',marker='o',mec='#000000',mfc='#E0E0E0',color='#000000',linestyle='--',markersize=5,linewidth=0.8)
        sb.lineplot(data=ldf,x='yob',y='F_round',label='All litters',color='#505050',linestyle='--',markersize=5,linewidth=0.8)

        plt.gca().set_xlim(left=1989)
        plt.gca().set_ylim(bottom=0)
        plt.xlabel("Year of birth")
        plt.ylabel("Litter's COI")

        plt.savefig("{}/out_plots/COI_split_{}_{}.jpg".format(outfolder,label1,label2), dpi=199)
    else:
        print("No plot produced for COI split between {} and {}, as there were no {} litters found".format(label1,label2,label1))
    
    return imp_lit

def prop_imp_champ(ldf,impBred_lit,purpBred_lit,outfolder,ofname):
    ldf['PurposeBred']='pet'
    ldf.loc[ldf['litter'].isin(purpBred_lit),'PurposeBred']='PurposeBred'
    ldf['ImportBred']='National'
    ldf.loc[ldf['litter'].isin(impBred_lit),'ImportBred']='ImportBred'
    # Calculate proportions of import bred
    impprop = pd.crosstab(ldf['yob'],ldf['ImportBred'],normalize='index')
    impprop = impprop*100
    impprop.reset_index(inplace=True)

    ## Calculate the proportions
    champprop = pd.crosstab(ldf['yob'],ldf['PurposeBred'],normalize='index')
    champprop = champprop*100
    champprop.reset_index(inplace=True)

    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    q1 = 0
    q2 = 0
    try:
        plt.plot(champprop['yob'],champprop['PurposeBred'],label='PurposeBred',marker='o',mec='#0000ff',mfc='#9999ff',color='#0000ff')
    except KeyError:
        print("No champion litters found")
        q1 = 1
    try:
        plt.plot(impprop['yob'],impprop['ImportBred'],label='ImportBred',marker='o',mec='#cc3399',mfc='#ebadd6',color='#cc3399')
    except KeyError:
        plt.plot("No import litters found")
        q2 = 1
        
    if q1 == q2 == 1:
        print("No point in producing plot, as no champion or import litters found")
    else:    
        plt.ylim([0,100])
        plt.xlim(1989,)
        plt.legend()
        plt.xlabel("Year of birth")
        plt.ylabel("Percentage of litters (%)")

        plt.savefig("{}/out_plots/trends_prop_litters_show_imp_{}.jpg".format(outfolder,ofname), dpi=199)

        ## Merge and save to file
        prop = pd.merge(impprop,champprop,on='yob')
        prop.to_csv("{}/out_tables/trends_prop_litters_show_imp_{}.csv".format(outfolder,ofname), index=False)

    return ldf

def prop_born_bred(uniq_yob,sex,pid,psire,pdam,yob,outfolder,ofname):
    yms = np.zeros(len(uniq_yob[:-5]))
    yfd = np.zeros(len(uniq_yob[:-5]))
    males = np.where(sex=='Dog')
    females = np.where(sex=='Bitch')
    sirex = np.where(np.in1d(pid,psire))
    damx = np.where(np.in1d(pid,pdam))
    
    for y in range(len(uniq_yob[:-5])):
        ## All dogs born that year
        born = np.where(yob==uniq_yob[y])
        ## Males born in that year
        bornm = np.intersect1d(born,males)
        ## Males which became sires
        borns = np.intersect1d(bornm,sirex)
        ## Proportion
        try:
            yms[y]=len(borns)/len(bornm)*100
        except ZeroDivisionError:
            yms[y] = np.NaN
    
    
        ## Repeat for dams
        bornf = np.intersect1d(born,females)
        bornd = np.intersect1d(bornf,damx)
        try:
            yfd[y]=len(bornd)/len(bornf)*100
        except ZeroDivisionError:
            yfd[y] = np.NaN
    
    ## Fit regression lines
    maskm = ~np.isnan(yms)
    uy = np.array(uniq_yob[:-5])
    try:
        pM_reg = stats.linregress(uy[maskm],yms[maskm])
        pM_a = pM_reg[1]
        pM_b = pM_reg[0]
        pM_p = pM_reg[3]
        pM_se = pM_reg[4]
    except ValueError:
        pM_a=np.NaN
        pM_b=np.NaN
        pM_p=np.NaN
        pM_se = np.NaN
    
    maskf = ~np.isnan(yfd)
    try:
        pF_reg = stats.linregress(uy[maskf],yfd[maskf])
        pF_a = pF_reg[1]
        pF_b = pF_reg[0]
        pF_p = pF_reg[3]
        pF_se = pF_reg[4]
    except ValueError:
        pF_a=np.NaN
        pF_b=np.NaN
        pF_p=np.NaN
        pF_se = np.NaN
    
    
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots()
    ## Calculate regression lines
    # c = np.linspace(start=1990,stop=uniq_yob[-5],num=50)
    c = np.linspace(start=min(uniq_yob),stop=uniq_yob[-5],num=50)
    mabline = [pM_b*i + pM_a for i in c]
    fabline = [pF_b*i + pF_a for i in c]
    
    plt.plot(uniq_yob[:-5],yms,marker='o',mec='#0000ff',mfc='#9999ff',color='#0000ff',label='Dogs')
    plt.plot(uniq_yob[:-5],yfd,marker='o',mec='#cc3399',mfc='#ebadd6',color='#cc3399',label='Bitches')
    ## Add regression lines
    plt.plot(c,mabline,color='#0000ff',linestyle=':',linewidth = 0.7)
    plt.plot(c,fabline,color='#cc3399',linestyle=':',linewidth = 0.7)
    
    anchored_text = AnchoredText("Year effect for dogs = {:3.2f} (p-value {:3.2f})\nYear effect for bitches = {:3.2f} (p-value {:3.2f})".format(pM_b,pM_p, pF_b, pF_p),loc=2)
    ax.add_artist(anchored_text)
    
    plt.xlim(1989,2021)
    
    try:
        if max(max(yms[~np.isnan(yms)]),max(yfd[~np.isnan(yfd)],))<70:
            plt.ylim([0,75])
        else:
            plt.ylim([0,100])
            
        plt.xlabel("Year of birth")
        plt.ylabel("Percentage of dogs born used in breeding (%)")
        plt.legend()
        plt.savefig("{}/out_plots/trends_prop_breeding_dogs_{}.jpg".format(outfolder,ofname), dpi=199)
    except ValueError:
        pass
           

    
    ## Save to file
    bb = pd.DataFrame(columns=['yob','propM','propF'])
    bb['yob']=uniq_yob[:-5]
    bb['propM']=yms
    bb['propF']=yfd
#     bb.to_csv("{}/out_tables/trends_prop_breeding_dogs_{}.csv".format(outfolder,ofname), index=False)
    return pM_a, pM_b, pM_se, pM_p, pF_a, pF_b, pF_se, pF_p



## For national litters, find the parents age, even if parents outside of the national group
def p_age(ldf,parent,pida,dseriala):
    '''
    Function calculating the age of the parent at birth of litter. 
    '''
    ds = np.array(ldf['dserial'])
    par = np.array(ldf[parent])
    pagearr=np.zeros(len(ldf))
    
    
    for l in range(len(ldf)):
        ## Find the index of the parent in the original pid array
        px = np.where(pida==par[l])
        if len(px[0])>0:
            page = dseriala[px]
            if page[0]!=693962:
                age = (ds[l] - dseriala[px])/30
                if age>0:
                    pagearr[l] = (ds[l] - dseriala[px])/30 
                else:
                    pagearr[l] = np.NaN
            else:
                pagearr[l]=np.NaN
        else:
            pagearr[l]=np.NaN
    return pagearr

def get_page(df,ldf,litters,psire,pdam,pida,dseriala,outfolder):
    '''
    Function adding parent age columns to ldf. 
    '''
    lit_ps = dict(zip(litters, psire))
    lit_pd = dict(zip(litters,pdam))
    ldf['ps']=ldf['litter'].map(lit_ps)
    ldf['pd']=ldf['litter'].map(lit_pd)
    
    ldf['sireage']=p_age(ldf,'ps',pida,dseriala)
    ldf['damage']=p_age(ldf,'pd',pida,dseriala)

    
    ldf.to_csv("{}/out_tables/litter_stats.csv".format(outfolder),index=False)
    
    return ldf

def plot_page(ldf2,imps,sdb,parent,outfolder):
    '''
    Modified in Jul 2023
    Rescaled the xaxis to start with 0, before relabeling the ticks to years. 
    '''
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True)
    if parent=='sire':
        p='ps'
    else:
        p='pd'
    
    ## Histogram of parent's age at birth of litter
    ## Lineplot for purpose-bred dogs
    page = np.array(ldf2.loc[ldf2[p].isin(sdb['pid'])]['{}age'.format(parent)])
    print("Found {} litters whose {} had a studbook No".format(len(page),parent))
    page = page[~np.isnan(page)]
    page = np.rint(page)
    sb.histplot(x=page,color='#0000ff', alpha=0.4,label='Show {}s'.format(parent),ax=ax1)
    
    ## Plot for imports
    page = np.array(ldf2.loc[ldf2[p].isin(imps['pid'])]['{}age'.format(parent)])
    print("Found {} litters whose {} was an import".format(len(page),parent))
    page = page[~np.isnan(page)]
    page = np.rint(page)
    sb.histplot(x=page,color='#cc3399', alpha=0.4,label='Imported {}s'.format(parent),ax=ax2)
    
    ## Other litters - non-purpose and after national dogs
    ot = ldf2.loc[(~ldf2[p].isin(sdb['pid'])) & (~ldf2[p].isin(imps['pid']))]    
    print("Found {} litters whose {} was neither an import, nor had a studbook No".format(len(ot),parent))
    sb.histplot(x=ot['{}age'.format(parent)], color='k',alpha=1,ax=ax3)
    
    ## Set axis labels to show years
    ax2.set_xlim(0,)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(12))
    labels = [int(x/12) for x in ax2.get_xticks().tolist()]
    ax2.set_xticklabels(labels)
    ax2.set_xlim([-1,144])
#     plt.legend()

#     ax2.tick_params(axis='y',color='r')
    ax3.set_ylabel("Other litters")
    ax1.set_ylabel("StudBook# {}".format(parent))
    ax2.set_ylabel("Import {}".format(parent))
    ax3.set_xlabel("{} age (y) at birth of litter".format(parent))
    
    plt.savefig("{}/out_plots/age_at_litter_{}.jpg".format(outfolder,parent),dpi=199)
    
    return

def plot_page_b(ldf2,imps,label,parent,outfolder):
    '''
    Modified in Jul 2023
    Rescaled the xaxis to start with 0, before relabeling the ticks to years. 
    For breeds where imps or studs are empty
    
    '''
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, (ax1,ax2) = plt.subplots(2, sharex=True)
    if parent=='sire':
        p='ps'
    else:
        p='pd'
    
    ## Histogram of parent's age at birth of litter    
    ## Plot for imports
    page = np.array(ldf2.loc[ldf2[p].isin(imps['pid'])]['{}age'.format(parent)])
    print("Found {} litters whose {} was in {}".format(len(page),parent,label))
    page = page[~np.isnan(page)]
    page = np.rint(page)
    sb.histplot(x=page,color='#cc3399', alpha=0.4,label='Imported {}s'.format(parent),ax=ax1)
    
    ## Other litters - non-purpose and after national dogs
    ot = ldf2.loc[~ldf2[p].isin(imps['pid'])]    
    print("Found {} litters whose {} was neither an import, nor had a studbook No".format(len(ot),parent))
    sb.histplot(x=ot['{}age'.format(parent)], color='k',alpha=1,ax=ax2)
    
    ## Set axis labels to show years
    ax2.set_xlim(0,)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(12))
    labels = [int(x/12) for x in ax2.get_xticks().tolist()]
    ax2.set_xticklabels(labels)
    ax2.set_xlim([-1,144])
#     plt.legend()

#     ax2.tick_params(axis='y',color='r')
    ax2.set_ylabel("Other litters")
    ax1.set_ylabel("{} {}".format(label,parent))
    ax2.set_xlabel("{} age (y) at birth of litter".format(parent))
    
    plt.savefig("{}/out_plots/age_at_litter_{}.jpg".format(outfolder,parent),dpi=199)
    
    return


def plot_page_c(ldf2,parent,outfolder):
    '''
    Modified in Jul 2023
    Rescaled the xaxis to start with 0, before relabeling the ticks to years. 
    For breeds where both imps and studs are empty
    
    '''
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax1 = plt.subplots()
    if parent=='sire':
        p='ps'
    else:
        p='pd'
    
    ## Histogram of parent's age at birth of litter        
    ## Other litters - non-purpose and after national dogs  
    print("Found {} litters whose {} was neither an import, nor had a studbook No".format(len(ldf2),parent))
    sb.histplot(x=ldf2['{}age'.format(parent)], color='k',alpha=1,ax=ax1)
    
    ## Set axis labels to show years
    ax1.set_xlim(0,)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(12))
    labels = [int(x/12) for x in ax1.get_xticks().tolist()]
    ax1.set_xticklabels(labels)
    ax1.set_xlim([-1,144])
#     plt.legend()

#     ax2.tick_params(axis='y',color='r')
    ax1.set_ylabel("All litters")
    ax1.set_ylabel("All litter per {}".format(parent))
    ax1.set_xlabel("{} age (y) at birth of litter".format(parent))
    
    plt.savefig("{}/out_plots/age_at_litter_{}.jpg".format(outfolder,parent),dpi=199)
    
    return



def sire_stats(ldf2,imps,sdb,outfolder,df,psire,top):
    print("\nCalculating sire statistics")
    par_litt_count = pd.DataFrame(np.unique(ldf2['ps'],return_counts=True))
    par_litt_count = par_litt_count.T
    par_litt_count.columns=['parent','n_litters']
    par_litt_count = par_litt_count.loc[par_litt_count['parent']!=0].reset_index(drop=True)
    par_litt_count['nlit_cat'] = pd.cut(par_litt_count['n_litters'],bins=[-0.01,5,15,25,35,100],labels=['1 - 5','6 - 15','16 - 25','26 - 35','>35'])
    ## Histogram of litter numbers per parent
    # sb.histplot(x=par_litt_count['n_litters'], color='#0000ff',bins=15,edgecolor="black")
    ## Add columns checking whether a parent is a show dog, or import
    par_litt_count['imp'] = np.isin(par_litt_count['parent'],imps['pid'])
    par_litt_count['purpBred'] = np.isin(par_litt_count['parent'],sdb['pid'])
    
    ## Plot histogram of litters per sire
    x = par_litt_count.groupby('nlit_cat').count().sort_values(by=['n_litters'],ascending=False).reset_index()
    
    plt.clf()
    sb.barplot(x='nlit_cat',y='n_litters',data=x,color='#0000ff',edgecolor='#0000ff')
    plt.xlabel("Litters per sire")
    plt.ylabel("Count of sires")
    plt.savefig("{}/out_plots/litters_per_sire_cat.jpg".format(outfolder), dpi=199)
    print("Number of litters per sire in categories saved in out_plots/litters_per_sire_cat.jpg")
    
    ## Update Jul 2023 - recreate the sire_nlit_cat_counts
    x = x[['nlit_cat','n_litters']]
    ## Add count of sires with >100 litters
    x.loc[len(x)+1]=['>100',len(par_litt_count.loc[par_litt_count['n_litters']>100])]
    
    x['freq']=x['n_litters']/sum(x['n_litters'])*100
    
    
    x.set_index('nlit_cat', inplace=True)
    x=x.reindex(['1 - 5','6 - 15','16 - 25','26 - 35','>35','>100'])
    x.to_csv("{}/out_tables/sire_nlit_cat_counts.csv".format(outfolder))
    
    
    sires = pd.merge(df[['KCreg','DogName','yob','pid']],par_litt_count,right_on=['parent'],left_on=['pid'],how='inner')
    sires.drop(columns=['parent'],inplace=True)
    par_pup_count = pd.DataFrame(np.unique(psire,return_counts=True))
    par_pup_count = par_pup_count.T
    par_pup_count.columns = ['pid','n_pups']
    temp = pd.merge(sires,par_pup_count,on=['pid'],how='inner')
    
    ## Calculate age at first litter for the sire
    ldf3 = ldf2.sort_values(by=['ps','sireage']).drop_duplicates(subset=['ps'],keep='first')[['ps','sireage']]
    ## Merge
    sires = pd.merge(temp,ldf3,left_on=['pid'],right_on=['ps'],how='inner')
    sires.drop(columns=['ps'],inplace=True)
    sires.rename(columns={"sireage": "age1stLitter"},inplace=True) 
    
#     sires = temp
    ## Save top 10 producing sires - number can be modified by changing top parameter
    sires.sort_values(by=['n_pups'], ascending=False).head(top).to_csv("{}/out_tables/top{}_sires.csv".format(outfolder,top), index=False)
    sires.loc[sires['yob']>2014].sort_values(by=['n_pups'], ascending=False).head(top).to_csv("{}/out_tables/top{}_sires_since2015.csv".format(outfolder,top), index=False)
    print("Top {} sires saved in out_tables/top{}_sires.csv and top{}_sires_since2015.csv".format(top,top,top))
    
    return sires

def ttest2(x,y,thresh):
    '''
    Function recreating the ttest2 from Matlab - Python does not produce CI as default in t-test.
    Assumes unequal variances.
    '''
    
    ## Run the t-test
    t,p=stats.ttest_ind(x,y,equal_var = False)
    
    ## Calculate the  95% confidence interval for the true difference in means, assuming unequal variances
    cm = sms.CompareMeans(sms.DescrStatsW(x), sms.DescrStatsW(y))
    ci_low,ci_high = cm.tconfint_diff(usevar='unequal')

    ## Print out results:
    if p>=thresh:
        print("The p-value of the t-test is > {}, therefore accept the null hypothesis".format(thresh))
    elif np.isnan(p)==True:
        print("The p-value couldn't be calculated")
    else:
        print("The t-test returned a significant result with p={:4.3f}, accept the alternative hypothesis".format(p))
    
#     print()
#     print("t = {:3.2f}".format(t))
#     print("p = {:4.3f}".format(p))
#     print("CI low: {:5.4f}".format(ci_low))
#     print("CI high: {:5.4f}".format(ci_high))
    
    
    return p

def sire_summary_stats(temp, sire_list,var):
    '''
    Function producing stats per sire in a given list. Need to specify whether stats should be per litter or per npups.
    '''
    ## Select the data for the provided list of sires
    try:
        sdf = temp.loc[temp['pid'].isin(sire_list)]
        mo = stats.mode(sdf[var])[0][0]
        med = np.nanmedian(sdf[var])
        mean = np.mean(sdf[var])
        std = np.std(sdf[var],ddof=1)
        mn = min(sdf[var])
        mx = max(sdf[var])    
        return mo,med,mean,std,mn,mx
    except IndexError:
        return np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN

def get_sstats(temp,sires,variable,cat):
    mo,med,mean,std,mn,mx = sire_summary_stats(temp, sires,variable)
    row = [cat,variable,len(sires),mo,med,mean,std,mn,mx]
    return row

def add_srow(ss,sires,selected,cat):
    for variable in ['n_litters','n_pups','age1stLitter']:
        try:
            ss.loc[len(ss)+1]=get_sstats(sires,selected['pid'],variable,cat)
        except IndexError:
            ss.loc[len(ss)+1]=[cat,variable,0,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]
        
    return ss

def sire_tab(sires,outfolder):
    print("\nStatistical testing of difference in number of litters/puppies between sires in different categories")
    ## Create a dataframe to store the results
    ss = pd.DataFrame(columns=['cat','variable','n_sires','mode','median','mean','std','min','max'])
    
    ## All sires
    all_sires = sires.copy()
    ss = add_srow(ss,sires,all_sires,'all')
    
    ## Purpose bred - all
    pB_all = sires.loc[sires['purpBred']==True]
    ss = add_srow(ss,sires,pB_all,'purpBred')
    
    ## Non purpose bred - all
    npB_all = sires.loc[sires['purpBred']==False]
    ss = add_srow(ss,sires,npB_all,'Non-purpBred')
    
    ## Purpose bred - national
    nimp_pB = sires.loc[(sires['purpBred']==True) & (sires['imp']==False)]
    ss = add_srow(ss,sires,nimp_pB,'NonImp purpBred')
    
    ## Import purposeBred
    imp_pB = sires.loc[(sires['purpBred']==True) & (sires['imp']==True)]
    ss = add_srow(ss,sires,imp_pB,'Imp purpBred')
    
    ## Imports not purposeBred
    imp_npB = sires.loc[(sires['purpBred']==False) & (sires['imp']==True)]
    ss = add_srow(ss,sires,imp_npB,'Imp non purpBred')
    
    ## Imports
    imp_sires = sires.loc[sires['imp']==True]
    ss = add_srow(ss,sires,imp_sires,'Imp')
    
    ## Non-Imports
    nonimp = sires.loc[sires['imp']==False]
    ss = add_srow(ss,sires,nonimp,'Non-imp')
    
    ## Sort
    # ss.sort_values(by=['variable'],inplace=True)
    ss.to_csv("{}/out_tables/sireCat_stats.csv".format(outfolder),index=False)
    
    ### Testing for significance of the difference
    print("Testing whether there is a significant difference in number of litters between:")
    print("Dogs with StudBook No and without:")
    g = open("{}/out_tables/sireCat_pvals.csv".format(outfolder),'w')
    p=ttest2(pB_all['n_litters'],npB_all['n_litters'],0.05)
    g.write("Dogs with StudBook No and without,{}\n".format(p))
    
    print("Imports and national dogs:")
    p=ttest2(imp_sires['n_litters'],nonimp['n_litters'],0.05)
    g.write("Imports and national dogs,{}\n".format(p))
    
    print("Imports with Studbook No and without:")
    p=ttest2(imp_pB['n_litters'],imp_npB['n_litters'],0.05)
    g.write("Imports with Studbook No and without,{}\n".format(p))
    
    print("Imported and national dogs with Studbook No:")
    p=ttest2(imp_pB['n_litters'],nimp_pB['n_litters'],0.05)
    g.write("Imported and national dogs with Studbook No,{}\n".format(p))
    g.close()
    
    
    return ss
            
def sdb_before_after(sdb,ldf2,outfolder):
    print("\nComparing productivity of sires before and after obtaining StudBook No")
    if len(sdb)>0:
        ### Sires only!!
        ## Add column with StudBookNo Gen Date converted to ordinal
        sdb2 = sdb.loc[~sdb['StudBookNoGenDate'].isna()].reset_index(drop=True)
        sdb2['StudBookNoGenDate'] = pd.to_datetime(sdb2['StudBookNoGenDate'], dayfirst=True, errors='coerce')
        sdb2['sdb_dserial']=sdb2['StudBookNoGenDate'].apply(lambda x: x.toordinal() + 366)
        print("Found {} dogs with known StudBookNo Generation Date".format(len(sdb2)))

        ## Extract litters whose sire had sdb
        sdbs_litt = ldf2.loc[ldf2['ps'].isin(sdb2['pid'])]
        ## Add column for sire dob
        stemp = pd.merge(sdbs_litt,sdb2[['pid','sdb_dserial']],left_on='ps',right_on='pid',suffixes=['','_sire']) 
        sdbs_litt = stemp.drop(columns=['pid'])
        print("Calculating statistics for {} national litters born since 1990, with sire having a known SBNo Generation Date".format(len(sdbs_litt)))

        sdbs_litt['SB_date_compare'] = sdbs_litt['dserial']-sdbs_litt['sdb_dserial']
        ## Divide by 30 to get months
        sdbs_litt['month_diff']=sdbs_litt['SB_date_compare']/30

        ## Counting positive and negative date comparisons - column 0 marks litters prior to SDBNo Generation date
        # sdbs_litt.groupby('ps').SB_date_compare.apply(lambda x: pd.Series([(x < 0).sum(), (x > 0).sum()])).unstack()
        sdbs_litt['year_diff']=sdbs_litt['month_diff']/12
        plt.clf()
        plt.rc('figure', figsize=(6.4,4.8))
        fig,ax = plt.subplots()
        sb.histplot(sdbs_litt['year_diff'],color='#0000ff',ax=ax)
        plt.axvline(x=0,color='k')
        plt.xlabel("Years between birth of litter and award of sire's StudBook No.")
        plt.ylabel("Count of litters")
        plt.savefig("{}/out_plots/litterBeforeAfter_StudBook.jpg".format(outfolder),dpi=199)
        print("Plot saved in out_plots/littersBeforeAfter_StudBook.jpg")
    else:
        print("No dogs with StudBookNo found")
    
    return      
## Test that there is only one type of entry per litter
def CS_stats(df,ldf,outfolder):
    print("\nCalculating statistics for Csections, AI and multi-sire litters")
    cs = df[['litter','CS','CS_type','AI','Multi_sire']].drop_duplicates()
    c = cs.groupby('CS').count()['litter'].reset_index()
    c.columns=['Outcome','C-Section']
    ## Multi-sired litters
    ms = cs.groupby('Multi_sire').count()['litter'].reset_index()
    ms.columns=['Outcome','Multi-sired']
    cms = pd.merge(c,ms,on=['Outcome'],how='outer')
    ## AIs
    ai = cs.groupby('AI').count()['litter'].reset_index()
    ai.columns=['Outcome','AI']
    acms = pd.merge(cms,ai,on='Outcome',how='outer')
    
    cst = cs.groupby('CS_type').count()['litter'].reset_index()
    cst.columns=['Outcome','C-Section']
    # cst['Multi_sired'] = [np.NaN for x in range(len(cst))]
    # cst['AI']=[np.NaN for x in range(len(cst))]
    
    # cst.columns=['Outcome','AI','C-Section','Multi-sired']
    res = pd.concat([acms,cst])

    res.to_csv("{}/out_tables/Csection_multi_sire_counts.csv".format(outfolder),index=False)

    if len(cs)==len(ldf):
        temp = pd.merge(ldf,cs,on='litter')
        temp.to_csv("{}/out_tables/litter_stats.csv".format(outfolder),index=False)
        print("Litter statistics updated in litter_stats.csv")
        return res,temp
    else:
        print("More than one Csection or multi-sire record per litter. If needed, check the values. \nLitter_stats.csv was not updated.")
        return res,ldf

def plot_regr(df,variable,outfolder):
    '''
    Function plotting trends of variable over time, with regression line fitted.
    As input requires a dataframe with variable count per yob.
    
    Note - regression calculated between year limits!
    '''
    fig,ax = plt.subplots()
    plt.rc('figure', figsize=(6.4,4.8))
    plt.plot(df['yob'],df[variable],'o-',mec='#0000ff',mfc='#9999ff',color='#0000ff',linewidth=0.9)
    plt.ylim(0,max(df[variable])+1.5)
    plt.xlabel("Year of birth")
    plt.ylabel("Count of {}".format(variable))
    results = stats.linregress(df['yob'],df[variable])
    a = results[1]
    b = results[0]
    pval = results[3]
    se =results[4]
    
    c = np.linspace(start=min(df['yob']),stop=max(df['yob'])+1,num=max(df['yob'])-min(df['yob']))
    abline = [b*i + a for i in c]
    plt.plot(c,abline,color='#000000',linestyle=':',label='regression (m)', linewidth=1)
    anchored_text = AnchoredText("Year effect = {:3.2f},\np-value = {:3.2f}".format(b,pval), loc=2)
    ax.add_artist(anchored_text)
    plt.savefig("{}/out_plots/trends_{}.jpg".format(outfolder,variable), dpi=199)
    
    return a,b,se,pval   
    
def trends_CS_AI(ldf4,outfolder):
    '''
    Function plotting the overall number of Csections over time, with regression line fitted.
    '''
    
    ## Plot number of Csections by year
    if len(ldf4.loc[ldf4['CS']=="Yes"])>10:
        print("\nPlotting trends in Csections: at least 10 records found and records per litter are unique")
        cy = ldf4.loc[ldf4['CS']=='Yes'].groupby(['yob']).count()['litter'].reset_index()
        cy.columns=['yob','Csection litters']
        aCS,bCS,seCS,pvalCS = plot_regr(cy,'Csection litters',outfolder)
        print("Trends in Csections saved in out_plots/trends_CSection litters.jpg")
    
    ## See if there areat least 10 AI records
    if len(ldf4.loc[ldf4['AI']=='Yes'])>10:
        print("\nPlotting trends in AI: at least 10 records found and records per litter are unique")
        adf = ldf4.loc[ldf4['AI']=='Yes'].groupby(['yob']).count()['litter'].reset_index()
        # adf = ldf4.groupby('AI').count()['litter'].reset_index()
        adf.columns = ['yob','AI litters']
        aAI,bAI,seAI,pvalAI = plot_regr(adf,'AI litters',outfolder)
        print("Trends in AI saved in out_plots/trends_AI litters.jpg")
    
    try:
        return aCS,bCS,seCS,pvalCS,aAI,bAI,seAI,pvalAI
    except UnboundLocalError:
        try:
            return aCS,bCS,seCS,pvalCS,np.NaN,np.NaN,np.NaN,np.NaN
        except UnboundLocalError:
            try:
                return np.NaN,np.NaN,np.NaN,np.NaN,aAI,bAI,seAI,pvalAI
            except UnboundLocalError:
                return np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN
    
def plot_Rel(dfr,outfolder):
    print("\nPlotting mean relationships")
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    sb.histplot(x=dfr['R'],color='#0000ff',bins=30)
    plt.xlabel("Mean relationship (%)")
    plt.ylabel("Count of dogs in active set")
    plt.savefig("{}/out_plots/relationships_hist.jpg".format(outfolder),dpi=199)
    print("Plot saved in out_plots/relationships_hist.jpg")
    return

def trend_R(dfr,outfolder):
    print("\nPlotting trends in mean relationship of individuals to the current cohort by yob")
    ## Plot the trend in mean relationships 
    plt.clf()
    plt.rc('figure', figsize=(6.4,4.8))
    fig, ax = plt.subplots()
    ax = sb.lineplot(data=dfr, x='yob',y='R',marker='o',mec='#0000ff',mfc='#9999ff',color='#0000ff')
    plt.xlabel("Year of birth")
    #plt.xlim(1989,)
    plt.ylabel("Mean relationship (%)")

    rel_reg = ols('R ~ yob', data=dfr).fit()
    rra,rrb = rel_reg.params
    rr = print_my_summary(rel_reg)
    rr[['slope','SE']] = rr[['slope','SE']].astype('float').round(2)
    rr['p-value']=rr['p-value'].astype('float').round(3)

    c = np.linspace(start=min(dfr['yob']),stop=max(dfr['yob'])+1)
    abline = [rrb*i + rra for i in c]
    plt.plot(c,abline,color='#000000',linestyle=':',label='regression (m)', linewidth=1)
    anchored_text = AnchoredText("Year effect = {:3.2f}, \np-value = {}".format(rrb,rr.loc[rr['parameter']=='yob']['p-value'].values[0]), loc=3)
    ax.add_artist(anchored_text)
    plt.savefig("{}/out_plots/relationships_trends.jpg".format(outfolder), dpi=199)
    print("Plot saved in out_plots/relationships_trends.jpg")
    
    return

