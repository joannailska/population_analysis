#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:44:58 2022

Scripts which reads in the outputs of JAW suite, and prepares pop analysis stats. 

Active list - bitches born in last 8 years, males born in last 10 years and in Breed Registry.

Update July 2023 - converting the script to new version of JAW suite

Note - the script produces proportion born bred - this applies to national dogs only. 
This is no longer used in the reports, as interpretation is difficult given variable trends in popularity. 
Instead, males_sires_2015.py calculates proportion of national males born in 2015 that were used in breeding,

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
        print("\n*********************************************************************")
        print("Processing breed: ",breed)
        
        ## Read the input file
        try:
            if os.path.splitext(filename)[1]=='.xlsx':
                ped = pd.read_excel(filename,encoding='ISO-8859-1')
                ped = ped.drop(columns=['(Do Not Modify) Dog', '(Do Not Modify) Row Checksum','(Do Not Modify) Modified On'])
                ped.columns = ['Breed', 'Register', 'RegType','OrigCountry', 'ChampTitle', 'KennelPrefix', 'StudBookNo','StudBookNoGenDate', 'IRN','KCreg', 'DogName', 'sex', 'DOB','IRN_sire','KCreg_sire','DogName_sire','IRN_dam','KCreg_dam','DogName_dam','CS','CS_type','AI','Multi_sire']
            else:
                ped = pd.read_csv(filename,encoding='ISO-8859-1')
                try:
                    ped.columns = ['Breed','Register', 'RegType','OrigCountry', 'ChampTitle', 'KennelPrefix', 'StudBookNo','StudBookNoGenDate', 'IRN','KCreg', 'DogName', 'sex', 'DOB','IRN_sire','KCreg_sire','DogName_sire','IRN_dam','KCreg_dam','DogName_dam','CS','CS_type','AI','Multi_sire']
                except ValueError:
                    ped.columns = ['DogGuid','Register', 'RegType','OrigCountry', 'ChampTitle', 'KennelPrefix', 'StudBookNo','StudBookNoGenDate', 'IRN','KCreg', 'DogName', 'sex', 'Breed','DOB','IRN_sire','KCreg_sire','DogName_sire','IRN_dam','KCreg_dam','DogName_dam','CS','CS_type','AI','Multi_sire']
        except FileNotFoundError:
            print("Check file names - no such file was found.")
        
        if len(ped)>0:
            print("Read in file: ", filename)
            ## output path
            outfolder = os.path.dirname(os.path.abspath(filename))
            ## Read in recoded pedigree
            jjiv_f = "{}/JAW/{}_jjiv7_ped.csv".format(outfolder,breed)
            wml_f = "{}/JAW/{}_wmlv5_pedf.csv".format(outfolder,breed)
            coll_f = "{}/JAW/{}_colle_pedc.csv".format(outfolder,breed)
            coan_f="{}/JAW/group_coancestry.csv".format(outfolder)
            
            ## Read group coancestry for yearly runs
            coan = pd.read_csv(coan_f)
            
            ## Read in the recoded pedigree
            jjiv = pd.read_csv(jjiv_f)
            jjiv.columns=['pid','ps','pd','psex','tag']
            
            ## Read in COIs
            wml = pd.read_csv(wml_f)
            
            ## Merge
            meu = pd.merge(jjiv,wml[['f','tag']],on='tag')
            
            ## Read colleau output
            coll = pd.read_csv(coll_f, names=['pid','ps','pd','psex','COI','rel_own','rel_other','tag'])
            
            ## Merge the outputs with ped
            t1 = pd.merge(meu,coll[['tag','rel_own','rel_other']],on='tag',how='outer')
            t1.rename(columns={'tag':'IRN','f':'F'},inplace=True)
            
            ## Merge with pedigree 
            t2 = pd.merge(ped,t1,on='IRN',how='inner')
            dfa = t2.reset_index(drop=True)
            
                        
            ## If there are any duplicates in the original pedigree - these need to be removed. 
            ## Drop duplicated pid entries
            dfa = dfa.drop_duplicates(subset=['pid'])
            
            ## Date handling - this may be time consuming, but pandas date handling is more straightforward than using numpy
            ## Convert DOB to datetime        
            dfa['DOB'] = pd.to_datetime(dfa['DOB'], dayfirst=True, errors='coerce')
            
            ## Fill missing values with 01/01/1900
            try:
                dfa['DOB'] = dfa['DOB'].fillna('01/01/1900 00:00:00')
            except AttributeError:
                dfa['DOB'] = dfa['DOB'].fillna('01/01/1900')
            
            ## Convert DOB to datetime        
            dfa['DOB'] = pd.to_datetime(dfa['DOB'], dayfirst=True, errors='coerce')
            dfa['yob'] = dfa['DOB'].dt.year
            
            ## Update - clean up dates with yob>2022. Keep the dog record, but remove DOB
            fut = dfa.loc[dfa['yob']>2022]
            print("Removing DOBs for {} dogs, as their year of birth is >2022".format(len(fut)))
            dfa.loc[dfa['yob']>2022,'DOB']='1900-01-01'
            dfa.loc[dfa['yob']>2022,'yob']=1900
            
            
            
            ## Get dserial
            dfa['dserial']=dfa['DOB'].apply(lambda x: x.toordinal() + 366)
            
            ## Sort values by pid
            dfa = dfa.sort_values(by=['pid'])
            
            ## Set the type for registrations - otherwise some bugs may come up
            dfa['Register']=dfa['Register'].astype('str')
            dfa['RegType']=dfa['RegType'].astype('str')
            
            ## Convert ped to numpy arrays for faster processing
            pida,psirea,pdama,yoba,sexa,dseriala,Fa = paf.pd_to_np(dfa)
            print("{} dogs in pedigree, total".format(len(dfa)))
            print("\nTotal number of parents in pedigree:")
            print("Unique sires: ",len(np.unique(psirea))-1)
            print("Unique dams: ", len(np.unique(pdama))-1)
            
            ## Assign unique litters
            littersa = paf.assign_litters(pida,psirea,pdama,dseriala)
            dfa['litter']=littersa
            
            ## Get some basic stats
            nanima, maxyoba, minyoba, uniq_yoba, nyrsa, tborna,nlittsa = paf.basic_stats(pida,yoba,littersa)
            
            ## -----------------------------------------------------------------------------
            ## -----------------------------------------------------------------------------
            ## Create an empty table to store regression coefficients
            regcoeff = pd.DataFrame(columns=['variable','intercept','regression_coeff','SE','pval'])
            
            ## Basic statistics on complete data 
            ## Registration statistics - full pedigree. Output goes to out_tables/imports_by_country.csv and registration_types.csv
            ## Update July 2023 - remove country, where it says UK
            print("\n*********************************************************************")
            print("IMPORTS")
            print("Removing country of origin for {} imports, where the country was listed as UK".format(len(dfa.loc[dfa['OrigCountry']=='United Kingdom'])))
            dfa.loc[dfa['OrigCountry']=='United Kingdom', 'OrigCountry']=np.NaN
            imps = paf.reg_type_stats(dfa,outfolder)
            
            ## Import trends - output goes to out_plots/import_trends.jpg
            try:
                ibyc,ibya,ibyb,se,pval = paf.imp_trends(imps, psirea,pdama,outfolder)
                regcoeff.loc[len(regcoeff)+1]=['N imports over yob',ibya,ibyb,se,pval]  
            except TypeError:
                paf.imp_trends(imps, psirea,pdama,outfolder)
            print("\n")
            
            ## Champion statistics. In further analyses, sdb (studbook number holding dogs) are used to approximate purpose-bred dog population
            champs,sdb = paf.champ_stats(dfa,outfolder)
            
            
            ## Get the number of dogs born in a year
            # yob = np.array(dfa['yob'])
            yob_counts = np.unique(yoba,return_counts=True)
            
            ## -----------------------------------------------------------------------------
            ## -----------------------------------------------------------------------------
            ## Tom's functions which require the complete pedigree including imports and their ancestors. Any filtering here causes spurious results, as calculated F doesn't match depth of pedigree after filtering. 
            
            print("\n*********************************************************************")
            print("Genetic parameter calculation on whole pedigree\n\n")
            ## Calculate number of generations per yob using the full pedigree - output goes to ngen_v_yob_py.jpg
            paf.FUNC01_ngen(dfa,nanima,pida,psirea,pdama,outfolder)
            
            ## Calculate census statistics for filtered data
            tborna,nlittsa = paf.FUNC02_census_stats(nyrsa,uniq_yoba,yoba,psirea,pdama,littersa,outfolder,'census_stats_all')
            
            ## Generation interval and breeding stats - output saved to gen_int_log_py.csv
            ## Data includes imports!
            tyrs = paf.FUNC04_genint_ji(nyrsa,outfolder,uniq_yoba,yoba,pida,psirea,pdama,dseriala)

            k = paf.FUNC05_F_by_year(nyrsa,uniq_yoba,yoba,Fa,outfolder,'all')
            ## Add coancestry per year - this is added separately outside of the function, as this is NOT produced for filtered data
            k3 = pd.merge(k,coan,on=['YOB'])
            k3.to_csv('{}/out_tables/F_by_year_{}.csv'.format(outfolder,'all'),index=False)
            paf.FUNC05_F_by_year_plot_a(k,outfolder,'all')
            ## Repeat later for national dogs only
            
            ## Function plotting observed and expected inbreeding - based on individual dog records. Note, this includes all data. Can't do this for national dogs only, as it doesn't read the generation interval correctly
            k2=paf.FUNC06_obs_exp_F(k,coan,uniq_yoba,outfolder,tyrs)
            
            ## Function calculating effective population size
            g = paf.FUNC_Ne(k2,uniq_yoba,tyrs,outfolder)
            print(g)
            print("\n\n")
            
            ## Function calculating sire and dam statistics in 5-year periods
            y1, nps, npd = paf.FUNC7_sd_stats(yoba,uniq_yoba,maxyoba,tborna,psirea,pdama,outfolder)
            
            ## Function calculating 5-year dF and Ne
            paf.FUNC7_dfNe_stats(y1,uniq_yoba,tyrs,k2,'Mean_F',outfolder)
            
            ## Function calculating 5-year dF and Ne
            paf.FUNC7_dfNe_stats(y1,uniq_yoba,tyrs,k2,'group_coancestry',outfolder)
            
            ## Function plotting stem plot for sires and dams
            paf.FUNC07_stem(nps,npd,y1,uniq_yoba,outfolder)
            
            print("\n*********************************************************************")
            print("Data filter for actual dogs registered (Breed registry), including imports (but not their ancestors), as well as ATCs; born since 1990")
            df1 = dfa.loc[((dfa['Register']=='Breed register') | (dfa['RegType']=='ATC')) & (dfa['yob']>1989)]
            print("Retained {} ({:3.2f}%) of {} dogs in complete pedigree".format(len(df1), (len(df1)/len(dfa)*100),len(dfa)))
            
            if len(df1)>0:
            
                ## Convert ped to numpy arrays for faster processing
                pidr,psirer,pdamr,yobr,sexr,dserialr,Fr = paf.pd_to_np(df1)
                
                ## Read litters
                littersr = np.array(df1['litter'])
                
                ## Get some basic stats
                nanimr, maxyobr, minyobr, uniq_yobr, nyrsr, tbornr,nlittsr = paf.basic_stats(pidr,yobr,littersr)
    
                #paf.reg_volatility(tbornr,uniq_yobr,yob_counts,outfolder)
                
                print("\n*********************************************************************")
                print("\nData filter - only national dogs (removed imports, their ancestors and ATCs)")
                # df2 = dfa.loc[(dfa['RegType']!='nan') & (dfa['RegType']!='ATC') & (dfa['RegType']!='Importations')]
                
                ## This is changed in v2. Now, the national dogs are defined as beingin Breed Register, and not being imports
                df2 = dfa.loc[(dfa['Register']=='Breed register') & (dfa['RegType']!='Importations')]
                df = df2.loc[df2['yob']>1989]
                df = df.reset_index(drop=True)
                print("Retained {} ({:3.2f}%) of {} dogs in complete pedigree".format(len(df), (len(df)/len(dfa)*100),len(dfa)))
                print("Stats on national dogs\n")
                
                if len(df)>0:
    
                    ## Convert ped to numpy arrays for faster processing
                    pid,psire,pdam,yob,sex,dserial,F = paf.pd_to_np(df)
                    
                    ## Assign unique litters
                    litters = np.array(df['litter'])
                    
                    ## Get some basic stats
                    nanim, maxyob, minyob, uniq_yob, nyrs, tborn, nlitts = paf.basic_stats_b(pid,yob,litters)   
                    
                    
                    ## Calculate census statistics for filtered data
                    tborn,nlitts = paf.FUNC02_census_stats(nyrs,uniq_yob,yob,psire,pdam,litters,outfolder,'census_stats_national')
        
                    ## Calculate trends for number of registrations - national dogs only
                    a,b,se,pval = paf.FUNC03_reg_trends(tborn,uniq_yob,maxyob,outfolder,'tborn') 
                    regcoeff.loc[len(regcoeff)+1]=['N born over yob',a,b,se,pval]
        
                    ## Calculate trends for number of registrations - national dogs only
                    a,b,se,pval = paf.FUNC03_reg_trends(nlitts,uniq_yob,maxyob,outfolder,'litters') 
                    regcoeff.loc[len(regcoeff)+1]=['N litters over yob',a,b,se,pval]
                    
                    ## Proportion of dogs used in breeding
                    pM_a, pM_b, pM_se, pM_p, pF_a, pF_b, pF_se, pF_p = paf.prop_born_bred(uniq_yob,sex,pid,psire,pdam,yob,outfolder,'national')
                    regcoeff.loc[len(regcoeff)+1]=['prop males bred over yob',pM_a,pM_b,pM_se,pM_p]
                    regcoeff.loc[len(regcoeff)+1]=['prop females bred over yob',pF_a,pF_b,pF_se,pF_p]
                    
                    ## Proportion of dogs used in breeding - both sexes, born between 2005 and 2016
                    df3 = df.loc[(df['yob']>2004) & (df['yob']<2016)]
                    df3.reset_index(inplace=True, drop=True)
                    
                    if len(df3)>0:
                        parents = df3.loc[(df3['pid'].isin(df['ps'])) | (df3['pid'].isin(df['pd']))]
                        print("Proportion of dogs born between 2005 and 2015 inclusive, that became parents: {:3.2f}%".format(len(parents)/len(df3)*100))
                        print("  N born: ", len(df3))
                        print("  N parents: ", len(parents))
                    else:
                        print("Proportion of dogs born between 2005 and 2015 inclusive, that became parents: NA")
                        print("  N born: ", len(df3))
                        print("  N parents: 0")#, len(parents))
                    
                    ## Function comparing F when 0s are included and when they are removed. The larger the difference, the more imports. This will be done for national dogs only, as ancestors of imports will be inflating the 0 part. 
                    k = paf.FUNC05_F_by_year(nyrs,uniq_yob,yob,F,outfolder,'national')
                    paf.FUNC05_F_by_year_plot_a(k,outfolder,'national')
                    
                    ## Calculate trends for litters size
                    ldf,a,b,se,pval = paf.litter_trends(litters,psire,pdam,dserial,yob,outfolder,'litterSize_stats_national')
                    regcoeff.loc[len(regcoeff)+1]=['litter size over yob',a,b,se,pval]
                    
                    ## Plot distribution of litter size
                    paf.litt_dist(ldf,outfolder,'national')
                    ## Calculate change of litter size with F - extremes removed
                    ldf, a,b,se,pval = paf.COI_litter_b(ldf,litters,F,outfolder,'national')
                    regcoeff.loc[len(regcoeff)+1]=['litter size over COI',a,b,se,pval]

                    
                    ## Box plot of litter sizes in COI categories - includes extremes - EDIT, save to table too, for across breed comparisons!
                    ldf = paf.Fcat_litter(ldf,outfolder,'national')
                    
                    ## Litter's COI with, and without litters where at least one parent was an import
                    impBred_lit = paf.COI_split(ldf,imps,psire,pdam,litters,'import','national',outfolder)
                    
                    ## COI by show status - non-show litters are those were neither parent was granted a studbook no
                    purpBred_lit = paf.COI_split(ldf,sdb,psire,pdam,litters,'Purpose-bred','Pet',outfolder)
                    
                    ## COI by show status - non-show litters are those were neither parent was granted a studbook no
                    champBred_lit = paf.COI_split(ldf,champs,psire,pdam,litters,'Champion-bred','NonChampion-bred',outfolder)
        
                    ## Percentage of litters produced from show, and imported dogs
                    ldf=paf.prop_imp_champ(ldf,impBred_lit,purpBred_lit,outfolder,'national')
                    
                    ## Calculate litter statistics - data for each litter printed out to file!
                    ldf2=paf.get_page(df,ldf,litters,psire,pdam,pida,dseriala,outfolder)
                    print("Litter statistics for {} litters saved in litter_stats.csv\n".format(len(ldf2)))
                    
                    ## Filter the litter data - remove any litters where sire's age is <6 months, and dam's age is <8 months, corresponding to 4 and 6 months at mating
                    n = len(ldf2)
                    ldf2 = ldf2.loc[(ldf2['sireage']>=6) & (ldf2['damage']>=8) & (ldf2['sireage']<180) & (ldf2['damage']<144)]
                    print("Dropped {} litters, after removing litters where sire was <6mo or >15 years old and/or dam was <8mo or >12 years old at birth of litter".format(n-len(ldf2)))
                    print("Data saved in litter stats files includes these outlier litters!")
    
    
    
                    ## Depending on data availability, use correct function
                    if len(imps)>0 and len(sdb)>0:
                        paf.plot_page(ldf2,imps,sdb,'sire',outfolder)
                        paf.plot_page(ldf2,imps,sdb,'dam',outfolder)
                    elif len(imps)>0 and len(sdb)==0:
                        paf.plot_page_b(ldf2,imps,'Imports','sire',outfolder)
                        paf.plot_page_b(ldf2,imps,'Imports','dam',outfolder)
                    elif len(imps)==0 and len(sdb)>0:
                        paf.plot_page_b(ldf2,imps,'StudBookNo','sire',outfolder)
                        paf.plot_page_b(ldf2,imps,'StudBookNo','dam',outfolder)
                    else:
                        paf.plot_page_c(ldf2,'sire',outfolder)
                        paf.plot_page_c(ldf2,'dam',outfolder)
                    
                    
                    sires= paf.sire_stats(ldf2,imps,sdb,outfolder,dfa,psire,10)
                    
                    ss = paf.sire_tab(sires,outfolder)
                    
                    paf.sdb_before_after(sdb,ldf2,outfolder)
                    
                    res,ldf4 = paf.CS_stats(df,ldf,outfolder)
                    try:
                        aCS,bCS,seCS,pvalCS,aAI,bAI,seAI,pvalAI = paf.trends_CS_AI(ldf4,outfolder)
                        regcoeff.loc[len(regcoeff)+1]=['Csections over yob',aCS,bCS,seCS,pvalCS]
                        regcoeff.loc[len(regcoeff)+1]=['AI over yob',aAI,bAI,seAI,pvalAI]    
                    except (ValueError, KeyError) as e:
                        pass
            
                    print(res)
                    
                    
                    ##################################################################
                    ## RELATIONSHIPS
                    
                    print("\n*********************************************************************")
                    print("Relationships\n\n")
                    print("\nData filter - active set")
                    df4 = dfa.loc[~dfa['rel_own'].isna()]
                    dfr = df4.reset_index(drop=True)
                    print("Retained {} ({:3.2f}%) of {} dogs in complete pedigree".format(len(dfr), (len(dfr)/len(dfa)*100),len(dfa)))
                    ## Convert ped to numpy arrays for faster processing
                    pidr,psirer,pdamr,yobr,sexr,dserialr,Fr = paf.pd_to_np(dfr)
                    ## Multiply the relationships to get %
                    dfr['R']=dfr['rel_own']*100
                    
                    ## Assign unique litters
                    littersr = np.array(dfr['litter'])
                    
                    ## Get some basic stats
                    nanimr, maxyobr, minyobr, uniq_yobr, nyrsr, tbornr, nlittsr = paf.basic_stats_b(pidr,yobr,littersr)   
                    
                    ## Basic statistics
                    paf.print_stats(dfr['R'],outfolder,'relationships_stats')
                    
                    ## Histogram of mean relationships
                    paf.plot_Rel(dfr,outfolder)
                    
                    ## Trend in R - in small breeds this throws a weird error where ols doesn't produce regression coefficient, instead estimates each level for yob. This isn't that useful anyway, so not using it anymore. Jul 2023
                    # paf.trend_R(dfr,outfolder)
                    
                    ## Compare mean relationship of dogs with Studbook Np and those without
                    print("Comparing mean relationship of dogs with and without Studbook No")
                    if len(sdb)>0:
                        x = dfr.loc[dfr['StudBookNo'].isna()]['R']
                        y = dfr.loc[~dfr['StudBookNo'].isna()]['R']
                        p = paf.ttest2(x,y,0.05)
                    
                        try:
                            ## Calculate summary statistics
                            resx = paf.print_stats(x)
                            resy = paf.print_stats(y)
                            
                            res = pd.merge(resx,resy,on='statistic',suffixes=['_noSDB','_SDB'])
                            res.loc[len(res)+1] = ['p_val for comparison',np.NaN,p]
                            
                            res.to_csv("{}/out_tables/relationships_SDB_compare_stats.csv".format(outfolder), index=False)
                        except IndexError:
                            print("No dogs with Studbook No in active set")
                    else:
                        print("No dogs with StudBookNo available in this breed")
                        
                        
                    ## Evidence of non-random mating
                    fexp = round(round(dfr['rel_own'].mean(),4)/2,4)
                    fobs = round(dfr['F'].mean(),4)
                    
                    alpha = 1- ((1-fobs)/(1-fexp))
                    print("\nAssessing evidence for non-random mating")
                    print("Alpha = {:4.3f}".format(alpha))
                    print("Alpha values below 0 indicate completely random mating")
                    print("Alpha values above 0 indicate selection")
                    

                    
                    # ## Plot % contribution of the top 3 or 2 kennels per year. Proportion against the total number born in that year.
                    # paf.top_kennels(dfa,yob_counts,outfolder)
                        
                    regcoeff.to_csv("{}/out_tables/regression_coefficients.csv".format(outfolder),index=False)
                    
                    plt.close('all')
                else:
                    print("No national dogs retained")
                
            else:
                print("No dogs retained after filtering out ATC and ancestors of imports")
                
            ## Kennel prefix stats - output goes to out_plots/kennel_dogsN.jpg
            paf.kennel_stats(dfa,outfolder)
                                
            
        else:
            print("Pedigree file appears to be empty")

    else:
        print("The script requires the input file path and the breed code")
        
    