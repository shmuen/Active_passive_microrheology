# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:37:03 2019

@author: Sebastian

Goal: analysis of rheology data
display of plots of analysis

"""

#import of necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sb
import pandas as pd
import os
import scipy.io
import scipy.constants as const
import pickle
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PIL import Image
#import registration
import pickle 
import time
import scikit_posthocs as sp
import statsmodels.api as sm
import copy
    #p = pickle.load(open("p","rb"))   


#hardcoded variables: 
#in_axis = "y"
bead_diameter = 10**-6
passive_binning = 20
identifier = 0
frequency_passive = np.array([.1, .2, .5, 1, 2.2, 4.6, 10, 21.5, 46.4, 100, 215, 464, 1000, 2154, 4642, 10000])
# frequency_passive = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0])


    
def save_var(name,var):
    "saves variable var with und name and extension .pkl"
    with open(name,'wb') as file_pointer:      
        pickle.dump(var,file_pointer) 
   
def load_var(name):
    return pickle.load(open(name,"rb"))   

#%% read in of data

#read data of active rheology measurement
def get_response(single_folder, idx = 0):
    "reads in data of active rheology experiment"
    #get all files with name 'deformation_response.mat' and 'parameters.mat'
    #they contain the responses for every frequency and timestamps
    response_paths=[single_folder + '/' + k for k in os.listdir(single_folder) if 'deformation_response.mat'  in k or (k.startswith('Active_microrheology') and k.endswith('.lvb')) or ('amplitude' in k)]   
    parameters_paths = [single_folder + '/' + k for k in os.listdir(single_folder) if 'parameters.mat' in k]
    results_paths = [single_folder + '/' + k for k in os.listdir(single_folder) if 'results.mat' in k]
    
    #check if data was read in from Bart's program
    new_prog = "Active_microrheology" in response_paths[0]
    tills_prog = "amplitude" in response_paths[0]
    
    #create variables for frequency f, response in x and y (ax, ay), slopes, time, trap position and beadsizes
    f, ax, ay = [],[],[]
    slopes, slopes_lunam = [], []
    time_list, trap_list = [], []
    beadsize = []
        
    #first get lunam slopes
    try:
        slopes_lunam = scipy.io.loadmat(response_paths[0])["xy_slope"][0][0:4]
    except:
        slopes_lunam = [0, 0, 0, 0]
           
    #then frequency, response and detector slopes
    #if Bart's program was used, use function to read in and analyse data
    if new_prog:
        f, ax, ay, slopes, trap_list, time_list, beadsize, raw_response = get_modulus(response_paths)
    elif tills_prog:
        f, ax, ay, slopes, trap_list, time_list, beadsize, raw_response = get_modulus_till(response_paths)
    else:
        if '1000_100deformation_response.mat' in '-'.join(response_paths):
            response_paths.sort()   
            
        for response_file in response_paths:
            #if Timo's program is used, take 1 µm as standard diameter
            beadsize = 1
            try:
                response = scipy.io.loadmat(response_file) 
                f.append(response["f"][0][0])
                ax.append(response["a_xd"][0][0])
                ay.append(response["a_yd"][0][0])
                slopes.append(response["det_slopes"][0][0:2])
            except:
                #if response can not be read nan will be written for that variable
                f.append(np.nan)
                ax.append(np.nan)
                ay.append(np.nan)
                slopes.append([1, 1])
                
        for folder in parameters_paths:
            #timestamps are saved in parameters file in format "HH:MM AM"
            #read in and calculate time in minutes
            try:
                #save time 
                time_list.append(get_time(folder))
            except:
                time_list.append(0)
                
        for folder in results_paths:
            #read in trap positions
            try:
                results = scipy.io.loadmat(folder)
                trap_list.append(results["Traps"][0][0]+1j*results["Traps"][0][1])
            except:
                trap_list.append([0])
    
    #convert diameter to meter
    beadsize = beadsize * 10**-6
    
    #check if it's frequency sweep or frequency time set
    #to distinguish, check if file format is 1000_100   
    count = 0
    if '1000_100deformation_response.mat' in '-'.join(response_paths):
        response_paths.sort()
        for r_path in response_paths:
            if '1000' in os.path.basename(r_path):
                count += 1
                
        total_length = len(response_paths)
        if np.mod(total_length, count) == 0:
            repeat = np.repeat(np.arange(total_length/count),count)
        else:
            repeat = np.repeat(np.arange(count),np.floor(total_length/count))
            repeat = np.concatenate([repeat,np.repeat(count,(len(response_paths)-len(repeat)))])
        
        ID = repeat + identifier   
    else:    
        repeat = np.ones(len(f))*idx
        ID = np.ones(len(f)) * identifier


    #convert to arrays
    f = np.array(f)  
    ax = np.array(ax)
    ay = np.array(ay)
    
    #calculate shear modulus from generalized Stokes-Einstein-Equation: G = 1/(6pi*R*alpha)
    Gx = 1/(3*np.pi*beadsize*ax)
    Gy = 1/(3*np.pi*beadsize*ay)

    #save lunam and detector slopes as complex numbers (real: x, imag: y)
    lunam_cmplx = np.ones(len(f))*(slopes_lunam[0]+1j*slopes_lunam[1])
    lunam_R = np.ones(len(f))*(slopes_lunam[2]+1j*slopes_lunam[3])
    det_cmplx = np.vstack(slopes)
    det_cmplx = det_cmplx[:,0]+1j*det_cmplx[:,1]

    #timelist
    # t = np.ones(len(f))*time_list[0]
    t = time_list
    
    #traplist
    trap = np.ones(len(f))*trap_list[0]
    
    beads = np.ones(len(f))*beadsize
    
    #measurement in that cell
    zero = np.zeros(len(f))

    #save as array with
    active = np.transpose(np.vstack([zero, zero, zero, repeat, f, ax, ay, Gx, Gy, lunam_cmplx, lunam_R, det_cmplx, beads, trap,t, ID,zero]))
    
    raw_response[:,-1] = ID[0]
    
    return active, raw_response


def get_fluctuations(input_folder, f_active = 0, calc_vh_msd = True): #idx_in,f_active = 0): 
    "reads in data of passive rheology experiment"
    
    #find all folders with subfolders that contain passive data
    fluctuation_folders = [dirpath for dirpath, dirnames, filenames in os.walk(input_folder) if "multiple_run_" in dirpath or ("Passive_microrheology" in dirpath) or ('passive' in dirpath)]
    passive, passive_f = [], []    
    idx_repeat = 0
    vanHove = []
    
    #go through every folder
    for single_folder in fluctuation_folders:
        #find all files of raw data containing 'data_streaming'
        fluctuation_paths = [single_folder + '/' + k for k in os.listdir(single_folder) if 'data_streaming_' in k or (k.startswith('Passive_microrheology') and k.endswith('.lvb')) or ('amplitude_' in k)]
        parameters_paths = [single_folder + '/' + k for k in os.listdir(single_folder) if 'Parameters.mat' in k]
        
        if fluctuation_paths:
            #check if data is produced by Bart's program
            new_prog = "Passive_microrheology" in fluctuation_paths[0]
            tills_prog = "passive" in fluctuation_paths[0]
            
            slopes,lunam_slopes = [],[]
            f_b,PSDx_b,PSDy_b,t = [],[],[],[]
            PSDx_b_f, PSDy_b_f,  f_b_to_f, repeat2, ID2 = [], [], [], [], []
            det_cmplx,lunam_cmplx, zero, repeat = [],[], [], []    
            lunam_R = []
            time_list,ID = [],[]
            vanHove, msd = [], []
            rate = 50000   #sampling rate (50 kHz for Bart and Timo's program)
            
        #    data = read_raw(fluctuation_paths,new_prog)
            #read in data in big endian encoding
            if not tills_prog:
                rawdata = [np.fromfile(file,dtype = '>f8') for file in fluctuation_paths]
            
            #data is stored differently in Bart's program
            if new_prog:
                metadata_path = [file[:-6]+"_metadata" + file[-6:-4] + ".txt" for file in fluctuation_paths]
                for metadata in metadata_path:
                    meta_table = np.loadtxt(metadata,dtype = str)
                    n_chan = np.int(meta_table[meta_table[:,0] == "Number_channels",1])
                    slope_x = (np.float(meta_table[meta_table[:,0] == "Detector_slope_x",1]))
                    slope_y = (np.float(meta_table[meta_table[:,0] == "Detector_slope_y",1]))
                    time_hms = np.str(meta_table[meta_table[:,0] == "Date_time",1])[-10:-2]
                    time_list.append(np.int(time_hms[:2])*3600 + np.int(time_hms[3:5])*60 + np.int(time_hms[-2:]))
                    slopes.append([slope_x, slope_y])
                    lunam_slopes.append([0,0,0,0])
                
                data = [np.reshape(k,[n_chan,int(len(k)/n_chan)]) for k in rawdata]
                
                row_y = 1
                
            elif tills_prog:
                raw = [scipy.io.loadmat(file) for file in fluctuation_paths]
                data = [np.vstack((dat["pos_x"][0], dat["pos_y"][0])) for dat in raw]
                for dat in raw:
                    slopes.append([dat["betax"][0][0], dat["betay"][0][0]])
                    lunam_slopes.append([0,0,0,0])
                    time_list.append(0)
                    
                rate = 65536
            else:
                #reshape and cut of first 4 rows (lunam signal and photodiode)
                data = [np.transpose(np.reshape(k[int(k[0])+1:],[int(k[2]),int(k[1])]))[4:,:] for k in rawdata]
                row_y = 5    
            
            #van Hove distribution
            
            #load slopes for detector of all fluctuation data
        
            for idx in range(len(fluctuation_paths)):     
                if not new_prog and not tills_prog:
                    #print(folder + '/fluctuation_data1' + '%03d' % idx + '.mat')        
                    try:
            #            slopes.append(scipy.io.loadmat(single_folder + '/fluctuation_data1' + '%03d' % idx + '.mat',variable_names = ['det_slopes'])["det_slopes"][0])
                        slopes.append(scipy.io.loadmat(single_folder + '/fluctuation_data1' + '%03d' % idx + '.mat',variable_names = ['det_slopes'])["det_slopes"][0])
                    #if slopes can not be read or do not exist append [1,1]
                    except:
                        print("no detector slopes loaded from " + single_folder + '/fluctuation_data1' + '%03d' % idx)
                        slopes.append(np.array([1,1]))
                        
                    try:
                        lunam_slopes.append(scipy.io.loadmat(single_folder + '\\scan_1' + '%03d' % idx + '.mat',variable_names = ['xy_slopes'])["xy_slopes"][0])
                    except:
                        print("no lunam slopes loaded from " + single_folder + '\\scan_1' + '%03d' % idx + '.mat')
                        lunam_slopes.append(np.array([0,0,0,0]))    
                        
                    try:
                        #save time 
                        time_list.append(get_time(parameters_paths[0]))
                    except:
                        time_list.append(0)    
                          
                #calculate PSD for x and y
                psd_x = power_sd(data[idx][0], rate)
                psd_y = power_sd(data[idx][1], rate)            
                
                if isinstance(f_active,np.ndarray):
                    binned_x_to_f = log_f(psd_x["f"],psd_x["PSD"],f_active)
                    binned_y_to_f = log_f(psd_y["f"],psd_y["PSD"],f_active)
                    
                    f_b_to_f.append(f_active)
                    PSDx_b_f.append(binned_x_to_f)
                    PSDy_b_f.append(binned_y_to_f)
                    
        #        else:
                #Timo's binning
        #            binned_x = log_bin(psd_x["f"],psd_x["PSD"],passive_binning)["y"]
        #            binned_y = log_bin(psd_y["f"],psd_y["PSD"],passive_binning)["y"]
        #            f_b.append(log_bin(psd_x["f"],psd_x["PSD"],passive_binning)["x"])
                
                binned_x = log_f(psd_x["f"],psd_x["PSD"],frequency_passive)
                binned_y = log_f(psd_y["f"],psd_y["PSD"],frequency_passive)
                f_b.append(frequency_passive)
                
                #binning to frequencies of active measurement
                #PSDx_b.append(binned_x["y"])
                #PSDy_b.append(binned_y["y"])
                
                PSDx_b.append(binned_x)
                PSDy_b.append(binned_y)
                
                t.append(np.ones(len(f_b[idx]))*time_list[idx])
                
                zero.append(np.zeros(len(f_b[idx])))
                det_cmplx.append(np.ones(len(f_b[idx]))*(slopes[idx][0]+1j*slopes[idx][1]))
                lunam_cmplx.append(np.ones(len(f_b[idx]))*(lunam_slopes[idx][0]+1j*lunam_slopes[idx][1]))
                lunam_R.append(np.ones(len(f_b[idx]))*(lunam_slopes[idx][2]+1j*lunam_slopes[idx][3]))
                
                repeat.append(np.ones(len(f_b[idx]))*(idx_repeat))
                repeat2.append(np.ones(len(f_b_to_f[idx]))*(idx_repeat))
    
                ID = np.concatenate([ID,(np.ones(len(f_b[idx]))*identifier)]) 
                ID2 = np.concatenate([ID2,(np.ones(len(f_b_to_f[idx]))*identifier)])
                
                idx_repeat += 1
                   
            if calc_vh_msd:
                vh = calcVH(data, slopes)
                tmp_msd = MSD(data, slopes)
                zero3 = np.zeros([len(vh),3])
                zero4 = np.zeros([len(tmp_msd),3])
                vanHove.append(np.hstack((zero3, vh)))
                msd.append(np.hstack((zero4, tmp_msd)))
            else:
                vanHove.append([0,0,0,0])
                msd.append([0,0,0,0])
        
            #turn into array
            slopes = np.array(slopes)
            lunam_slopes = np.array(lunam_slopes)
            ID = np.array(ID)
            ID2 = np.array(ID2)
            
            
            zero = np.hstack(zero)
            zero2 = np.zeros(len(np.hstack(f_b_to_f)))
    
    #        return zero3, vh
        
            passive.append(np.transpose(np.vstack([zero,zero,zero,np.hstack(repeat),np.hstack(f_b),np.hstack(PSDx_b),np.hstack(PSDy_b),np.hstack(lunam_cmplx),np.hstack(lunam_R),np.hstack(det_cmplx),np.hstack(t),ID,zero])))
            passive_f.append(np.transpose(np.vstack([zero2,zero2,zero2,np.hstack(repeat2),np.hstack(f_b_to_f),np.hstack(PSDx_b_f), np.hstack(PSDy_b_f),ID2,zero2])))

    vanHove = np.vstack((vanHove))
    msd = np.vstack((msd))
    passive = np.vstack((passive))
    passive_f = np.vstack((passive_f))        
    
    return passive, passive_f, vanHove, msd

#%%analysis functions log binning and PSD calculator
    
def metadata_changer(file_path, row, change_to):
    active_metadata_paths = [dirpath for dirpath, dirnames, filenames in os.walk(file_path) if "Active_microrheology" in dirpath]
    passive_metadata_paths = [dirpath for dirpath, dirnames, filenames in os.walk(file_path) if "Passive_microrheology" in dirpath]
    for metadata_path in active_metadata_paths:
        active_metadata_files = [metadata_path + '/' + k for k in os.listdir(metadata_path) if (k.startswith('Active_microrheology') and k.endswith('.lvb'))]
        
        for metadata in active_metadata_files:
            metadata = metadata[:-6] +  "_metadata" + metadata[-6:-4] + ".txt"
            print(metadata)
            meta_table = np.loadtxt(metadata,dtype = str)
            meta_table[row,1] = change_to
            np.savetxt(metadata, meta_table, fmt = '%s')
            

def get_modulus(file_path, power_law_fit = True):
    "get medulus from raw data"
    buffer,f,n_chan,acq_rate, slope_out, trap_list, time_list, beadsize = [], [], [], [], [], [], [], []
    raw_cut = []
    response_x,response_y = [],[]
    noise = []
    alpha_x = 4.68*10**-11
    alpha_y = 4.85*10**-11
    
    # fig, ax = plt.subplots(4,4,sharex = True,sharey = True)   
    # i = j = 0
    
    for file in file_path:
        #read in file as big endian
        rawdata = np.fromfile(file,dtype = '>f8')
        #read in metadata (same name, but ends in 'metadata.txt')
        metadata_path = file[:-6]+"_metadata" + file[-6:-4] + ".txt"
        try:
            table = np.loadtxt(metadata_path,dtype = str)
        except:
            txt_file = open(metadata_path, "r")
            txt = txt_file.read()
            txt = txt.replace(' ','')
            txt_file.close()
            txt_file = open(metadata_path, "w")
            txt_file.write(txt)
            txt_file.close()
            table = np.loadtxt(metadata_path, dtype = str)
    
        #save values from metadata
        buffer.append(np.float(table[table[:,0] == "Buffer_size",1]))
        f.append(np.float(table[table[:,0] == "Frequency",1]))
        n_chan.append(np.int(table[table[:,0] == "Number_channels",1]))
        acq_rate.append(np.float(table[table[:,0] == "Acq_rate",1]))
        time_hms = np.str(table[table[:,0] == "Date_time",1])[-10:-2]
        time_list.append(np.int(time_hms[:2])*3600 + np.int(time_hms[3:5])*60 + np.int(time_hms[-2:]))
        slope_x = (np.float(table[table[:,0] == "Detector_slope_x",1]))
        slope_y = (np.float(table[table[:,0] == "Detector_slope_y",1]))
        trap_str = table[table[:,0] == "Trap_pos(px)",1][0]
        beadsize = np.float(table[table[:,0] == "radius(um)",1])
#        trap_str = "0;0"
#        print(trap_str)
        trap = trap_str.split(';')
        trap_list.append(np.float(trap[0]) + 1j* np.float(trap[1]))
        slope_out.append([slope_x,slope_y])
        #reshape data according to metadata
        data = np.reshape(rawdata,[n_chan[-1],int(len(rawdata)/n_chan[-1])])
        print((np.max(data[4,:])-np.min(data[4,:]))/slope_y)
        
        #choose direction of oscillation
        lunam_x = data[0,:]
        lunam_y = data[1,:]
        det_signal_x = data[3,:]
        det_signal_y = data[4,:]
        
        #calculate fourier transformation of detector and lunam signal
        det_f_x = np.fft.fft(det_signal_x)
        det_f_y = np.fft.fft(det_signal_y)
        lunam_f_x = np.fft.fft(lunam_x)
        lunam_f_y = np.fft.fft(lunam_y)
        
        if power_law_fit:
            amp_l = np.abs(lunam_f_y)[1:]
            amp_d = np.abs(det_f_y)[1:]
            freq = np.fft.fftfreq(int(buffer[-1]), 1/acq_rate[-1])[1:]
            l = len(freq)//2
            freq_positive = freq[0:l]
    
            f0 = np.abs(freq_positive - f[-1]*0.5).argmin()
            f1 = np.abs(freq_positive - f[-1]*5).argmin()
            
            f_arg = np.abs(freq_positive[f0:f1] - f[-1]).argmin()
            current_f = np.abs(freq - f[-1]).argmin()
            # print(file)
            freqs_wo_f = freq[f0:f1][np.arange(f0, f1) != current_f]
            try:
                fit_pl, pcov = scipy.optimize.curve_fit(power_law_log2, freq[f0:f1][np.arange(f0, f1) != current_f], 
                                                        np.log((2/amp_d.size)*amp_d[f0:f1][np.arange(f0, f1) != current_f]))#, maxfev = 10000)
                norm_d = (2/amp_d.size)*amp_d[f0:f1]/(np.exp(power_law_log2(freq[f0:f1], *fit_pl)))
                # print(fit_pl)
                sd = np.std(norm_d)
                mean = np.mean(norm_d)
                f_norm_d = norm_d[f_arg]
                noise.append(f_norm_d > mean + sd)
                
                raw_cut.append(np.vstack((freq[f0:f1], norm_d, np.repeat(f_arg, len(norm_d)), np.repeat(f[-1], len(norm_d)), np.repeat(0, len(norm_d)))).T)
                
                # ax[i,j].loglog(freq_positive[f0:f1], (2/amp_d.size)*amp_d[f0:f1]/(np.exp(power_law_log2(freq_positive[f0:f1], *fit_pl))))
                # ax[i,j].loglog(freq_positive[f0:f1][f_arg], f_norm_d, 'o')
                # ax[i,j].loglog(freq_positive, (2/amp_l.size)*amp_l[0:l])
                # ax[i,j].loglog(freq_positive, (2/amp_d.size)*amp_d[0:l])
                # j += 1
                # if j == 4:
                #       i += 1
                #       j = 0  
            except:
                print("no fit for power law on raw data " + file)
                fit_pl = [1,1]
                noise.append(False)
                
            #active

    
                
        else:
            noise.append(0)
        
        #find maximum of detector 
        lunam_f_max_x = np.where(np.abs(lunam_f_x[1:]) == np.max(np.abs(lunam_f_x[1:])))[0][0]+1
        lunam_f_max_y = np.where(np.abs(lunam_f_y[1:]) == np.max(np.abs(lunam_f_y[1:])))[0][0]+1
        
        #calculate ratio of lunam and detector
        ratio_x = det_f_x[lunam_f_max_x]/lunam_f_x[lunam_f_max_x]
        ratio_y = det_f_y[lunam_f_max_y]/lunam_f_y[lunam_f_max_y]
        
        prefactor_x = 1./(10**6*alpha_x*slope_x)
        prefactor_y = 1./(10**6*alpha_y*slope_y)
        
        response_x.append(ratio_x*prefactor_x)
        response_y.append(ratio_y*prefactor_y)
        
        

    return f,response_x,response_y,slope_out, trap_list, time_list, beadsize, np.vstack((raw_cut))

def get_modulus_till(file_path):
    #active
    # act_path = [act_pas_path + '\\' + k for k in os.listdir(act_pas_path) if 'active' in k][0]
    # file_path = [act_path + '\\' + k for k in os.listdir(act_path) if 'amplitude' in k]
    
    buffer,f,acq_rate, slope_out, trap_list, time_list = [], [], [], [], [], []
    raw_cut = []
    response_x,response_y = [],[]
    noise = []
    # alpha_x = 4.68*10**-11
    alpha_y = []
    # beta_y = []
    force_x, force_y, disp_x, disp_y = [], [], [], []
    beadsize = 1#10**-6
    
    # fig, ax = plt.subplots(4,4,sharex = True,sharey = True)   
    # i = j = 0
    
    for file in file_path:
        #read in file as big endian
        table = scipy.io.loadmat(file)

        #save values from metadata
        alpha_y = np.float(table["alphay"][0][0])
        alpha_x = np.float(table["alphax"][0][0])
        sum_signal = table["pos_xy"][0]
        slope_x = np.float(table["betax"][0][0])
        slope_y = np.float(table["betay"][0][0])
        slope_out.append([slope_x, slope_y])
        f.append(np.float(table["mirror_freq_y"][0][0]))
        acq_rate.append(np.float(table["rate"][0][0]))
        force_y.append(table["forcey"][0])
        force_x.append(table["forcex"][0])
        disp_y.append(table["pos_y"][0])# / sum_signal)
        disp_x.append(table["pos_x"][0])
        buffer.append(len(force_y[-1]))
        trap_list.append(0)
        time_list.append(0)
        
        #calculate fourier transformation of detector and lunam signal
        det_f_x = np.fft.fft(disp_x[-1])
        det_f_y = np.fft.fft(disp_y[-1])
        force_f_x = np.fft.fft(force_x[-1])
        force_f_y = np.fft.fft(force_y[-1])
        
        #find maximum of detector 
        lunam_f_max_x = np.where(np.abs(force_f_x[1:]) == np.max(np.abs(force_f_x[1:])))[0][0]+1
        lunam_f_max_y = np.where(np.abs(force_f_y[1:]) == np.max(np.abs(force_f_y[1:])))[0][0]+1
        
        #calculate ratio of lunam and detector
        ratio_x = det_f_x[lunam_f_max_x]/force_f_x[lunam_f_max_x]
        ratio_y = det_f_y[lunam_f_max_y]/force_f_y[lunam_f_max_y]
        
        prefactor_x = slope_x/alpha_x
        prefactor_y = slope_y/alpha_y
        
        response_x.append(ratio_x*prefactor_x)
        response_y.append(ratio_y*prefactor_y)
        
        if 1:
            amp_l = np.abs(force_f_y)[1:]
            amp_d = np.abs(det_f_y)[1:]
            freq = np.fft.fftfreq(int(buffer[-1]), 1/acq_rate[-1])[1:]
            l = len(freq)//2
            freq_positive = freq[0:l]
    
            f0 = np.abs(freq_positive - f[-1]*0.5).argmin()
            f1 = np.abs(freq_positive - f[-1]*5).argmin()
            
            f_arg = np.abs(freq_positive[f0:f1] - f[-1]).argmin()
            current_f = np.abs(freq - f[-1]).argmin()
            # print(file)
            freqs_wo_f = freq[f0:f1][np.arange(f0, f1) != current_f]
            try:
                fit_pl, pcov = scipy.optimize.curve_fit(power_law_log, freq[f0:f1][np.arange(f0, f1) != current_f], 
                                                        np.log((2/amp_d.size)*amp_d[f0:f1][np.arange(f0, f1) != current_f]), maxfev = 1000)
                norm_d = (2/amp_d.size)*amp_d[f0:f1]/(np.exp(power_law_log(freq[f0:f1], *fit_pl)))
                sd = np.std(norm_d)
                mean = np.mean(norm_d)
                f_norm_d = norm_d[f_arg]
                noise.append(f_norm_d > mean + sd)
                
                raw_cut.append(np.vstack((freq[f0:f1], norm_d, np.repeat(f_arg, len(norm_d)), np.repeat(f[-1], len(norm_d)), np.repeat(0, len(norm_d)))).T)
            except:
                print("no fit for power law on raw data")
                noise.append(False)
        
    return f, response_x, response_y, slope_out, trap_list, time_list, beadsize, np.vstack((raw_cut))#np.array([[0,0],[0,0]])#passive#, np.vstack((raw_cut))
    

def power_sd(raw,srate):
    "calculates the power spectral density for given data and sampling rate"
    
    p = len(raw)
    #low_f = srate/p
    
    #fNyq  = srate/2
    delta_t = 1/ srate
    
    time = np.arange(0,p*delta_t,delta_t)
    T = max(time)
    
    #Fourier transformation of raw signal and multiplication with complex conjugate
    Y = np.fft.fft(raw,p)
    Pyy = Y*np.conj(Y)/ (p*srate)
    
    f = srate/p*(np.arange(0, p/2+1))
    Pyy_out = Pyy[0:int(p/2)+1]
     
    data = {"f":f, "PSD":Pyy_out,"T":T}
    
    return data

def log_bin(x_in,y_in,final_length):
    "logarithmic binning with specified final length"
    
    #Timo's way of binning 
    f_m = len(x_in)
    up = np.log10(f_m)
    interm = np.arange(-(up-up/final_length),up/final_length,up/final_length)+(up-up/final_length)
    bdr = np.round(10**interm)
    np.append(bdr,bdr[-1]+1)
    
    #k = 1.5
    
    x,x_,y,y_,y_std = [],[],[],[],[]
    
    for i in range(1,len(bdr)):
        if bdr[i-1] < bdr[i]:
            if bdr[i-1] == bdr[i]-1:
                part = int(bdr[i-1])-1
            else:
                part = np.arange(int(bdr[i-1])-1,int(bdr[i])-1)
            np.transpose(part)
            x.append(np.mean(x_in[part]))
            x_.append(np.median(x_in[part]))
            y.append(np.mean(y_in[part]))
            y_.append(np.median(y_in[part]))
            y_std.append(np.std(y_in[part]))
    
    
    #skip first value because it's zero and that screws up the loglog
    x = x[1:]
    y = y[1:]
    
    data = {"x":x, "y":y}
    
    return data

def log_f(x_in, y_in, f):
    "log bin with specified frequenices"
    
    #create list with boundary, starting at 0
    bdr = [0]
    #create list for output (mean of psd values at frequencies)
    y_out = []
    
    #add one more frequency
    f = np.append(f,10**(np.log10(f[-1])+np.log10(f[-1])-np.log10(f[-2])))
 
    #go through all frequencies
    for idx in range(1,len(f)):
        #calculate logarithmic middle between two frequencies and save as boundary
        bdr.append(np.real(10**((np.log10(f[idx-1])+np.log10(f[idx]))/2)))
        #get all indices of values that lie within these borders
        values = (np.logical_and(bdr[idx-1] < x_in, x_in <= bdr[idx]))
        #calculate mean of all the values that are between two boundaries
        y_out.append(np.mean(y_in[values]))
    
    #add mean from last boundary to end
    y_out.append(np.mean(y_in[x_in > bdr[-1]]))
    
    return y_out[:-1]
    

def get_time(folder):
    "reads in the times field of parameters.mat file and calculates time in minutes from midnight"
    
    parameters = scipy.io.loadmat(folder)
    #split format at space
    temp_time = (parameters["Parameter"]["Time"][0][0][0]).split(' ')
    #split again at : and convert to integers
    time = temp_time[0].split(':')
    time[0] = int(time[0])
    time[1] = int(time[1])
    #convert AM/PM to 24h format and calculate time in minutes
    if temp_time[1] == 'PM' and time[0] != 12:
        time[0] = time[0]+12
    
    if temp_time[1] == 'AM' and time[0] == 12:
        time[0] = time[0]-12
    
    time_in_min = time[0]*3600 +time[1]*60
    
    return time_in_min

#%% batch analysis
from PyQt5 import QtCore, QtWidgets, QtGui, uic
    
def analyse_measurement(folder, idx_measure = 0,suppress = False, progressBar = None, calc_vh_msd = True):
    "analyses active and passive rheology of one measurement and bins passive data \
    \nOutput can be suppressed by 'suppress = True'"
    
    global identifier
    
    start = time.time()

    #active
    response_folders = [dirpath for dirpath, dirnames, filenames in os.walk(folder) if "response_function" in dirpath or ("Active_microrheology" in dirpath) or (("active" in dirpath and not "figure" in dirpath))]    
    active, raw_response = [get_response(folder,idx) for idx,folder in enumerate(response_folders)][-1]
    active = np.vstack(active)
    active[:,2] = idx_measure
    
    scan_folder_list = [os.path.dirname(dirpath) for dirpath, dirnames, filenames in os.walk(folder) if "response_function" in dirpath or ("Active_microrheology" in dirpath) or (("active" in dirpath and not "figure" in dirpath))]
    scans = get_scans(scan_folder_list)
    
    scans[:,2] = idx_measure
#    active[:,-2] = identifier
    #passive
#    fluctuation_folders = [dirpath for dirpath, dirnames, filenames in os.walk(folder) if "multiple_run_" in dirpath or ("Passive_microrheology" in dirpath)]
#    passive_out = [get_fluctuations(folder,idx+identifier,np.unique(active[:,4])) for idx,folder in enumerate(fluctuation_folders)] 
    passive_out = get_fluctuations(folder, f_active = np.unique(active[:,4]), calc_vh_msd = calc_vh_msd)
    
#    passive_out = [get_fluctuations(folder,idx,np.unique(active[:,4])) for idx,folder in enumerate(fluctuation_folders)] 
#    passive = [get_fluctuations(folder,idx+identifier) for idx,folder in enumerate(fluctuation_folders)] 
#    passive = np.vstack([idx[0] for idx in passive_out])
    passive = passive_out[0]
    passive[:,2] = idx_measure
    passive_f = passive_out[1]    
#    passive_f = np.vstack([idx[1] for idx in passive_out])
    passive_f[:,2] = idx_measure
#    passive[:,-2] = identifier
    vanHove = passive_out[2]
    vanHove[:,2] = idx_measure
    
    msd = passive_out[3]
    msd[:,2] = idx_measure
    
    identifier = active[-1,-2]+1
    if progressBar is not None:
        Bar, lblProgress, total = progressBar
        Bar.setValue(int((identifier) / total * 100))
        lblProgress.setText(str(int(identifier)) + "/" + str(total))
        QtWidgets.QApplication.processEvents()
        
    
    str_folder = str(folder)
    
    data = {"active":active, "passive":passive, "passive_f":passive_f, "vanHove":vanHove, "msd":msd, "scans":scans, "_folder":str_folder, "raw_response":raw_response}
               
    end = time.time()
    elapsed_time = end - start
    
    if suppress:   
        return
    return data, elapsed_time

#go through subfolders of measurements and run full analysis followed by plot_all
#input: folder of one cell
def analyse_cell(folder,idx_cell = 0, progressBar = None, calc_vh_msd = True):
    "analyses all measurements of one cell by calling analyse_measurements for each measurement"
    
    folder_list = [os.path.dirname(dirpath) for dirpath, dirnames, filenames in os.walk(folder) if "response_function" in dirpath or ("Active_microrheology" in dirpath) or ("active" in dirpath and not "figure" in dirpath)]
    folder_list_unique = np.unique(folder_list)
    
    data = {"active":[], "passive":[], "passive_f":[], "vanHove":[], "msd":[], "_folder":[], "scans":[], "raw_response":[]}
    for idx_measure, current_folder in enumerate(list(dict.fromkeys(folder_list_unique))):
        print("measurement " + str(idx_measure + 1) + " of " + str(len(folder_list_unique)))
        temp, elapsed_time = analyse_measurement(current_folder,idx_measure, progressBar = progressBar, calc_vh_msd = calc_vh_msd)
        data["active"].append(temp["active"])
        data["passive"].append(temp["passive"])
        data["passive_f"].append(temp["passive_f"])
        data["vanHove"].append(temp["vanHove"])
        data["msd"].append(temp["msd"])
        data["_folder"].append(temp["_folder"])
        data["scans"].append(temp["scans"])
        data["raw_response"].append(temp["raw_response"])
    
    data["active"] = np.vstack(data["active"])
    data["passive"] = np.vstack(data["passive"])
    data["passive_f"] = np.vstack(data["passive_f"])
    data["vanHove"]  =  np.vstack(data["vanHove"])
    data["msd"]  =  np.vstack(data["msd"])
    data["scans"] = np.vstack(data["scans"])
    data["raw_response"] = np.vstack(data["raw_response"])
    data["active"][:,1] = idx_cell
    data["passive"][:,1] = idx_cell       
    data["passive_f"][:,1] = idx_cell
    data["vanHove"][:,1] = idx_cell
    data["msd"][:,1] = idx_cell
    data["scans"][:,1] = idx_cell
    
    
    return data, elapsed_time
    
#go through all measurements of one cell and save as list
#input: folder of one measurement day
def analyse_day(folder,idx_day = 0, progressBar = None, calc_vh_msd = True):
    "analyses all measurements of one day by calling analyse_cell for each cell"
    #doesn't work with new data with os.path.dirname(os.path.dirname(dirpath))
    folder_list = [os.path.dirname(os.path.dirname(dirpath)) for dirpath, dirnames, filenames in os.walk(folder) if ("response_function" in dirpath) or ("Active_microrheology" in dirpath)]
    folder_list_unique = np.unique(folder_list)
    data = {"active":[], "passive":[], "passive_f":[], "vanHove":[], "msd":[], "_folder":[], "scans":[], "raw_response":[]}
    for idx_cell, current_folder in enumerate(folder_list_unique):
        # if progressBar is not None:
        #     progressBar.setValue(int(idx_cell / len(folder_list_unique) * 100))
        print("cell " + str(idx_cell + 1) + " of " + str(len(folder_list_unique)))# + ". (Path: " + current_folder + ")")
        temp, elapsed_time = analyse_cell(current_folder, idx_cell, progressBar = progressBar, calc_vh_msd = calc_vh_msd)
        data["active"].append(temp["active"])
        data["passive"].append(temp["passive"])
        data["passive_f"].append(temp["passive_f"])
        data["vanHove"].append(temp["vanHove"])
        data["msd"].append(temp["msd"])
        data["_folder"].append(temp["_folder"])
        data["scans"].append(temp["scans"])
        data["raw_response"].append(temp["raw_response"])
        
    data["active"] = np.vstack(data["active"])
    data["passive"] = np.vstack(data["passive"])
    data["passive_f"] = np.vstack(data["passive_f"])
    data["vanHove"] = np.vstack(data["vanHove"])
    data["msd"] = np.vstack(data["msd"])
    data["scans"] = np.vstack(data["scans"])
    data["raw_response"] = np.vstack(data["raw_response"])
    data["active"][:,0] = idx_day
    data["passive"][:,0] = idx_day
    data["passive_f"][:,0] = idx_day
    data["vanHove"][:,0] = idx_day
    data["msd"][:,0] = idx_day
    data["scans"][:,0] = idx_day
    
    return data, elapsed_time

def analyse_multiple_days(folders, progressBar = None, calc_vh_msd = True):
    
    start = time.time()
    
    global identifier
    identifier = 0
    
    counter = 0
    
    for folder in folders:
        for subdir, dirs, files in os.walk(folder):
            for dirpath in dirs:
                if "multiple_run_" in dirpath or ("Passive_microrheology" in dirpath):
                    counter += 1
    
    print("Raw approximate total time for analysis of " + str(counter) + " measurements: " + str(counter * 5) + " seconds.")
        
    data = {"active":[], "passive":[], "passive_f":[], "vanHove":[], "msd":[], "_folder":[], "scans":[], "raw_response":[]}
    for idx, folder in enumerate(folders):
        print("day " + str(idx + 1) + " of " + str(len(folders)))
        if progressBar is not None:
            progressBar = (*progressBar, counter)
        temp, elapsed_time = analyse_day(folder,idx_day = idx, progressBar = progressBar, calc_vh_msd = calc_vh_msd)
        print("Approximated total time " + str(elapsed_time*counter) + " seconds. Time elapsed: " + str(time.time() - start) + " seconds")
        data["active"].append(temp["active"])
        data["passive"].append(temp["passive"])
        data["passive_f"].append(temp["passive_f"])
        data["vanHove"].append(temp["vanHove"])
        data["msd"].append(temp["msd"])
        data["_folder"].append(temp["_folder"])
        data["scans"].append(temp["scans"])
        data["raw_response"].append(temp["raw_response"])

    data["active"] = np.vstack(data["active"])
    data["passive"] = np.vstack(data["passive"])
    data["passive_f"] = np.vstack(data["passive_f"])
    data["vanHove"] = np.vstack(data["vanHove"])
    data["msd"] = np.vstack(data["msd"])
    data["scans"] = np.vstack(data["scans"])
    data["raw_response"] = np.vstack(data["raw_response"])

    data["_description"] = {"active": ["day","cell","measurement", "repeat", "frequency", "response in x", 
    "response in y", "G in x", "G in y", "lunam slope", "lunam slope R²", "detector slope",
    "beadsize", "trap", "timestamp", "number", "phase"],"passive": ["day","cell","measurement", "repeat", 
    "binned frequency", "PSD in x", "PSD in y", "lunam slope", "lunam slope R²", "detector slope",
    "timestamp", "number", "phase"], "vanHove": ["day","cell","measurement", "repeat", "bin middle", "x lag 1", "x lag 0.1",  
    "x lag 0.01", "y lag 1", "y lag 0.1", "y lag 0.01"], "msd": ["day","cell","measurement", "repeat", "lagtime", 
    "msd x", "msd y"]}
  
    return data

def get_scans(scan_folder_list):
    
    # scan_list = [dirpath for dirpath, dirnames, filenames in os.walk(folder) if "Scan_stage" in dirpath] 
    scans, time_list = [], []
    for measurement_scans in scan_folder_list:
        scan_list = [dirpath for dirpath, dirnames, filenames in os.walk(measurement_scans) if "Scan_stage" in dirpath] 
        
        for single_scan in scan_list:
            try:
                table = np.loadtxt(single_scan + '\\Scan_stage_metadata.txt',dtype = str)
                time_hms = np.str(table[table[:,0] == "Date_time",1])[-10:-2]
                time_list.append(np.int(time_hms[:2])*3600 + np.int(time_hms[3:5])*60 + np.int(time_hms[-2:]))
                scans.append(np.float(table[table[:,0] == "Detector_slope_y",1]))
            except:
                print("scan could not be loaded")
        
    return np.transpose(np.vstack((np.zeros([3,len(scans)]), time_list, scans)))

#%% evaluation
    
def remove_bad_scan(input_data,threshold = 0.95):
    data = {}
    data["active"] = input_data["active"]*1
    data["passive"] = input_data["passive"]*1
    data["_description"] = input_data["_description"]
    data["_folder"] = input_data["_folder"]
    reduced_active = data["active"][np.imag(data["active"][:,10])>threshold,:]
    reduced_passive = data["passive"][np.imag(data["passive"][:,8])>threshold,:]
    data["active"] = reduced_active
    data["passive"] = reduced_passive
    return data

def assign_phase(data, phase, old_phase = False):
    "assigns the phase specified by argument to corresponding measurement according to day, cell, measurement (in this order)"
            
    for idx, phase_idx in enumerate(phase):
        if old_phase:
            if phase_idx == 3:
                phase_idx = 2
            elif phase_idx == 4:
                phase_idx = 3
            elif phase_idx > 4:
                phase_idx = phase_idx - 2
                    
        data["active"][data["active"][:,-2] == idx,-1] = phase_idx
        data["passive"][data["passive"][:,-2] == idx,-1] = phase_idx
        data["passive_f"][data["passive_f"][:,-2] == idx, -1] = phase_idx
                
    #remove nan
    data["active"] = data["active"][~np.isnan(data["active"][:,6]),:]    
    data["passive"] = data["passive"][~np.isnan(data["passive"][:,6]),:]
    data["passive_f"] = data["passive_f"][~np.isnan(data["passive_f"][:,6])]
        
    
def stats(data):
    "plots statistics for the slopes"
    
    #create indices with single entries for all the slopes
    idx_a,idx_p = [],[]
    for number in np.real(np.unique(data["active"][:,-2])):
        idx_a.append(np.argwhere(data["active"][:,-2] == int(number))[0])
        idx_p.append(np.argwhere(data["active"][:,-2] == int(number))[0])
        
    idx_a = np.hstack(np.array(idx_a))
    idx_p = np.hstack(np.array(idx_p))
        
    plt.figure()
    plt.boxplot([np.real(data["active"][idx_a,9]),np.imag(data["active"][idx_a,9]),np.real(data["passive"][idx_p,7]),np.imag(data["passive"][idx_p,7]),np.real(data["active"][idx_a,11]),np.abs(np.imag(data["active"][idx_a,11])),np.real(data["passive"][idx_p,9]),np.imag(data["passive"][idx_p,9])],labels = ["L x active","L y", "L x passive", "L y", "det x", "det y", "det x", "det y"])
    plt.title("lunam and detector slope")

#    plt.figure()
#    plt.boxplot([np.real(data["passive"][idx_p,8]),np.imag(data["passive"][idx_p,8]),np.real(data["passive"][idx_p,7]),np.imag(data["passive"][idx_p,7])],labels = ["L x", "L y", "det x", "det y"])
#    plt.title("lunam and detector slope")
    plt.figure()
    plt.boxplot([np.real(data["active"][idx_a,9]/np.real(data["active"][idx_a,11])), np.imag(data["active"][idx_a,9]/np.abs(np.imag(data["active"][idx_a,11]))),np.real(data["passive"][idx_p,7]/np.real(data["passive"][idx_p,9])),np.imag(data["passive"][idx_p,7]/np.imag(data["passive"][idx_p,9]))],labels = ["x", "y", "x","y"])
    plt.title("Lunam slopes divided by detector slopes")
    plt.ylabel("ratio")
    plt.xlabel("active       passive")
    
    df = make_dataFrame(data)
    df_dup = df["passive"].drop_duplicates(subset = ["day","cell","measure","repeat","number"])
    short_passive = df_dup.values
    plt.figure()
    plt.plot(short_passive[:,-2],np.transpose([np.real(short_passive[:,9]),np.real(short_passive[:,7])]),'o')
    for day in np.arange(int(np.real(min(data["active"][:,0]))),int(np.real(max(data["active"][:,0])))):
        plt.plot(np.repeat(data["active"][np.where(data["active"][:,0] == day)[0][0],-2],2),[0 ,2.5],'--k')
    plt.legend(["detector", "lunam"])
    plt.title("lunam and detector slopes in x")
    
    plt.figure()
    plt.plot(short_passive[:,-2],np.transpose([np.imag(short_passive[:,9]),np.imag(short_passive[:,7])]),'o')
    for day in np.arange(int(np.real(min(data["active"][:,0]))),int(np.real(max(data["active"][:,0])))):
        plt.plot(np.repeat(data["active"][np.where(data["active"][:,0] == day)[0][0],-2],2),[0 ,2.5],'--k')
    plt.title("lunam and detector slopes in y")
    plt.legend(["detector", "lunam"])
    
    plt.figure()
    plt.boxplot([np.real(data["active"][idx_a,10]),np.imag(data["active"][idx_a,10])],labels = ["x","y"])
    plt.ylabel("R²")
    plt.title("R² for fit for lunam scan")
    
    if "akz_Ta" in data:
        fig, ax1 = plt.subplots()
        
        #t = np.arange(0.01, 10.0, 0.0001)
        data1 = data["akz_Ta"][:,0]
        data2 = data["akz_Ta"][:,1]
        
        color = 'tab:blue'
        ax1.set_xlabel('measurement')
        ax1.set_ylabel('alpha', color = color)
        ax1.set_ylim([0,1])
        lns1 = ax1.plot(data1, 'o',label = 'alpha')
        ax1.tick_params(axis = 'y', labelcolor = color)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:orange'
        ax2.set_ylabel('kappa', color = color)  # we already handled the x-label with ax1
        lns2 = ax2.plot(data2, 'o', color = color,label = 'kappa')
        ax2.set_ylim([-0.00005,0.0005])
        ax2.tick_params(axis = 'y', labelcolor = color)
        
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc = 2)
        fig.tight_layout()  # ot
        
        for day in np.arange(int(np.real(min(data["active"][:,0]))),int(np.real(max(data["active"][:,0])))):
            plt.plot(np.repeat(data["active"][np.where(data["active"][:,0] == day)[0][0],-2],2),[-1 ,0.00045],'--k')
        

def single_phase(data,phase = 0,fixed_slope = 0, mode = 0):
    "plots whatever"
    "mode defines what is to be plotted"
    "modes: "
    "0 (default): plots effective energy"
    "1: plots active/passive response"
    "2: plots shear moduli"
    "3: plots effective energy with fits"
    "4: plots shear moduli with springpot fit"
    
    if phase == 0:
        active = data["df_ap"]
        passive = data["passive"]
        akz = data["akz_Ta"]
        pot = data["s_pot"]
    else:
        active = getPhase(data, phase)
        passive = data["passive"][data["passive"][:,-1] == phase,:]
        akz = data["akz_Ta"][getPhase(data, phase)["number"].to_numpy().astype(int)]
        pot = data["s_pot"][getPhase(data, phase)["number"].to_numpy().astype(int)]
    
    frequency_PSD = np.real(np.unique(data["passive"][:,4]))
    unique_frequencies = active["f"].drop_duplicates()
    
    tmp_resp,tmp_std = [],[]
    for frequency in unique_frequencies:
        idx = active.index[active["f"] == 1]
        tmp_resp.append(np.nanmean(active["response y"][idx]))
        tmp_std.append(np.nanstd(active["response y"][idx]))
        
    #plot errorbar for the unique frequencies with all corresponding responses
    
    #reshape all passive data of one phase to array, multiply with slopes (column 9, (in um, squared because of square in PSD calculation))
    if fixed_slope == 0:
        passive_response = np.real(passive[:,6]/(np.imag(passive[:,9])*10**6)**2)*passive[:,4]*np.pi/(4.2800104364899995e-21)
    else:
        passive_response = np.real(passive[:,6]/(fixed_slope*10**6)**2)*passive[:,4]*np.pi/(4.2800104364899995e-21)
   
    
    noise_indices = [9]
    ind = np.ones(len(frequency_PSD)).astype(bool)
    ind[noise_indices] = 0
    
    # ind = [0,1,2,3,4,5,6,7,8,11,12,13,14,15]
    interpolated_lst = []
    idx_lst = []
    
    if mode == 3:
        single_fig,ax_s = plt.subplots(4,5,sharex = True,figsize = (19,9.5))
        i = j = 0
    elif mode != 0:
        single_fig,ax_s = plt.subplots(4,5,sharey = True,sharex = True,figsize = (19,9.5))
        i = j = 0

    #go through all the measurements from that phase and set this as active_set
        
    for nr, idx in enumerate(active["number"].drop_duplicates()):
        idx = int(np.real(idx))
        active_set = active[active["number"] == idx]
        
        if not active_set.empty:
            
            if mode == 0:
                if np.mod(nr+1,10) == 0:
                    plt.figure()
                    idx_lst = []
            else: 
                #                if np.mod(nr+1,3) == 0:
                    idx_lst = []

            idx_lst.append(idx)
            interpolated_active = active_set["response y"]
            interpolated_lst.append(active_set["response y"])
            
            #calculate the mean of the passive measurements and add to list
            mean_passive = np.real(np.nanmean(np.reshape(passive_response[np.real(passive[:,-2]) == idx],[int(len(passive_response[np.real(passive[:,-2]) == idx])/len(frequency_PSD)),len(frequency_PSD)]),0))
            
            #calculate shift of passive data points to overlap active measurements at ~850 Hz (data point 13)
            
            #plot the effective energy for every measurement (passive response/active response) and add to list Eeff and Eeff_shift
            #for the shifted response
            
            if mode == 0:
                plt.semilogx(frequency_PSD[ind],np.transpose(mean_passive[ind]/interpolated_active[ind]),'-o')
            else:
                if mode == 1:
    #                plt.loglog(np.hstack((frequency_PSD,frequency_PSD)),np.hstack((mean_passive, interpolated_active)),'-o')
                    ax_s[i,j].loglog(frequency_PSD,mean_passive,'-or')
                    ax_s[i,j].loglog(frequency_PSD,interpolated_active,'-ob')
                if mode == 2:
                    ax_s[i,j].loglog(frequency_PSD, active_set["G' y"],'ob')
                    ax_s[i,j].loglog(frequency_PSD, active_set["G'' y"], 'or')
                    ax_s[i,j].loglog(frequency_PSD, np.exp(G_prime(frequency_PSD, *akz[nr,:3])),'--b')
                    ax_s[i,j].loglog(frequency_PSD, np.exp(G_2prime(frequency_PSD, *akz[nr,:3])),'--r')
                    
                if mode == 3: 
                    ax_s[i,j].semilogx(frequency_PSD[ind], active_set["eff E y"][ind],'o')
                    ax_s[i,j].semilogx(np.arange(0.1,10000,0.01), fit_eeff(np.arange(0.1,10000,0.01), *akz[nr,3:5], akz[nr,0], akz[nr,2]))
                    # plt.semilogx(np.arange(.1,10000,.01),fit_eeff(np.arange(.1,10000,.01), *data_new["akz_Ta"][n,3:5], data_new["akz_Ta"][n,0], data_new["akz_Ta"][n,2]))
                    
                if mode == 4:
                    ax_s[i,j].loglog(frequency_PSD, active_set["G' y"],'ob')
                    ax_s[i,j].loglog(frequency_PSD, active_set["G'' y"], 'or')
                    ax_s[i,j].loglog(frequency_PSD, np.exp(G1_pot(frequency_PSD, *pot[nr,:4])),'--b')
                    ax_s[i,j].loglog(frequency_PSD, np.exp(G2_pot(frequency_PSD, *pot[nr,:4])),'--r')
                    
                ax_s[i,j].grid()
                ax_s[i,j].legend([str(idx_lst) + "; phase: " + str(int(active_set["phase"].drop_duplicates().to_numpy()))])
                # ax_s[i,j].legend(idx_lst)
                
                j+=1
                if j == 5:
                    i+=1
                    j = 0  
                    if i == 4:
                        i = 0
                        if mode == 3:
                            single_fig, ax_s = plt.subplots(4,5,sharex = True, figsize = (19,9.5))
                        else:
                            single_fig,ax_s = plt.subplots(4,5,sharey = True,sharex = True,figsize = (19,9.5))
            
            plt.legend(idx_lst)
    
def delete_number(data,number):
    "deletes entry with given number from dataset"
    "number to delete can be single value or array"
    
    #copy dataset to new dictionary
    output = {}
    output["active"] = data["active"]*[1]
    output["passive"] = data["passive"]*[1]
    if "passive_f" in data:
        output["passive_f"] = data["passive_f"]*[1]
    if "_description" in data:
        output["_description"] = data["_description"]
    if "_folder" in data:
        output["_folder"] = data["_folder"]
    output["akz_Ta"] = data["akz_Ta"]*[1]
    output["msd"] = data["msd"]*[1]
    output["vanHove"] = data["vanHove"]*[1]
    
    
    #delete all data with defined number
    for num in number:
        output["active"] = np.delete(output["active"],np.where(output["active"][:,-2] == num),0)
        output["passive"] = np.delete(output["passive"],np.where(output["passive"][:,-2] == num),0)
        if "passive_f" in data:
            output["passive_f"] = np.delete(output["passive_f"],np.where(output["passive"][:,-2] == num),0)
            output["akz_Ta"] = np.delete(output["akz_Ta"],np.where(output["akz_Ta"][:,-1] == num),0)
            output["msd"] = np.delete(output["msd"],np.where(output["msd"][:,-1] == num),0)
            output["vanHove"] = np.delete(output["vanHove"],np.where(output["vanHove"][:,-1] == num),0)
    
    return output

def interpolate_active(data, fixed_slope = 0, f = 0, chosen_passive = None, progressBar = None, calc_akz = True, 
                       calc_spot = True, calc_eeff_pl = True, overwrite = False, skip_real = 3):
    "interpolates active response to binned frequencies of passive rheology"
    "calculates mean of passive rheology repeats"
    "calculates effective Energy"
    
    #decide whether to use x (0) or y (1) direction 
    x_or_y = 1
    #create empty array for interpolated data of active rheology
    active_array = []
    passive_array = []
    passive_array_f = []
    PSD_fits, psd_fits2 = [], []
    akz_Ta = []
    shift = []
    R2_G, R2_Ta = [], []
    power_param = []
    pot = []
    pot2= []
    eeff_respFit = []
    pl_fits = []
    
    data_scan = interpolateScans(data)
    if not "correct" in data:
        filter_bad(data)
    
    #go through all the measurements
    for idx, identifier in enumerate(np.unique(data["active"][:,-2])):
    # for idx, identifier in enumerate(np.arange(125,140)):
        # print(identifier)
        
        #get frequency and length of frequency
        if isinstance(f,np.ndarray):            
            frequencies = np.real(f)
        else:
            frequencies = np.real(np.unique(np.real(data["passive"][data["passive"][:,-2] == identifier,4])))
        l = len(frequencies)
                
        active_set = data["active"][data["active"][:,-2] == identifier,:]
        correct = data["correct"][data["correct"][:,2] == identifier, 0][::-1].astype(bool)
        beadsize = np.real(active_set[:,-5])[0]
        
        #kill imaginary part with values smaller than 10**-10
        too_small = np.abs(np.imag(active_set[:,6])) < 10**-10
        #make function to interpolate active response for frequencies of passive measurement in x and y
        #_r for real
        inter_func_x = scipy.interpolate.interp1d(np.log(np.real(active_set[:,4])[np.logical_and(np.imag(active_set[:,5]) != 0, np.abs(np.imag(active_set[:,5])) > 10**-10)]),
                       np.log(np.abs(np.imag(active_set[:,5])[np.logical_and(np.imag(active_set[:,5]) != 0, np.abs(np.imag(active_set[:,5])) > 10**-10)])),kind = 'linear',bounds_error = False)
        inter_func_y = scipy.interpolate.interp1d(np.log(np.real(active_set[:,4])[np.logical_and(np.imag(active_set[:,6]) != 0, np.abs(np.imag(active_set[:,6])) > 10**-10)]),
                       np.log(np.abs(np.imag(active_set[:,6])[np.logical_and(np.imag(active_set[:,6]) != 0, np.abs(np.imag(active_set[:,6])) > 10**-10)])),kind = 'linear',bounds_error = False)
        
        inter_func_x_r = scipy.interpolate.interp1d(np.log(np.real(active_set[:,4])),np.log(np.abs(np.real(active_set[:,5]))),kind = 'linear',bounds_error = False)
        inter_func_y_r = scipy.interpolate.interp1d(np.log(np.real(active_set[:,4])),np.log(np.abs(np.real(active_set[:,6]))),kind = 'linear',bounds_error = False)
        
        try:
            inter_func_y_corr = scipy.interpolate.interp1d(np.log(np.real(active_set[correct,4])),np.log(np.abs(np.imag(active_set[correct,6]))),kind = 'linear',bounds_error = False)
            inter_func_y_r_corr = scipy.interpolate.interp1d(np.log(np.real(active_set[correct,4])),np.log(np.abs(np.real(active_set[correct,6]))),kind = 'linear',bounds_error = False)
        except:
            inter_func_y_corr = inter_func_y
            inter_func_y_r_corr = inter_func_y_r
        
        #interpolate on frequencies of passive rheology
        interpolated_x = np.exp(inter_func_x(np.log(frequencies)))
        interpolated_y = np.exp(inter_func_y(np.log(frequencies)))
        interpolated_x_r = np.exp(inter_func_x_r(np.log(frequencies)))
        interpolated_y_r = np.exp(inter_func_y_r(np.log(frequencies)))
        
        interpolated_y_c = np.exp(inter_func_y_corr(np.log(frequencies)))
        interpolated_y_r_c = np.exp(inter_func_y_r_corr(np.log(frequencies)))
        
        
        scan_set = data_scan[data_scan[:,-2] == identifier, :]
        inter_func_scan = scipy.interpolate.interp1d(np.log(np.real(scan_set[:,4])[np.logical_and(np.imag(scan_set[:,7]) != 0, np.abs(np.imag(scan_set[:,7])) > 10**-10)]),
                   np.log(np.abs(np.imag(scan_set[:,7])[np.logical_and(np.imag(scan_set[:,7]) != 0, np.abs(np.imag(scan_set[:,7])) > 10**-10)])),kind = 'linear',bounds_error = False)
        inter_func_scan_r = scipy.interpolate.interp1d(np.log(np.real(scan_set[:,4])),np.log(np.abs(np.real(scan_set[:,7]))),kind = 'linear',bounds_error = False)
        interpolated_y_scan = np.exp(inter_func_scan(np.log(frequencies)))
        interpolated_y_scan_r = np.exp(inter_func_scan_r(np.log(frequencies)))
        
        inter_reshape = np.hstack([np.reshape(np.tile(scan_set[0,0:4],l),[l,4]), np.transpose(np.vstack((frequencies, interpolated_x, interpolated_y, interpolated_y_scan, interpolated_y_c))) ,
                                   np.transpose(1/(3*np.pi*beadsize*np.vstack((interpolated_x_r+1J*interpolated_x,interpolated_y_r+1J*interpolated_y, interpolated_y_scan_r + 1J*interpolated_y_scan, interpolated_y_r_c + 1J*interpolated_y_c)))),
                                   np.repeat(scan_set[0,11], l)[:,None], np.repeat(scan_set[-1,12], l)[:,None], np.reshape(np.tile(scan_set[0,13:],l),[l,5])])
        #reshape and fill with values of original data
        # inter_reshape = np.hstack([np.reshape(np.tile(active_set[0,0:4],l),[l,4]), np.transpose(np.vstack((frequencies, interpolated_x, interpolated_y))),np.transpose(1/(3*np.pi*beadsize*np.vstack((interpolated_x_r+1J*interpolated_x,interpolated_y_r+1J*interpolated_y)))),np.reshape(np.tile(active_set[0,9:],l),[l,8])])
        active_array.append(inter_reshape)
        
        ##passive means of single measurements
        current_passive_measure = data["passive"][data["passive"][:,-2] == identifier,:] 
        current_passive_f = data["passive_f"][data["passive_f"][:,-2] == identifier,:]
        
        #check if passive measurement has been repeated and calculate mean if that's the case
        if fixed_slope == 0:
            slope_x = ((np.real(current_passive_measure[:,9])*10**6)**2)[0]
            slope_y = ((np.imag(current_passive_measure[:,9])*10**6)**2)[0]
        else:
            slope_x = slope_y = (fixed_slope*10**6)**2
            
        passive_f_freq = np.unique(data["passive_f"][data["passive_f"][:,-2]== identifier, 4])
        l_f = len(passive_f_freq)
        #calculate mean if there are multiple passive measurements
        if chosen_passive is not None:
            used_passive = ~np.isin(current_passive_measure[:,3], chosen_passive[chosen_passive[:,5] == identifier, 0:5] - 1)    
            passive_mean_x = np.nanmean(np.reshape(np.real(current_passive_measure[:,5][used_passive])/slope_x,[int(len(current_passive_measure[used_passive])/l),l]),0)
            passive_mean_y = np.nanmean(np.reshape(np.real(current_passive_measure[:,6][used_passive])/slope_y,[int(len(current_passive_measure[used_passive])/l),l]),0)
            
            used_passive_f = ~np.isin(current_passive_f[:,3], chosen_passive[chosen_passive[:,5] == identifier, 0:5] -1)
            passive_mean_x_f = np.nanmean(np.reshape(np.real(current_passive_f[:,5][used_passive_f])/slope_x,[int(len(current_passive_f[used_passive_f])/l_f),l_f]),0)
            passive_mean_y_f = np.nanmean(np.reshape(np.real(current_passive_f[:,6][used_passive_f])/slope_y,[int(len(current_passive_f[used_passive_f])/l_f),l_f]),0)
        else:
            passive_mean_x = np.nanmean(np.reshape(np.real(current_passive_measure[:,5])/slope_x,[int(len(current_passive_measure)/l),l]),0)
            passive_mean_y = np.nanmean(np.reshape(np.real(current_passive_measure[:,6])/slope_y,[int(len(current_passive_measure)/l),l]),0)
            
            passive_mean_x_f = np.nanmean(np.reshape(np.real(current_passive_f[:,5])/slope_x,[int(len(current_passive_f)/l_f),l_f]),0)
            passive_mean_y_f = np.nanmean(np.reshape(np.real(current_passive_f[:,6])/slope_y,[int(len(current_passive_f)/l_f),l_f]),0)

        
        
        #append to list and calculate response
        passive_array.append(np.transpose(np.vstack((passive_mean_x,passive_mean_y))*np.pi*frequencies/(scipy.constants.k *310)))
        passive_array_f.append(np.transpose(np.vstack((passive_mean_x_f,passive_mean_y_f))*np.pi*passive_f_freq/(scipy.constants.k *310)))
        
        
        ### shift for effective energy
        shift.append(np.tile(np.mean(np.divide(active_array[-1][:,5:7], passive_array[-1])[12:15],0), (l,1)))
        
        ###fit different models
        # skip_real = 3
        skip_imag = skip_real
        ##fit G; skip the 4 highest frequencies of real part
        x = np.real(np.hstack([active_set[:,4][skip_real:], active_set[:,4][skip_imag:][np.abs(np.imag(active_set[:,8][skip_imag:])) > 10**-10]]))
        y = np.hstack([np.log(np.abs(np.real(active_set[:,8][skip_real:]))), np.log(np.abs(np.imag(active_set[:,8][skip_imag:][np.abs(np.imag(active_set[:,8][skip_imag:])) > 10**-10])))])
        
        # correct = data["correct"][data["correct"][:,-1] == identifier][:-skip_real, 0].astype(bool)
        # x = np.real(np.hstack([active_set[:,4][skip_real:][correct], active_set[:,4][skip_imag:][correct][np.abs(np.imag(active_set[:,8][skip_imag:][correct])) > 10**-10]]))
        # y = np.hstack([np.log(np.abs(np.real(active_set[:,8][skip_real:][correct]))), np.log(np.abs(np.imag(active_set[:,8][skip_imag:][correct][np.abs(np.imag(active_set[:,8][skip_imag:][correct])) > 10**-10])))])
  
        length = len(active_set[:,4][skip_real:])
        
        ### fit a power-law to response from passive
        no_noise = np.array([1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0]).astype("bool")
        try:
            PSD_fits.append(scipy.optimize.curve_fit(power_law_log2, frequencies[no_noise], np.log(passive_array[-1].T[1][no_noise]), maxfev=5000)[0])
            # psd_fit2.append(scipy.optimize.curve_fit(double_plaw, frequencies[no_noise], np.log(passive_array[-1].T[1][no_noise]), maxfev=5000)[0])
        except:
            PSD_fits.append([0,0])
            
        try:
            psd_fits2.append(scipy.optimize.curve_fit(double_plaw, frequencies[no_noise], np.log(passive_array[-1].T[1][no_noise]), maxfev=5000)[0])
        except:
            psd_fits2.append([0,0,0,0])
        
        ##fit akz and save in akz_tmp; calculate R² of akz fit (Whylies model)
        if ("akz_Ta" not in data or overwrite) and calc_akz:
            akz_tmp = []
            # try:
            akz_tmp, pcov =  scipy.optimize.curve_fit(lambda f, a, k, z: G12prime(f, a, k, z, beadsize, length), x, y, p0 = [.6,2e-5,.55], maxfev = 10000)
            R2_G =gof(x,y, np.real(np.hstack((akz_tmp, beadsize, len(active_set[:,4][skip_real:])))), G12prime)
            # else:
                # print("no fit for Whylie's model possible for data set number " + str(int(identifier)))
                # akz_tmp = 
    
    #        passive_array_y.append(passive_mean_y*np.pi*frequency_PSD/(4.2800104364899995e-21))
                        
            #effective energy
            eeff = np.real(passive_array[-1]/active_array[-1][:,5:7])
            eeff_shift = np.real(shift[-1]*passive_array[-1]/active_array[-1][:,5:7])
            #set infinite values to 10**10
            eeff[np.isinf(eeff)] = 10**10
            eeff_shift[np.isinf(eeff_shift)] = 10**10
            #fit curve for T_A
            
            #restriction of fitting area to 
            noise_indices = [] #9
            index = np.ones(l).astype(bool)
            index[noise_indices] = 0
            # index = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]
            num_eeff = ~np.isnan(eeff[index][:,x_or_y])
            num_eeff_shift = ~np.isnan(eeff_shift[index][:,x_or_y])
            ### there's a discrepancy between Ta if starting conditions are used or not
            # Ta_tmp,pcov = scipy.optimize.curve_fit(lambda f, Ta, tau: fit_eeff(f, Ta, tau, akz_tmp[0], akz_tmp[2]),frequencies[index][num_eeff],eeff[index][num_eeff,x_or_y],p0 = [20,0.0003],maxfev = 50000,bounds = ([-np.inf,0],np.inf))
            try:
                Ta_tmp,pcov = scipy.optimize.curve_fit(lambda f, Ta, tau: fit_eeff(f, Ta, tau, akz_tmp[0], akz_tmp[2]),np.real(frequencies[index][num_eeff]),eeff[index][num_eeff,x_or_y],maxfev = 50000,bounds = ([-np.inf,0],np.inf))
            except:
                print("no fit for TA for dataset number " + str(int(identifier)))
                Ta_tmp = [0, 0]
                
            try:
                Ta_tmp_shift, pcov = scipy.optimize.curve_fit(lambda f, Ta, tau: fit_eeff(f, Ta, tau, akz_tmp[0], akz_tmp[2]), np.real(frequencies[index][num_eeff_shift]), eeff_shift[index][num_eeff_shift,x_or_y], maxfev = 50000, bounds = ([-np.inf,0], np.inf))
            except:
                print("no fit for TA with shifted data for dataset number " + str(int(identifier)))
                Ta_tmp_shift = [0, 0]
            # gof(frequencies, eeff[:,1], np.hstack((*Ta_tmp, akz_tmp[0], akz_tmp[2])), fit_eeff)
            # R2_Ta.append(gof(f, Ta, tau: fit_eeff(f, *Ta_tmp, akz_tmp[0], akz_tmp[2]), fit_eeff))
    
            #append alpha, kappa, zeta and Ta and tau
            akz_Ta.append(np.hstack([akz_tmp, Ta_tmp, Ta_tmp_shift, R2_G, identifier]))

        if "power" not in data:
            # both_power, pcov = scipy.optimize.curve_fit(lambda f, a, b, c: power2(f, a, b, c, length), x, y, maxfev = 10000, bounds = ([-np.inf, -np.inf, 0], np.inf))      
            power_G1, pcov = scipy.optimize.curve_fit(G_power, np.real(active_set[:,4][skip_real:]), np.log(np.abs(np.real(active_set[:,8][skip_real:]))), maxfev = 5000, bounds = ([-np.inf, -np.inf, 0], np.inf))
            power_G2, pcov = scipy.optimize.curve_fit(G_power, np.real(active_set[:,4][np.abs(np.imag(active_set[:,8])) > 10**-10]), np.log(np.abs(np.imag(active_set[:,8][np.abs(np.imag(active_set[:,8])) > 10**-10]))), maxfev = 5000, bounds = ([-np.inf, -np.inf, 0], np.inf))
        
            r_g1 = gof(np.real(active_set[:,4][skip_real:]), np.log(np.abs(np.real(active_set[:,8][skip_real:]))), power_G1, G_power)                                                                                                                                          
            r_g2 = gof(np.real(active_set[:,4][np.abs(np.imag(active_set[:,8])) > 10**-10]), np.log(np.abs(np.imag(active_set[:,8][np.abs(np.imag(active_set[:,8])) > 10**-10]))), power_G2, G_power)                                                                                                                                          
            # r_g12 = gof(x, y, np.real(np.hstack((both_power, length))), power2)          
              
            power2Para_G1, pcov = scipy.optimize.curve_fit(G_power_2_para, np.real(active_set[:,4][skip_real:]), np.log(np.abs(np.real(active_set[:,8][skip_real:]))), maxfev = 5000, bounds = ([-np.inf, 0], np.inf))
            power2Para_G2, pcov = scipy.optimize.curve_fit(G_power_2_para, np.real(active_set[:,4][np.abs(np.imag(active_set[:,8])) > 10**-10]), np.log(np.abs(np.imag(active_set[:,8][np.abs(np.imag(active_set[:,8])) > 10**-10]))), maxfev = 5000, bounds = ([-np.inf, 0], np.inf))
            r2Para_g1 = gof(np.real(active_set[:,4][skip_real:]), np.log(np.abs(np.real(active_set[:,8][skip_real:]))), power2Para_G1, G_power_2_para)                                                                                                                                          
            r2Para_g2 = gof(np.real(active_set[:,4][np.abs(np.imag(active_set[:,8])) > 10**-10]), np.log(np.abs(np.imag(active_set[:,8][np.abs(np.imag(active_set[:,8])) > 10**-10]))), power2Para_G2, G_power_2_para)                                                                                                                                          
          
            # power_param.append(np.hstack((power_G1, power_G2, both_power, r_g1, r_g2, r_g12, power2Para_G1, power2Para_G2, r2Para_g1, r2Para_g2)))
            power_param.append(np.hstack((power_G1, power_G2, r_g1, r_g2, power2Para_G1, power2Para_G2, r2Para_g1, r2Para_g2)))
        
        if ("s_pot" not in data or overwrite) and calc_spot: 
            try:
                pot_tmp, pcov = scipy.optimize.curve_fit(lambda f, A, B, alpha, beta: springpot(f, A, B, alpha, beta, length), x, y, p0 = [11, 1, 0.3, 0.9], bounds = ([0, 0, -np.inf, -np.inf], np.inf), maxfev = 50000)
                r_pot = gof(x, y, np.real(np.hstack((pot_tmp, len(active_set[:,4][skip_real:])))), springpot)
                
                #switch alpha and beta if beta is smaller than alpha
                if pot_tmp[2] > pot_tmp[3]:
                    pot_tmp = [pot_tmp[1], pot_tmp[0], pot_tmp[3], pot_tmp[2]]
            except:
                print("no fit for springpot model for dataset number: " + str(int(identifier)))
                pot_tmp = [0, 0, 0, 0]
                r_pot = 0
              
            # active_array[-1][0][:,10]
            correct = data["correct"][data["correct"][:,-1] == identifier][:-skip_real, 0].astype(bool)[::-1]
            correct_len = np.sum(correct)
            x = np.real(np.hstack([active_set[:,4][skip_real:][correct], active_set[:,4][skip_imag:][correct][np.abs(np.imag(active_set[:,8][skip_imag:][correct])) > 10**-10]]))
            y = np.hstack([np.log(np.abs(np.real(active_set[:,8][skip_real:][correct]))), np.log(np.abs(np.imag(active_set[:,8][skip_imag:][correct][np.abs(np.imag(active_set[:,8][skip_imag:][correct])) > 10**-10])))])
            try:
                pot_tmp2, pcov = scipy.optimize.curve_fit(lambda f, A, B, alpha, beta: springpot(f, A, B, alpha, beta, correct_len), x, y, p0 = [11, 1, 0.3, 0.9], bounds = ([0, 0, -np.inf, -np.inf], np.inf), maxfev = 50000)
                r_pot2 = gof(x, y, np.real(np.hstack((pot_tmp, len(active_set[:,4][skip_real:])))), springpot)
                
                #switch alpha and beta if beta is smaller than alpha
                if pot_tmp2[2] > pot_tmp2[3]:
                    pot_tmp2 = [pot_tmp2[1], pot_tmp2[0], pot_tmp2[3], pot_tmp2[2]]
            except:
                print("no fit for springpot model for dataset number: " + str(int(identifier)))
                pot_tmp2 = [0, 0, 0, 0]
                r_pot2 = 0
            pot2.append(np.hstack((pot_tmp2, r_pot2)))
    
            pot.append(np.hstack((pot_tmp, r_pot)))
    
            


        if "s_pot2" in data:
            g = np.reshape(np.exp(springpot(np.tile(frequencies,2), *data["s_pot2"][int(identifier),:4], 16)),[2,16])
            g_star = g[0] + 1J*g[1]
            #true divide error because of nans in real part of g_star
            response_fit = np.abs(np.imag(1/(3 * np.pi * beadsize * g_star)))
            eeff_respFit.append(passive_array[-1][:,1]/response_fit)
            


            if ("eeff_pl" not in data or overwrite) and calc_eeff_pl:
                # na = ~np.isnan(np.log(phase_data[phase_data["number"] == number]["eff E y fit"]))
                tmp_e = passive_array[-1][:,1]/active_array[-1][:,6]
                na = ~np.isnan(np.log(tmp_e))
                na_fit = ~np.isnan(np.log(eeff_respFit[-1]))
                #exclude 46.5 and 100 Hz
                na[8:11] = False
                na[-1] = False
                na_fit[8:11] = False
                na_fit[-1] = False
                if np.isin(active_set[0,-1], [2,3,4,5,6]):
                    na[0] = False
                    na_fit[0] = False
                    
                eeff_shift = np.real(shift[-1][:,1]*passive_array[-1][:,1]/response_fit)
                num_eeff_shift = ~np.isnan(eeff_shift)
                num_eeff_shift[8:10] = False
                num_eeff_shift[-1] = False
                
                try:
                    tmp_pl = scipy.optimize.curve_fit(power_law_log, frequencies[na], np.log(tmp_e)[na], maxfev = 50000)[0]
                    tmp_pl_fit = scipy.optimize.curve_fit(power_law_log, frequencies[na_fit], np.log(eeff_respFit[-1])[na_fit], maxfev = 50000)[0]
                    tmp_pl_shift = scipy.optimize.curve_fit(power_law_log, frequencies[num_eeff_shift], np.log(eeff_shift[num_eeff_shift]), maxfev = 50000)[0]
                    r_pl = gof(frequencies[na], np.log(tmp_e)[na], tmp_pl, power_law_log)
                    r_pl_fit = gof(frequencies[na_fit], np.log(eeff_respFit[-1])[na_fit], tmp_pl_fit, power_law_log)
                except:
                    tmp_pl = [0,0]
                    tmp_pl_fit = [0,0]
                    r_pl = r_pl_fit = 0
                

                try:
                    tmpE = scipy.optimize.curve_fit(lambda f, Tcv, v: fit_new_eeff(f, Tcv, v, *data["s_pot2"][int(identifier), :4]), frequencies[na_fit], (eeff_respFit[-1])[na_fit], p0=[30,1], maxfev = 50000)[0]
                    tmpE = scipy.optimize.curve_fit(lambda f, Tcv, v: fit_new_eeff2(f, Tcv, v, data["s_pot2"][int(identifier), 0], data["s_pot2"][int(identifier), 2]), frequencies[na_fit], (eeff_respFit[-1])[na_fit], p0=[30,1], maxfev = 50000)[0]
                    r_tmpE = gof(frequencies[na_fit], eeff_respFit[-1][na_fit], np.hstack((tmpE, data["s_pot2"][int(identifier), :4])), fit_new_eeff)
                    tmpE2 = scipy.optimize.curve_fit(lambda f, Tcv, v: fit_new_eeff(f, Tcv, v, *data["s_pot2"][int(identifier), :4]), frequencies[num_eeff_shift], eeff_shift[num_eeff_shift], p0=[30,1], maxfev = 50000)[0]
                    r_tmpE2 = gof(frequencies[num_eeff_shift], eeff_shift[num_eeff_shift], np.hstack((tmpE2, data["s_pot2"][int(identifier), :4])), fit_new_eeff)
                except:
                    tmpE = [0, 0]    
                    r_tmpE = 0
                    r_tmpE2
                
                # plt.figure()
                # plt.semilogx(frequencies, tmp_e, 'o')
                # plt.semilogx(frequencies, eeff_respFit[-1], '-')
                # plt.semilogx(frequencies, fit_new_eeff(frequencies, *tmpE, *data["s_pot2"][int(identifier), :4]))
                # plt.semilogx(frequencies, fit_new_eeff(frequencies, *tmpE2, *data["s_pot2"][int(identifier), :4]))
                # plt.semilogx(frequencies, np.exp(power_law_log(frequencies, *tmp_pl_fit)))
                # plt.semilogx(frequencies, np.repeat(1, len(frequencies)), '--k')
                # plt.legend(["data", "data from spot fit", "model fit", "data spot shift" , "power law", "1 kbT"])
                
                pl_fits.append(np.real(np.hstack((tmp_pl, tmp_pl_fit, tmpE, tmpE2, tmp_pl_shift, r_pl, r_pl_fit, r_tmpE, r_tmpE2))))
                
        if progressBar is not None:
            Bar, lblBar = progressBar
            Bar.setValue(idx / len(np.unique(data["active"][:,-2])) * 100)
            lblBar.setText("interpolating data...")
            # lblProgress.setText(str(int(identifier)) + "/" + str(total))
            QtWidgets.QApplication.processEvents()
        
    #stack data into active_inter and add to input data
    
    if ("akz_Ta" not in data or overwrite) and calc_akz:   
        akz_Ta = np.vstack(akz_Ta)
        data["akz_Ta"] = np.real(akz_Ta)
        
    if ("s_pot" not in data or overwrite) and calc_spot:
        pot = np.vstack(pot)
        data["s_pot"] = pot
        pot2 = np.vstack(pot2)
        data["s_pot2"] = pot2
        
    if ("eeff_pl" not in data or overwrite) and calc_eeff_pl:
        data["eeff_pl"] = np.vstack(pl_fits)


    output_active = np.vstack(active_array)
    output_passive = np.vstack(passive_array)
    output_passive_f = np.vstack(passive_array_f)
    
    PSD_fits = np.vstack((PSD_fits))
    psd_fits2 = np.vstack((psd_fits2))

    if "power" not in data:
        power_param = np.vstack(power_param)
        data["power"] = power_param
     
    eeff_all = output_passive/output_active[:,5:7]
    eeff_all[np.isinf(eeff_all)] = 10**10
    eeff_scan =  output_passive[:,1]/output_active[:,7]
    eeff_scan[np.isinf(eeff_scan)] = 10**10
    eeff_corr = output_passive[:,1]/output_active[:,8]
    eeff_shifted = np.real(output_passive[:,1]*np.vstack((shift))[:,1]/output_active[:,8])
    
    if eeff_respFit:
        eeff_fit = np.real(np.hstack(eeff_respFit))
    else:
        eeff_fit = np.real(eeff_all[:,1])


    #create data input with entries for day, cell, measurement, interpolated active, mean of passive, effective energy (all in x and y), identifier, phase
    ap = np.real(np.hstack((output_active[:,0:3],output_active[:,4:9],output_passive, eeff_all, eeff_scan[:, None], eeff_corr[:,None],np.abs(np.real(output_active[:,9:13])),np.abs(np.imag(output_active[:,9:13])),np.real(output_active[:,13:15]), np.imag(output_active[:,13:15]), np.vstack((shift)), output_active[:,15][:,None], output_active[:,17:])))
    data["df_ap"] = pd.DataFrame({"day":ap[:,0], "cell":ap[:,1],
                     "measure":ap[:,2], "f":ap[:,3], "response x":ap[:,4],
                     "response y":ap[:,5], "response y scan":ap[:,6], "response y corrected":ap[:,7], "PSD x":ap[:,8], "PSD y":ap[:,9],
                     "eff E x":ap[:,10], "eff E y":ap[:,11], "eff E y scan":ap[:,12], "eff E y fit": eeff_fit, "eff E y corr": ap[:,13],
                     "eff E y shift":eeff_shifted,
                     "G' x":ap[:,14],"G' y":ap[:,15], "G' y scan":ap[:,16], "G' y corr":ap[:,17],
                     "G'' x":ap[:,18], "G'' y":ap[:,19], "G'' y scan":ap[:,20], "G'' y corr":ap[:,21],
                     "det slope x":ap[:,22], "det slope y":ap[:,23], "det slope x scan":ap[:,24],
                     "det slope y scan":ap[:,25], "shift x":ap[:,26], "shift y":ap[:,27], "beadsize":ap[:,28],
                     "time":ap[:,29], "number":ap[:,30], "phase":ap[:,31]})
    data["fitParameters"] = pd.DataFrame({"A": data["s_pot2"][:,0], "B": data["s_pot2"][:,1], "alpha": data["s_pot2"][:,2],
                                          "beta": data["s_pot2"][:,3], "R2 spring": data["s_pot2"][:,4], "pre": data["eeff_pl"][:,2],
                                          "ex": data["eeff_pl"][:,3], "pre shift": data["eeff_pl"][:,8], "ex shift": data["eeff_pl"][:,9], "psd pre": PSD_fits[:,0], "psd ex": PSD_fits[:,1],
                                          "pre1": psd_fits2[:,0], "pre2": psd_fits2[:,1], "ex1": psd_fits2[:,2], "ex2": psd_fits2[:,3],
                                          "number":np.arange(len(data["s_pot2"][:,0]))})
    
    data["_description"]["power"] = ["a G'", "b G'", "c G'", "a G''", "b G''", "c G''", "a G' and G''", "b G' and G''", "c G' and G''", "R² G'", "R² G''", "R² G' and G''"]
    data["_description"]["akz"] = []
    data["_description"]["s_pot"] = ["A", "B", "alpha", "beta", "R square"]
    return output_active, akz_Ta, pot, output_passive_f, psd_fits2

def interpolateScans(data):
    output = []
    
    if data["scans"].any():
        scans = np.hstack((data["scans"], np.arange(len(data["scans"]))[:,None]))
        #get difference of days, cells and measure of scans
        diff = np.vstack((np.diff(scans[:,:4], axis = 0), [10,10,10,10]))
        #check for scans on same day in same cell
        timelapseScans = np.all(diff[:,:3] == [0,0,1], axis = 1)
        #restrict to cells that have multiple measurements (by checking if there is more than 1 measurement)
        cellScans = np.all(diff[:,:3] == [0,0,0], axis = 1)
        arr = np.ones(sum(cellScans)).astype(bool)
        pos = []
        new_arr_time = []
        
        if timelapseScans.any():
            timelapseScans2 = scans[timelapseScans][np.all(np.vstack((np.diff(scans[timelapseScans,:2], axis = 0), [1,1])) == [0,0], axis = 1),:2]
            days_cell_timelapse = pd.DataFrame({"day": timelapseScans2[:,0], "cell": timelapseScans2[:,1]}).drop_duplicates().to_numpy()
       
        #days and cells number of cells with timelapse acquisition
     
    
            for dayCell in days_cell_timelapse:
                #all scans for that day/cell
                current_scans = scans[np.all(scans[:,0:2] == dayCell, 1)]
                
                #go through every scan and calculate new response
                for scan in current_scans[1:]:              
                    current_set = data["active"][np.all(data["active"][:,0:3] == scan[:3], 1)]
                    inter = scipy.interpolate.interp1d([np.real(current_set[0,14]), scan[3]], [current_scans[0, 4], scan[4]])
                    times = np.hstack((np.real(current_set[:,-3]), scan[3]))
                    try:
                        slopes = inter(times[:-1] + np.diff(times)/2)
                    except: 
                        print(scan)
                        
                    new_response = current_set[:,6] * (np.abs(np.imag(current_set[:,11]))/slopes)
                    #create new array for dictionary
                    new_arr_time.append(np.hstack((current_set[:,:7], new_response[:,None], current_set[:,7:9], current_set[:,8][:,None], current_set[:,11][:,None], (slopes-1J*slopes)[:,None], current_set[:,12:])))
                    
                try:
                    pos.append(np.where(np.all(scans[cellScans, :2] == dayCell, axis = 1))[0][0])
                except:
                    print("no") 
          
            #convert to array
            if len(new_arr_time) > 1:
                new_arr_time = np.vstack(new_arr_time)
            arr[pos] = 0
    
        scansBeforeAfter = scans[cellScans][arr]
        scanIdx = scansBeforeAfter[:,5].astype(int)
    
        new_arr_drug = []
        
        for i in range(len(scansBeforeAfter)):    
            current_set = data["active"][np.all(data["active"][:,0:3] == scansBeforeAfter[i,:3], axis = 1)]
            #take scan as first interpolation point
            # inter = scipy.interpolate.interp1d(scans[nr[i]:nr[i] + 2, 3], scans[nr[i]:nr[i] + 2, 4])
            #take first measurement as first interpolation point
            inter = scipy.interpolate.interp1d([np.real(current_set[0,14]), scans[scanIdx[i] + 1, 3]], scans[scanIdx[i]:scanIdx[i] + 2, 4])
            times = np.hstack((np.real(current_set[:,-3]), scans[scanIdx[i] + 1, 3]))
            try:
                slopes = inter(times[:-1] + np.diff(times)/2)
            except:
                slopes = np.repeat(1,len(times[:-1]))
                print("slope not got " + str(i))
                
            
            new_response = current_set[:,6] * (np.abs(np.imag(current_set[:,11]))/slopes)
            new_arr_drug.append(np.hstack((current_set[:,:7], new_response[:,None], current_set[:,7:9], current_set[:,8][:,None], current_set[:,11][:,None], (slopes-1J*slopes)[:,None], current_set[:,12:])))
            
        new_arr_drug = np.vstack(new_arr_drug)
        
            #create new array with 2 scans if available, else use single scan only

        for idx in range(int(max(data["active"][:,-2]))+1):
            if idx in new_arr_drug[:,-2]:
                output.append(new_arr_drug[new_arr_drug[:,-2] == idx, :])
            elif idx in new_arr_time[:,-2]:
                output.append(new_arr_time[new_arr_time[:,-2] == idx, :])
            else:
                current_set = data["active"][data["active"][:,-2] == idx, :]
                output.append(np.hstack((current_set[:,:7], current_set[:,6][:,None], current_set[:,7:9], current_set[:,8][:,None], current_set[:,11][:,None], current_set[:,11][:,None], current_set[:,12:])))
    else:
        for idx in range(int(max(data["active"][:,-2]))+1):
            current_set = data["active"][data["active"][:,-2] == idx, :]
            output.append(np.hstack((current_set[:,:7], current_set[:,6][:,None], current_set[:,7:9], current_set[:,8][:,None], current_set[:,11][:,None], current_set[:,11][:,None], current_set[:,12:])))

    return np.vstack(output)


def filter_bad(data, std_times = 1, use_fix = False, progressBar = None):
    
    skip_last = 1
    out = []
    number_succesful = []
    for number in range(int(max(data["raw_response"][:,4])) + 1): #np.arange(52,1940): #
        current_raw = data["raw_response"][data["raw_response"][:,4] == number,:]
        f = np.unique(current_raw[:,3])
        temp_zero = np.zeros(len(f))
        for idx, frequency in enumerate(f):
            current_f = current_raw[current_raw[:,3] == frequency, :]
            if use_fix:
                if use_fix < current_f[int(current_f[0,2]),1]:
                    temp_zero[idx] = True
            else:    
                std = np.std(current_f[:,1])        
                mean = np.mean(current_f[:,1])
                if ((mean + std * std_times) < current_f[int(current_f[0,2]),1]):
                    temp_zero[idx] = True
        out.append(np.vstack((temp_zero, f, np.repeat(number, len(f)))).T)
        if skip_last:
            number_succesful.append(np.hstack((np.count_nonzero(out[-1][:-skip_last,0]), len(f)-skip_last, number)))
        else:
            number_succesful.append(np.hstack((np.count_nonzero(out[-1][:,0]), len(f), number)))
            
        if progressBar is not None:
            Bar, lblBar = progressBar
            Bar.setValue(number / int(data["raw_response"][-1,4]) * 100)
            lblBar.setText("removing bad fits...")
            # lblProgress.setText(str(int(identifier)) + "/" + str(total))
            QtWidgets.QApplication.processEvents()
            
    out = np.vstack(out)
    number_successful = np.vstack(number_succesful)
    
    percentage = []
    for f in np.unique(out[:,1]):
        current_f = out[out[:,1] == f, :]
        percentage.append([np.count_nonzero(current_f[:,0])/len(current_f), len(current_f), f])
    percentage = np.array(percentage)
    
    data["correct"] = out
    
    # plt.figure()
    # plt.semilogx(percentage[:,2],percentage[:,0], 'o')
    # plt.xlabel("frequeny [Hz]")
    # plt.ylabel("percentage successful measurement")
    # plt.title("successful measurements with mean + sd * " + str(std_times))
    
    return number_successful

def calc_exclude(data, threshold = 0, cutoff_high_f = 3):
    excluded = []
    e_list = np.zeros(len(np.unique(data["correct"][:,-1])))
    for number in np.unique(data["correct"][:,-1]):
        #trick to get cutoff correct also for 0:
        #invert array, cutoff at start, invert again
        n_success = data["correct"][data["correct"][:,-1] == number, 0][::-1][cutoff_high_f:][::-1]
        if np.sum(n_success) < len(n_success) - threshold:
            excluded.append(int(number))
            e_list[int(number)] = 1
            
    return excluded
    

#%% plotting of data
def plot_data(data,fixed_slope = 0, mean_per_cell = False, scans = False):
    "plots active and passive mean for each phase"
     
    #create array with interpolated active response, mean of passive and effective Energy
    if not "df_ap" in data:
        interpolate_active(data,fixed_slope)
    
    #get frequencies of passive rheology and length
    frequencies = np.real(np.unique(data["passive"][:,4]))
    l = len(frequencies)
    #define lagtimes for MSD
    lagtimes = np.unique(np.round([10**power for power in np.arange(0, 5.75, .125)]).astype(int))
    
    response_y = "response y"
    eeff_y = "eff E y"
    g1_y = "G' y"
    g2_y = "G'' y"
    if scans:
        response_y = "response y scan"
        eeff_y = "eff E y scan"
        g1_y = "G' y scan"
        g2_y = "G'' y scan"
 
    
    #drop duplicates to filter for repeats per cell
    unique_day_phase = data["df_ap"][["day","cell","phase"]].drop_duplicates()
    
    #create figures to fill with data
    fig_ai,ax_ai = plt.subplots(3,4,sharex = True,sharey = True)      #active response of interpolated
#    fig_ai_m,ax_ai_m = plt.subplots(3,4,sharex = True,sharey = True)  #active interpolated mean of cells
    fig_p,ax_p = plt.subplots(3,4,sharex = True, sharey = True)       #passive response
    fig_ap,ax_ap = plt.subplots(3,4,sharex = True, sharey = True)     #active and passive response
    fig_e, ax_e = plt.subplots(3,4)                               #effective energy
    fig_em, ax_em = plt.subplots(3,4,sharex = True, sharey = True)    #mean of effective energy
    fig_apm, ax_apm = plt.subplots(3,4,sharex = True,sharey = True)   #moved passive to active response
    fig_m, ax_m = plt.subplots(3,4,sharex = True,sharey = True)       #effective energy of moved passive response to active response
    fig_e_box, ax_e_box = plt.subplots(3,4, sharex = True, sharey = True)
    fig_g, ax_g = plt.subplots(3,4,sharex = True, sharey = True)      #shear modulus
    fig_g_m, ax_g_m =  plt.subplots(3,4,sharex = True, sharey = True) #mean shear modulus
    fig_vh, ax_vh = plt.subplots(3,4, sharex = True, sharey = True)
    fig_msd, ax_msd = plt.subplots(3,4, sharex = True, sharey = True) #mean squared displacement
    fig_msd_m, ax_msd_m = plt.subplots(3,4, sharex = True, sharey = True) #mean squared displacement
    fig_pow, ax_pow = plt.subplots(3,4, sharex = True, sharey = True)
#    fig_akz, ax_akz = plt.subplots(3, 4, sharex = True, sharey = True)  #boxplot akz
    
    #create empty arrays and set running indices to 0
    i = j = 0
      
    #go through all phases
    for phase in range(1,13):#[1,2 ,4,6,7,8,9,10,11,12,13]:
        
        phase_data, numbers = getPhase(data, phase, mean_per_cell = mean_per_cell, num_out = True)
        akz_data = data["akz_Ta"][getPhase(data, phase)["number"].drop_duplicates().astype(int), :3]
        power_data = data["power"][getPhase(data, phase)["number"].drop_duplicates().astype(int),:]
        vh_data = data["vanHove"][np.isin(np.real(data["vanHove"][:,-1]), getPhase(data, phase)["number"]),:]
        msd_data = data["msd"][np.isin(np.real(data["msd"][:,-1]), getPhase(data, phase)["number"]),:]
        phase_per_cell = unique_day_phase[unique_day_phase["phase"] == phase]
        
        if not phase_data.empty:
            #create reshaped arrays with active, passive and effective energy data         
            #check if mean has to be calculated before
            if mean_per_cell:    
                #more arrays
                active = []
                passive = []
                eeff = []
                g1 = []
                g2 = []
                akz = []
                power = []
                for day_cell_idx in phase_per_cell.to_numpy():
                    current_data = data["df_ap"][np.all(data["df_ap"][["day","cell","phase"]] == day_cell_idx, axis =1)]
                    active.append(np.nanmean(np.reshape(current_data[response_y].to_numpy(),[int(len(current_data)/l),l]),0))
                    passive.append(np.nanmean(np.reshape(current_data["PSD y"].to_numpy(),[int(len(current_data)/l),l]),0))
                    eeff.append(np.nanmean(np.reshape(current_data[eeff_y].to_numpy(),[int(len(current_data)/l),l]),0))
                    g1.append(np.nanmean(np.reshape(current_data[g1_y].to_numpy(),[int(len(current_data)/l),l]),0))
                    g2.append(np.nanmean(np.reshape(current_data[g2_y].to_numpy(),[int(len(current_data)/l),l]),0))
                    akz.append(np.nanmean(akz_data,0))
                    power.append(np.nanmean(power_data, 0))
                active = np.transpose(np.vstack(active))
                passive = np.transpose(np.vstack(passive))
                eeff = np.transpose(np.vstack(eeff))
                g1 = np.transpose(np.vstack(g1))
                g2 = np.transpose(np.vstack(g2))
                akz = np.nanmedian(np.vstack(akz),0)
                power = np.nanmedian(np.vstack(power),0)
            else:
                active = np.transpose(np.reshape(phase_data[response_y].to_numpy(),[int(len(phase_data)/l),l]))
                passive = np.transpose(np.reshape(phase_data["PSD y"].to_numpy(),[int(len(phase_data)/l),l]))
                eeff = np.transpose(np.reshape(phase_data[eeff_y].to_numpy(),[int(len(phase_data)/l),l]))
                g1 = np.transpose(np.reshape(phase_data[g1_y].to_numpy(),[int(len(phase_data)/l),l]))
                g2 = np.transpose(np.reshape(phase_data[g2_y].to_numpy(),[int(len(phase_data)/l),l]))
                akz = np.nanmedian(akz_data,0)
                power = np.nanmedian(power_data,0)
                                
            
            #skip noise points
            noise_indices = [] #[8, 9]
            ind = np.ones(l).astype(bool)
            ind[noise_indices] = 0
            
            #remove rows with only nans 
            active_nan = np.sum(~np.isnan(active),1) > 0
            
            #calculate shift to shift passive measurement to active measurement at 850 Hz 
            shift = np.nanmean(active[13,:])/np.nanmean(passive[13,:])
            
            ax_ai[i,j].loglog(frequencies, active)
            ax_ai[i,j].loglog(frequencies[active_nan], np.nanmean(active[active_nan],1),'-ob', markersize = 3)
            ax_p[i,j].loglog(frequencies, passive)
            ax_p[i,j].loglog(frequencies, np.nanmean(passive,1),'-or', markersize = 3)
            ax_apm[i,j].errorbar(frequencies[active_nan], np.nanmean(active[active_nan],1), [[0]*sum(active_nan), np.nanstd(active[active_nan],1)],color = 'blue',capsize = 3,marker = 'o', markersize = 3)
            ax_apm[i,j].errorbar(frequencies, np.nanmean(passive,1)*shift, [[0]*l, np.nanstd(passive,1)],color = 'red',capsize = 3,marker = 'o', markersize = 3)
            ax_ap[i,j].errorbar(frequencies[active_nan], np.nanmean(active[active_nan],1),[[0]*sum(active_nan),np.nanstd(active[active_nan],1)],color = 'blue',capsize = 3,marker = 'o',markersize = 3)
            ax_ap[i,j].errorbar(frequencies, np.nanmean(passive,1),[[0]*l,np.nanstd(passive,1)],color = 'red',capsize = 3,marker = 'o',markersize = 3)
            # ax_ap[i,j].errorbar(frequencies, np.nanmedian(active,1),[[0]*l,np.nanstd(active,1)],color = 'blueviolet',capsize = 3,marker = 'o',markersize = 3)
            # ax_ap[i,j].errorbar(frequencies, np.nanmedian(passive,1),[[0]*l,np.nanstd(passive,1)],color = 'salmon',capsize = 3,marker = 'o',markersize = 3)
            ax_e[i,j].semilogx(frequencies, eeff,'-o')
            ax_e[i,j].semilogx(frequencies[active_nan], np.nanmedian(eeff[active_nan],1),'-o',color = 'orange')
            ax_em[i,j].errorbar(frequencies[active_nan], np.nanmedian(eeff[active_nan],1),[[0]*sum(active_nan),np.nanstd(eeff[active_nan],1)],color = 'orange',capsize = 3,marker = 'o',markersize = 3)
            ax_m[i,j].semilogx(frequencies[active_nan], shift*np.nanmean(passive[active_nan],1)/np.nanmean(active[active_nan],1),'-o',color = 'orange')
            # ax_m[i,j].boxplot(shift*passive[ind]/active[ind])
            
            mask = ~np.isnan(eeff)
            filtered = [e[m] for e, m in zip(eeff, mask)]
            ax_e_box[i,j].boxplot(filtered, showfliers = False)#, showmeans = True)
            ax_e_box[i,j].semilogy()
            ax_e_box[i,j].set_xticklabels(frequencies)
            
            ax_g[i,j].loglog(frequencies, g1)
            ax_g[i,j].errorbar(frequencies[active_nan], np.nanmean(g1[active_nan],1), [[0]*sum(active_nan), np.nanstd(g1[active_nan],1)], color = 'blue', capsize = 3, marker = 'o',markersize = 3)
            ax_g[i,j].loglog(frequencies, g2)
            ax_g[i,j].errorbar(frequencies[active_nan], np.nanmean(g2[active_nan],1), [[0]*sum(active_nan), np.nanstd(g2[active_nan],1)], color = 'red', capsize = 3, marker = 'o',markersize = 3)
    #        ax_g_m[i,j].errorbar(frequencies, np.nanmean(g1,1), [[0]*l, np.nanstd(g1,1)], color = 'blue', capsize = 3, marker = 'o',markersize = 3,fmt = 'none')
    #        ax_g_m[i,j].errorbar(frequencies, np.nanmean(g2,1), [[0]*l, np.nanstd(g2,1)], color = 'red', capsize = 3, marker = 'o',markersize = 3,fmt = 'none')
            ax_g_m[i,j].loglog(frequencies[active_nan], np.nanmedian(g1[active_nan],1), 'ob',markersize = 4)
            ax_g_m[i,j].loglog(frequencies[active_nan], np.nanmedian(g2[active_nan],1), 'or',markersize = 4)
    
            ax_g_m[i,j].loglog(frequencies, np.exp(G_prime(frequencies, *akz[:3])),'--b')
            ax_g_m[i,j].loglog(frequencies, np.exp(G_2prime(frequencies, *akz[:3])),'--r')
            ax_g_m[i,j].legend(["G'","G''"])
            
            ax_pow[i,j].loglog(frequencies[active_nan], np.nanmedian(g1[active_nan],1), 'ob',markersize = 4)
            ax_pow[i,j].loglog(frequencies[active_nan], np.nanmedian(g2[active_nan],1), 'or',markersize = 4)
            ax_pow[i,j].loglog(frequencies, np.exp(G_power(frequencies, *power[:3])),'--b')
            ax_pow[i,j].loglog(frequencies, np.exp(G_power(frequencies, *power[3:6])),'--r')
            ax_pow[i,j].legend(["G'", "G''"])
            
            # ax_g_m[j].loglog(frequencies, np.nanmean(g1,1), 'ob',markersize = 4)
            # ax_g_m[j].loglog(frequencies, np.nanmean(g2,1), 'or',markersize = 4)
    
            # ax_g_m[j].loglog(frequencies, np.exp(G_prime(frequencies, *akz[:3])),'--b')
            # ax_g_m[j].loglog(frequencies, np.exp(G_2prime(frequencies, *akz[:3])),'--r')
            # ax_g_m[j].legend(["G'","G''"], loc = 'upper left')
            
            
#            ax_vh[i,j].plot(np.unique(vh_data[:,4]),np.reshape(vh_data[:,8], [int(len(vh_data[:,8])/len(np.unique(vh_data[:,4]))),len(np.unique(vh_data[:,4]))]).T,'o', markerSize = 2)
            ###turned off for sets with 2##:
            vh_reshape = np.reshape(vh_data[:,8], [int(len(vh_data[:,8])/len(np.unique(vh_data[:,4]))),len(np.unique(vh_data[:,4]))]).T
            ax_vh[i,j].plot(np.unique(vh_data[:,4]),np.sum(vh_reshape,1)/np.sum(vh_reshape),'o', color = 'gray' , markerSize = 2)
#            ax_vh[i,j].plot(np.unique(vh_data[:,4])[185:225],np.sum(vh_reshape,1)[185:225]/np.sum(vh_reshape),'o', color = 'snow' , markerSize = 1)
            ax_vh[i,j].set_ylim(10**-8, 0.05)
            msd_reshape = np.reshape(msd_data[:,6], [int(len(msd_data[:,6])/len(lagtimes)),len(lagtimes)]).T
            #plot against lagtimes (has to be divided by scan rate)
            ax_msd[i,j].loglog(lagtimes/50000, msd_reshape,'o', markerSize = 2)
            ax_msd_m[i,j].loglog(lagtimes/50000, np.mean(msd_reshape,1), 'o-', markerSize = 2)

#            ax_vh[i,j].plot(vh_data[:,4],vh_data[:,9],'o', markerSize = 2)
#            ax_vh[i,j].plot(vh_data[:,4],vh_data[:,10],'o', markerSize = 2)
            
#            ax_akz[i,j].boxplot()
            
            ax_ap[i,j].loglog()
            ax_em[i,j].semilogx()
            ax_apm[i,j].loglog()
            ax_m[i,j].semilogx()
                    
            ax_ap[i,j].text(0.15,30,'n = ' + str(np.size(active,1)))
            
            #add titles 
            # titles = ["Prophase", "Pro - metaphase", "Metaphase", "Anaphase", "Anaphase", "Telophase", "after Cytokinesis", "Interphase","STC treated", "STC + Blebbistatin","STC + Cyto B", "Nocodazole", "Interphase + Noco"]
            # titles = ["750 nm", "1 µm; 300 mOsm", "2 µm", "Nocodazole", "Blebbistatin", "450 mOsm", "600 mOsm", "900 mOsm", "350 mOsm", "400 mOsm", "interphase", "Calyculin"]
            # titles = ["750 nm", "1 µm; 300 mOsm", "2 µm", "interphase", "350 mOsm", "400 mOsm", "450 mOsm", "600 mOsm", "900 mOsm", "Blebbistatin", "Nocodazole", "Calyculin"]
            # titles = ["750 nm", "1 µm; 300 mOsm", "2 µm", "Latrunculin A", "350 mOsm", "400 mOsm", "450 mOsm", "600 mOsm", "900 mOsm", "Blebbistatin", "Nocodazole", "Calyculin"]
            # titles = ["Prophase", "Pro - metaphase", "Metaphase", "Anaphase", "Anaphase", "Telophase", "after Cytokinesis", "Interphase","STC treated", "STC + Blebbistatin","STC + Cyto B", "Nocodazole", "Interphase + Noco", "750 nm", "1 µm; 300 mOsm", "2 µm", "interphase", "350 mOsm", "400 mOsm", "450 mOsm", "600 mOsm", "900 mOsm", "Blebbistatin", "Nocodazole", "Calyculin", 
                      # "Latrunculin", "Cyto B", "Caly Cyto", "inter blebb", "inter cyto", "150 mOsm", "y-27632", "500 mOsm", "550 mOsm"]
            # titles = ["Prophase", "Metaphase", "Anaphase", "Telophase", "after Cytokinesis", "Interphase","STC treated", "STC + Blebbistatin","STC + Cyto B", "Nocodazole", "Interphase + Noco", "750 nm", "1 µm; 300 mOsm", "2 µm", "interphase", "350 mOsm", "400 mOsm", "450 mOsm", "600 mOsm", "900 mOsm", "Blebbistatin", "Nocodazole", "Calyculin", 
            #           "Latrunculin", "Cyto B", "Caly Cyto", "inter blebb", "inter cyto", "150 mOsm", "y-27632", "450 + blebb", "500 mOsm", "550 mOsm",
            #           "Prophase", "Metaphase", "Anaphase", "Telophase", "after Cytokinesis", "Jaspla", "CK-666", "SMIFH2"]
            titles = ["inter", "pro", "meta", "ana", "telo", "after", ".75", "1", "2", "150", "350", "400" ,"450", "500", "550", "600", "900", "Bleb", "Lat", "Cyto", "Jasp", "CK", "SMI", "Noco", "Caly", "Y27", "i Bleb", "i Noc", "i Cyt"]

            
            for axes in [ax_ai, ax_p, ax_ap, ax_em, ax_e, ax_m, ax_e_box, ax_apm, ax_g, ax_g_m, ax_vh, ax_msd, ax_pow]:
                axes[i,j].set_title(titles[phase-1])
                axes[i,j].grid()
                
            ax_ap[i,j].set_xlim([10**-1, 10**4])
            ax_ap[i,j].set_ylim([10,10**7])
            ax_em[i,j].set_ylim([0, 200])
            ax_e[i,j].set_ylim([0, 200])
            ax_g[i,j].set_ylim([0.1,100000])
            ax_g_m[i,j].set_ylim([.1,100000])

            #go to next subplot
            j += 1
            if j == 4:
                i += 1
                j = 0  
     
    fig_titles = ["effective energy", "mean effective energy", "response of active microrheology of interpolated active measurements", 
                  "response of passive microrheology", "response of passive microrheology", "effective Energy after shift of passive data to active data at ~850 Hz",
                  "response of active and passive microrheology after shift of passive response to active data at ~850 Hz", "fit of model to G"]
    for idx, figure in enumerate([fig_e, fig_em, fig_ai, fig_p, fig_ap, fig_m, fig_apm, fig_g_m]):
        figure.suptitle(fig_titles[idx])
        
    # fig_e.suptitle("effective energy")
    # fig_em.suptitle("mean effective energy")
    # fig_ai.suptitle("response of active microrheology of interpolated active measurements")
    # fig_p.suptitle("response of passive microrheology")
    # fig_ap.suptitle("response of active and passive microrheology")
    # fig_m.suptitle("effective Energy after shift of passive data to active data at ~850 Hz")
    # fig_apm.suptitle("response of active and passive microrheology after shift of passive response to active data at ~850 Hz")
    # fig_g_m.suptitle("fit of model to G")
    
def plot_eeff(data, fixed_slope = 0):
    #get frequencies of passive rheology and length
    frequencies = np.real(np.unique(data["passive"][:,4]))
    l = len(frequencies)
    
    a, k, z, Ta, idx, akz_mean, akz_median, akz_err, eeff = [], [], [], [], [], [], [], [], []
    #go through all phases
    for phase in range(1,13): 
        phase_data = getPhase(data, phase)
        akz_data = data["akz_Ta"][getPhase(data, phase)["number"].drop_duplicates().astype(int), :4]
        
        eeff.append(phase_data[phase_data["f"] == 0.2]["eff E y"][~np.isnan(phase_data[phase_data["f"] == 0.2]["eff E y"])])
        
        if akz_data.any():
            akz_mean.append(np.nanmean(akz_data,0))
            akz_median.append(np.nanmedian(akz_data,0))
            akz_err.append(np.nanstd(akz_data,0))
            a.append(akz_data[:,0])
            k.append(akz_data[:,1])
            z.append(akz_data[:,2])
            Ta.append(akz_data[:,3])
        
    akz_mean = np.vstack((akz_mean))
    akz_median = np.vstack((akz_median))
    akz_err = np.vstack((akz_err))
    
    fig, ax = plt.subplots(2, 2, sharex = True)
    fig_med, ax_med = plt.subplots(2,2, sharex = True)
    fig_box, ax_box = plt.subplots(2,2, sharex = False)
    
    # x_labels = ["Prophase", "Metaphase", "Anaphase", "Telophase", "after Cytokinesis", "Interphase","STC treated", "STC + Blebbistatin","STC + Cyto B", "Nocodazole", "Interphase + Noco"]
    # x_labels = ["750 nm", "1 µm", "2 µm", "Nocodazole", "Blebbistatin", "150 mOsm", "300 mOsm", "600 mOsm"]
    x_labels = ["750 nm", "1 µm", "2 µm", "interphase", "350 mOsm", "400 mOsm", "450 mOsm", "600 mOsm", "900 mOsm", "Blebbistatin", "Nocodazole", "Calyculin"]
    titles = ["alpha", "kappa", "zeta", "Ta"]
    
    plt.figure()
    plt.boxplot(eeff, showfliers = False)
    plt.xticks(range(1,13), x_labels, rotation = 15)
    plt.ylabel("effective energy [kbT]")
    plt.title("effective energy at 0.2 Hz")
    
    para_list = [a, k, z, Ta] 
    
    i = j = 0
    for parameter in range(4):
        ax[i,j].bar(range(1,len(akz_mean) + 1), akz_mean[:,parameter])
        ax[i,j].errorbar(range(1,len(akz_mean) + 1), akz_mean[:,parameter], akz_err[:,parameter], capsize = 3, fmt = 'none')
        ax_med[i,j].bar(range(1,len(akz_median) + 1), akz_median[:,parameter])
        # ax_med[i,j].errorbar(range(1,len(akz_median) + 1), akz_median[:,parameter], akz_err[:,parameter], capsize = 3, fmt = 'none')
        ax_box[i,j].boxplot(para_list[parameter],showfliers=False)

        ax[i,j].set_title(titles[parameter])
        ax_med[i,j].set_title(titles[parameter])
        ax_box[i,j].set_title(titles[parameter])
        
        
        if i == 1:
            ax[i,j].set_xticks(range(1,len(akz_mean) + 1))
            ax[i,j].set_xticklabels(x_labels, rotation = 30)
            ax_med[i,j].set_xticks(range(1,len(akz_median) + 1))
            ax_med[i,j].set_xticklabels(x_labels, rotation = 30)
            ax_box[i,j].set_xticks(range(1,len(akz_median) + 1))
            ax_box[i,j].set_xticklabels(x_labels, rotation = 30)
        
        j += 1
        if j == 2:
            i +=1 
            j = 0
  
          
def plot_parameters(df, threshold = .9, num = 1):
    if num == 1:
        numbers = [29, 13, 16, 17, 18, 32, 33, 19, 20]; names = ["150 mOsm", "300 mOsm", "350 mOsm", "400 mOsm", "450 mOsm", "500 mOsm", "550 mOsm", "600 mOsm", "900 mOsm"]
    elif num == 2:
        #phases
        numbers = [6,1,2,3,4,5,7]; names = ["Interphase", "Prophase", "Metaphase", "Anaphase", "Telophase", "after Cytokinesis", "STC"]
    elif num == 3:
        #drugs
        # numbers = [13, 21, 22, 23, 24, 25, 26, 30]; names = ["STC", "Blebbistatin", "Nocodazole", "Calyculin", "Latrunculin A", "Cytochalasin B", "Caly + Cyto", "Y-27632"]; order = [13, 21, 22, 23, 24, 25, 26, 30]
        numbers = [7, 8,9,10,13, 21, 22, 23, 24, 25, 26, 30]; names = ["STC", "Blebbistatin", "Cytochalasin B", "Nocodazole", "STC", "Blebbistatin", "Nocodazole", "Calyculin", "Latrunculin A", "Cytochalasin B", "Caly + Cyto", "Y-27632"]
    elif num == 4:
        #blebb 450 mosm
        numbers = [13, 18, 21, 31]; names = ["300 mOsm", "Blebbistatin", "450 mOsm", "450 mOsm + Blebb"]
    elif num == 5:
        #inter
        numbers = [6, 15, 13, 27, 28]; names = ["Inter", "Inter", "STC", "Inter + Blebb", "Cyto B"]
    elif num == 6:
        numbers = [13, 17, 32, 19, 20]; names = ["300", "400", "500", "600", "900"]
    elif num == 10:
        numbers = range(34); 
        names = ["not used", "Pro", "Meta", "Ana", "Telo", "Cyto", "Inter", "STC", "Blebb", "Cyto B", "Noco", "i + noco", "750", "1", "2", "inter", 
                 "350", "400", "450", "600", "900", "blebb", "noco", "caly", "lat", "cyto b", "caly cyto", "i + blebb", "i + cyto", "150", "Y-27632", 
                 "450 + blebb", "500", "550"]
        #names = range(33); 

    parameters = [["alpha", df["akz"], df["akz"]["R square"] > threshold, 1],
            ["a G'", df["power"], df["power"]["R2 G'"] > threshold, 1],
            ["b G'", df["power"], df["power"]["R2 G'"] > threshold, 1],
            ["a G''", df["power"], df["power"]["R2 G''"] > threshold, 1],
            ["b G''", df["power"], df["power"]["R2 G''"] > threshold, 1],
            ["b G' and G''", df["power"], df["power"]["R2 G' and G''"] > threshold, 1],
            ["2p b G'", df["power"], df["power"]["2p R2 G'"] > threshold, 1],
            ["2p b G''", df["power"], df["power"]["2p R2 G''"] > threshold, 1]]
    
    
    for para in parameters:
        fig, ax = plt.subplots(1,1)
        sb.boxplot(x = "phase", y = para[0], data = para[1][np.logical_and(np.isin(para[1]["phase"], numbers), para[2] )], ax = ax, color = 'white', showfliers = False, order = numbers)#, showmeans = True)
        if para[3]:
            sb.swarmplot(x = "phase", y = para[0], data = para[1][np.logical_and(np.isin(para[1]["phase"], numbers), para[2] )], ax = ax, order = numbers)
#            plt.figure()
#            sb.barplot(x = "phase", y = para[0], data = para[1][np.logical_and(np.isin(para[1]["phase"], numbers), para[2])])
        if num == 10:
            plt.xticks(range(len(names)), names, rotation = 30)
            # print(len(para[1][np.logical_and(np.isin(para[1]["phase"], numbers), para[2] )]))
        else:
            plt.xticks(range(len(names)), names)
        # plt.yticks(np.arange(0.35, .8, 0.05))
        plt.grid(axis = "y")
        plt.xlabel("")
        plt.title(para[0])
    
            
def plot_rel_g(data):
    
    frequencies = data["df_ap"]["f"].drop_duplicates()
    l = len(frequencies)
    g1 = []
    g2 = []
    
    for phase in [2, 7, 8]:# [1,2,3]:
        phase_data = getPhase(data, phase)
        g1.append(np.nanmean(np.transpose(np.reshape(phase_data["G' y"].to_numpy(),[int(len(phase_data)/l),l])),1))
        g2.append(np.nanmean(np.transpose(np.reshape(phase_data["G'' y"].to_numpy(),[int(len(phase_data)/l),l])),1))
        # akz.append(data["akz_Ta"][:,data["akz_Ta"][:,-1] == phase_num])
        
    fig, ax = plt.subplots(3,3, sharex = True, sharey = True)
    for i in range(3):
        for j in range(3):
            ax[i,j].plot(g1[i]/g1[j],'or')    
            ax[i,j].plot(g2[i]/g2[j], 'ob')
    
    return g1, g2

def plot_tests(data, log = False, titles = 1):
    "plots various tests for eeff, alpha, kappa, zeta, Ta"
    
    a, k, z, Ta, eeff = [], [], [], [], []
    
    # treatment = ["placeholder", "750 nm", "1 µm", "2 µm", "Nocodazole", "Blebbistatin", "450 mOsm", "600 mOsm", "900 mOsm"]
    if titles:
        treatment = ["Prophase", "Metaphase", "Anaphase", "Telophase", "after Cytokinesis", "Interphase","STC treated", "STC + Blebbistatin","STC + Cyto B", "Nocodazole", "Interphase + Noco"]
    else:
        treatment = ["750 nm", "1 µm", "2 µm", "interphase", "350 mOsm", "400 mOsm", "450 mOsm", "600 mOsm", "900 mOsm", "Blebbistatin", "Nocodazole", "Calyculin"]
    
    
    #go through all phases
    for phase in range(6):#range(1,13): 
        phase_data = getPhase(data, phase)
        akz_data = data["akz_Ta"][getPhase(data, phase)["number"].drop_duplicates().astype(int), :4]
        #effective energy at 0.2 Hz
        
        if log:
            eeff.append(np.log(phase_data[phase_data["f"] == 0.2]["eff E y"][~np.isnan(phase_data[phase_data["f"] == 0.2]["eff E y"])]))
        else: 
            eeff.append(phase_data[phase_data["f"] == 0.2]["eff E y"][~np.isnan(phase_data[phase_data["f"] == 0.2]["eff E y"])])
            
        if akz_data.any():
            a.append(np.real(akz_data[:,0]))
            k.append(np.real(akz_data[:,1]))
            z.append(np.real(akz_data[:,2]))
            Ta.append(np.real(akz_data[:,3]))
    
    
    length = len(a)
    
    g = []
    g_alpha = []
    alphabet = [['a'], ['b'], ['c'], ['d'], ['e'], ['f'], ['g'], ['h'], ['i'], ['j'], ['k'], ['l'], ['m']]
    for idx, e in enumerate(eeff):
        g.append(alphabet[idx]*len(e))
        
    for idx, e in enumerate(a):
        g_alpha.append(alphabet[idx]*len(e))
    
    # g = [['a']*38, ['b']*62, ['c']*6, ['d']*35, ['e']*27, ['f']*22, ['g']*42, ['h']*10, ['i']*22, ['j']*70, ['k']*48, ['l']*24]
    tukey = sp.posthoc_tukey_hsd(np.concatenate(eeff),np.concatenate(g))

    fig, ax = plt.subplots()
    ax.matshow(tukey)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(range(length))
    ax.set_xticklabels(treatment,rotation = 15)
    ax.set_xticks(range(length))
    ax.set_yticklabels(treatment)
    ax.set_title("effective energy Tukey's test")
    
    ##
    tukey_alpha = sp.posthoc_tukey_hsd(np.concatenate(a), np.concatenate(g_alpha))
    fig, ax = plt.subplots()
    ax.matshow(tukey_alpha)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(range(length))
    ax.set_xticklabels(treatment,rotation = 15)
    ax.set_xticks(range(length))
    ax.set_yticklabels(treatment)
    ax.set_title("alpha Tukey's test")
    
    
    #function to make plots
    def make_plot(current_test, plot_title):
        
        #create figure with axes for colorbar
        fig, ax = plt.subplots()
        cax = fig.add_axes()
        current_test[current_test == -1] = 1
        color_array = current_test*[1]
        
        #assign number of 2 to 8 for statistical significance
        for idx, boundary in enumerate([1e-10, 1e-4, 1e-3, 1e-2, 5e-2, 0.1,  1.1]):
            color_array[color_array < boundary] = idx + 2

        n = 7 #color_array.max() - color_array.min() + 1
        cmap = plt.cm.Reds_r(np.linspace(0,1,n))
        cmap = LinearSegmentedColormap.from_list('Custom cmap', cmap, n)
      
        #plot statistics with colors and write p-values 
        image = ax.matshow(color_array - 2, vmin = 0, vmax = 7, cmap = cmap)
        for phase1 in range(length):
            for phase2 in range(length):
                if color_array[phase1,phase2] == 9-n:
                    ax.text(phase1-0.3,phase2,str('%.1e' % current_test[phase1,phase2]),color = 'white', size = 'small')
                else:
                    ax.text(phase1-0.3,phase2,str('%.1e' % current_test[phase1,phase2]), size = 'small')
                    
        
        cbar = fig.colorbar(image, cax = cax, ticks = np.arange(.5, 7))
        cbar.set_ticklabels([' < 1e-10', ' ****', ' ***', ' **', ' *', '< 0.1', 'ns'])
        
        
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticks(range(length))
        ax.set_xticklabels(treatment,rotation = 15)
        ax.set_xticks(range(length))
        ax.set_yticklabels(treatment)
        ax.set_title(plot_title)


    #dunn's posthoc test after kruskal-wallis
    #kruskal-wallis has to be rejected
    #scipy.stats.kruskal(eeff[0],eeff[1],eeff[2],eeff[3], eeff[4], eeff[5], eeff[6], eeff[7], eeff[8], eeff[9], eeff[10], eeff[11])
    
    
    test_nonpara = [(sp.posthoc_dunn(a).to_numpy(), "Dunn's posthoc test, \u03B1"), 
                    (sp.posthoc_dunn(a, p_adjust = 'bonferroni').to_numpy(),"Dunn's + Bonferroni, \u03B1"), 
                    (sp.posthoc_ttest(a), "t-test \u03B1"),
                    (sp.posthoc_ttest(eeff), "t-test, effective energy"),
                    (sp.posthoc_dunn(eeff).to_numpy(), "Dunn's posthoc test, effective energy")]
    for test in test_nonpara:
        make_plot(test[0], test[1])
        
    # dunns = sp.posthoc_dunn(a).to_numpy()
    # make_plot(dunns, "Dunn's posthoc test")
    # dunns_b = sp.posthoc_dunn(a, p_adjust = 'bonferroni').to_numpy()
    # make_plot(dunns_b, "Dunn's posthoc test with Bonferroni" )
    # t_test = sp.posthoc_ttest(a)
    # make_plot(t_test, "multi t-test")


    #pairwise tests    
    # title = ["pairwise t-test, alpha", "kruskal-wallis, alpha", "t-test kappa", "t-test Ta", "t-test effective energy at 0.2 Hz", "kruskal-wallis effective energy at 0.2 Hz"]
    # test_and_variable = [(scipy.stats.ttest_ind, a),  (scipy.stats.kruskal, a), (scipy.stats.ttest_ind, k), 
    #                      (scipy.stats.ttest_ind, Ta), (scipy.stats.ttest_ind, eeff), (scipy.stats.kruskal, eeff)]
  
    # for outer_idx, test in enumerate(test_and_variable):
    #     current_test = np.zeros((length,length))
    #     for phase1 in range(length):
    #         for phase2 in range(length):    
    #             out, current_test[phase1,phase2] = test[0](test[1][phase1],test[1][phase2], nan_policy = 'omit')
        
     
    #     make_plot(current_test, title[outer_idx])

            
    return eeff, a
            


def make_dataFrame(data):
    df_active = pd.DataFrame({"day":data["active"][:,0], "cell":data["active"][:,1],
                     "measure":data["active"][:,2], "repeat":data["active"][:,3],
                     "f":data["active"][:,4], "response x":data["active"][:,5],
                     "response y":data["active"][:,6], "G x":data["active"][:,7],
                     "G y":data["active"][:,8], "lunam slope":data["active"][:,9],"lunam R²":data["active"][:,10],
                     "detector slope":data["active"][:,11], "beadsize":data["active"][:,12], "traps":data["active"][:,13], 
                     "time":data["active"][:,14], "number":data["active"][:,15], "phase":data["active"][:,16]})
    df_active_noNa = df_active.dropna()
    
    df_passive = pd.DataFrame({"day":data["passive"][:,0], "cell":data["passive"][:,1],
                     "measure":data["passive"][:,2], "repeat":data["passive"][:,3],
                     "binned f":data["passive"][:,4], "PSD x":data["passive"][:,5],
                     "PSD y":data["passive"][:,6],"lunam slope":data["passive"][:,7], 
                     "lunam R²":data["passive"][:,8], "detector slope":data["passive"][:,9],
                     "time":data["passive"][:,10],"number":data["passive"][:,11], 
                     "phase":data["passive"][:,12]})  
    df_passive_noNa = df_passive.dropna()
  
    df_ap = pd.DataFrame({"day":data["ap"][:,0], "cell":data["ap"][:,1],
                     "measure":data["ap"][:,2], "f":data["ap"][:,3], "response x":data["ap"][:,4],
                     "response y":data["ap"][:,5], "response y scan":data["ap"][:,6], "PSD x":data["ap"][:,7], "PSD y":data["ap"][:,8],
                     "eff E x":data["ap"][:,9], "eff E y scan":data["ap"][:,10],
                     "G' x":data["ap"][:,11],"G' y":data["ap"][:,12], "G' y scan":data["ap"][:,13], 
                     "G'' x":data["ap"][:,14], "G'' y":data["ap"][:,15], "G'' y scan":data["ap"][:,16],
                     "det slope x":data["ap"][:,17], "det slope y":data["ap"][:,18], "det slope x scan":data["ap"][:,19],
                     "det slope y scan":data["ap"][:,20], "shift x":data["ap"][:,21], "shift y":data["ap"][:,22], "beadsize":data["ap"][:,23],
                     "time":data["ap"][:,24], "number":data["ap"][:,25], "phase":data["ap"][:,26]})
    df_ap_noNa = df_ap.dropna()
    
    df_power = pd.DataFrame({"a G'":data["power"][:,0], "b G'":data["power"][:,1], "c G'":data["power"][:,2], 
                             "a G''":data["power"][:,3], "b G''":data["power"][:,4], "c G''":data["power"][:,5], 
                             # "a G' and G''":data["power"][:,6], "b G' and G''":data["power"][:,7], "c G' and G''":data["power"][:,8], 
                              "R2 G'":data["power"][:,6], "R2 G''":data["power"][:,7], #"R2 G' and G''":data["power"][:,11],
                             "2p a G'":data["power"][:,8], "2p b G'":data["power"][:,9], "2p a G''":data["power"][:,10],
                             "2p b G''":data["power"][:,11], "2p R2 G'":data["power"][:,12], "2p R2 G''":data["power"][:,13], 
                              "number":np.real(data["akz_Ta"][:,-1]), "phase":data["df_ap"][data["df_ap"]["f"] == 1]["phase"]})
    
    # df_power = pd.DataFrame({"a G'":data["power"][:,0], "b G'":data["power"][:,1], "c G'":data["power"][:,2], 
    #                          "a G''":data["power"][:,3], "b G''":data["power"][:,4], "c G''":data["power"][:,5], 
    #                           "a G' and G''":data["power"][:,6], "b G' and G''":data["power"][:,7], "c G' and G''":data["power"][:,8], 
    #                           "R2 G'":data["power"][:,9], "R2 G''":data["power"][:,10], "R2 G' and G''":data["power"][:,11],
    #                          "2p a G'":data["power"][:,12], "2p b G'":data["power"][:,13], "2p a G''":data["power"][:,14],
    #                          "2p b G''":data["power"][:,15], "2p R2 G'":data["power"][:,16], "2p R2 G''":data["power"][:,17], 
    #                           "number":np.real(data["akz_Ta"][:,-1]), "phase":data["df_ap"][data["df_ap"]["f"] == 1]["phase"]})
    
    
    df_akz = pd.DataFrame({"alpha": np.real(data["akz_Ta"][:,0]), "kappa":np.real(data["akz_Ta"][:,1]), "zeta":np.real(data["akz_Ta"][:,2]), 
                           "Ta":np.real(data["akz_Ta"][:,3]), "tau":np.real(data["akz_Ta"][:,4]), "Ta_shifted":np.real(data["akz_Ta"][:,5]), 
                           "tau shifted":np.real(data["akz_Ta"][:,6]), "R square":np.real(data["akz_Ta"][:,7]), "number":np.real(data["akz_Ta"][:,-1]), 
                           "phase":data["df_ap"][data["df_ap"]["f"] == 1]["phase"]})
    
    df_pot = pd.DataFrame({"A": np.real(data["s_pot"][:,0]), "B": np.real(data["s_pot"][:,1]), "alpha": np.real(data["s_pot"][:,2]), 
                            "beta": np.real(data["s_pot"][:,3]), "R2": np.real(data["s_pot"][:,4]), "phase":data["df_ap"][data["df_ap"]["f"] == 1]["phase"]})
        
    output = {"active":df_active_noNa,"passive":df_passive_noNa, "ap":df_ap_noNa, "akz": df_akz, "power": df_power, "s_pot": df_pot}

    return output
   
 
def plot_time(data,frequency = 10):
    "plots response against time"
    
    #define variables
    active_time,passive_time = [],[]
    response,PSD, number = [], [], 0
    G1, G2 = [], []
    eeff = []
    fig,ax = plt.subplots(5,5)
    fig2, ax2 = plt.subplots(5,5)
    fig1, ax1 = plt.subplots(5,5)#, sharey=True)
    i = j = 0
    #go through unique indices 
    for idx in data["df_ap"]["number"].drop_duplicates():
        cell_data = data["active"][data["active"][:,-2] == idx]
        cell_data_p = data["passive"][data["passive"][:,-2] == idx]
        cell_data_ap = getNum(data, idx)
        #get frequencies of the arrays
        freq_a = np.argmin(np.abs(cell_data[:,4]-frequency))
        freq_p = np.argmin(np.abs(cell_data[:,4]-frequency))
        freq_ap = np.argmin(np.abs(cell_data_ap["f"]-frequency))
        #get times the data was acquired
        a_time = np.real(cell_data[cell_data[:,4] == cell_data[freq_a,4],-3:-1])
        p_time = np.real(cell_data_p[cell_data_p[:,4] == cell_data_p[freq_p,4],-3:-1])
        #get response, PSD, G' and G'' for data
        response.append(np.abs(np.imag(cell_data[cell_data[:,4] == cell_data[freq_a,4],6])))
        PSD.append(np.abs(np.real(cell_data_p[cell_data_p[:,4] == cell_data_p[freq_p,4],6])))
        G1.append(cell_data_ap[cell_data_ap["f"] == cell_data_ap["f"].iloc[freq_ap]]["G' y"].to_numpy())
        G2.append(cell_data_ap[cell_data_ap["f"] == cell_data_ap["f"].iloc[freq_ap]]["G'' y"].to_numpy())
        eeff.append(cell_data_ap[cell_data_ap["f"] == cell_data_ap["f"].iloc[freq_ap]]["eff E y"].to_numpy())
        #append times to active and passive time lists and convert to minutes
        if a_time.any():
            active_time.append(a_time[0])
        if p_time.any():
            passive_time.append(p_time[0])
        
    #turn lists into arrays
    active_time = np.array(active_time)
    passive_time = np.array(passive_time)
    #create empty list
    cell = []
    #go through all data and save every unique cell with [day, cell, measure]
    cell.append(data["active"][0,0:2])
    for idx in np.arange(1,len(data["active"])):
        if data["active"][idx,1] != data["active"][idx-1,1]:           
            cell.append(data["active"][idx,0:2])         
       
    idx_legend = []
    
    #go through all unique cells
    for day_cell in cell:
        
        indices = np.real(np.unique(data["active"][np.all(data["active"][:,[0,1]] == list(day_cell),axis = 1),-2]))
        act_cell = np.array([active_time[int(x),0].tolist() for x in indices])
        pas_cell = np.array([passive_time[int(x),0].tolist() for x in indices])
        
        act_cell = act_cell-act_cell[0]
        pas_cell = pas_cell-pas_cell[0]
        nr = [int(np.real(passive_time[int(x),1].tolist())) for x in indices]
        
        if len(pas_cell) > 5:
            ax[i,j].plot(pas_cell/60,np.array(G1)[nr],'-o') 
            ax[i,j].plot(pas_cell/60,np.array(G2)[nr],'-o') 
            
            # ax1[i,j].plot(pas_cell/60, data["akz_Ta"][nr,0], '-o') 
            ax1[i,j].plot(pas_cell/60, data["s_pot2"][nr, 0], '-o')
            ax1[i,j].plot(pas_cell/60, data["eeff_pl"][nr, 2], '-o')
            
            ax2[i,j].plot(pas_cell/60,np.array(eeff)[nr],'-o')
            
            
            number = number+1
            idx_legend.append(nr[0])
            if np.mod(number,1) == 0:
                ax[i,j].legend(idx_legend)            
                idx_legend = []
                j = j+1
                if j == 5:
                    j = 0
                    i = i+1

    fig.suptitle("G' and G'', frequency: " + str(frequency) + " Hz")
    fig1.suptitle("alpha")
    fig2.suptitle("effective energy, frequency: " + str(frequency) + " Hz")

    
def plot_cells(data, frequency):
    
    fig,ax = plt.subplots(3,3,figsize = (19,9.5))
    f = data["df_ap"]["f"].drop_duplicates()
    l = len(f)
    cmap = np.vstack((plt.cm.tab20(np.linspace(0,1,20)), plt.cm.tab20(np.linspace(0,1,20)), plt.cm.tab20(np.linspace(0,1,20))))
    i = j = 0
    for day in np.arange(0, 42):
        data_day = data["df_ap"][data["df_ap"]["day"] == day]
        for phase in data_day["phase"].drop_duplicates().astype(int): #np.unique(data_day[:,-1]).astype(int):
            ax[i,j].loglog(f, np.reshape(data_day[data_day["phase"] == phase]["response y"].to_numpy(), [int(len(data_day[data_day["phase"] == phase])/l),l]).T, color = cmap[phase])
            ax[i,j].set_ylim([5, 10**6])
        j = j+1
        if j == 3:
            j = 0
            i = i+1
            if i == 3:
                i = j = 0
                fig,ax = plt.subplots(3,3,figsize = (19,9.5))
      
    # sb.swarmplot(x = "phase", y = "response y", data = data["df_ap"][data["df_ap"]["f"] == 0.2], hue = "day")
    # sb.boxplot(x = "phase", y = para[0], data = para[1][np.logical_and(np.isin(para[1]["phase"], numbers), para[2] )], ax = ax, color = 'white', showfliers = False, order = numbers)#, showmeans = True)
       
            
def chose_passive(data, response = 0,fixed_slope = 0):
    "opens GUI to remove passive measurement outliers"
    
    #definition of variables
    output = []
    measurement = 0
    measurement_lst = np.unique(data["passive"][:,-2])
    # measurement_lst = np.arange(563, 1525)
    
    #method to plot next graph
    def inital():
        #call variables
        nonlocal measurement
        nonlocal measurement_lst
        nonlocal output
       
        #clear plot
        a.clear()
        x = []
        y = []
        slope = []
        
        #end condition
        if measurement == (len(measurement_lst)):
            canvas.draw()
            print("all data processed")
            bt1.config(state = "disabled")
            lis = []
            lis.append(var0.get())
            lis.append(var1.get())
            lis.append(var2.get())
            lis.append(var3.get())
            lis.append(var4.get())
            
            output.append(np.hstack((np.array(lis*np.arange(1,6)),int(np.real(measurement_lst[measurement-1])))))
            
            return

        #create data for plot of passive curves
        x = np.real(data["passive"][np.all(data["passive"][:,[0,1,2,3,-2]] == [0,0,0,0,0],axis = 1),4])
        
        length = len(x)
        
        
#        for measurement in range(np.max(data["passive"][:,-2])):
#        y = data["passive"][data["passive"][:,-2] == measurement,6]
        if response:
            if fixed_slope:
                y = data["passive"][data["passive"][:,-2] == measurement_lst[measurement],6]/((fixed_slope*10**6)**2)*data["passive"][data["passive"][:,-2] == measurement_lst[measurement],4]*np.pi/(4.2800104364899995e-21)
                y = np.transpose(np.reshape(y,[int(len(y)/length),length]))
            else:
                y = data["passive"][data["passive"][:,-2] == measurement_lst[measurement],6]/((np.imag(data["passive"][data["passive"][:,-2] == measurement_lst[measurement],9])*10**6)**2)*data["passive"][data["passive"][:,-2] == measurement_lst[measurement],4]*np.pi/(4.2800104364899995e-21)
                y = np.transpose(np.reshape(y,[int(len(y)/length),length]))

        else:
            y = data["passive"][data["passive"][:,-2] == measurement_lst[measurement],6]
            y = np.transpose(np.reshape(y,[int(len(y)/length),length]))
#        for p in range(len(data[k][o]["passive"])):
#            slope.append(np.round(data[k][o]["passive"][p]["slopes"],2))
#            for u in range(len(data[k][o]["passive"][p]["fl_y"])):
#                temp = log_bin(data[k][o]["passive"][p]["fl_y"][u]["f"],data[k][o]["passive"][p]["fl_y"][u]["PSD"],passive_binning)
#                x.append(temp["x"][1:])
#                y.append(temp["y"][1:])
        
        #plot 
        a.loglog(x,y)
        if response:
            a.set_ylim([10**1,10**7])     
            a.loglog([0.1,10**4],[10**6,1.1*10**2],'--k')
        else:
            a.set_ylim([10**-12,10**-1.5])
        
        #show slope in label and set counter to cell/measurement
        lblVar.set(slope)
#        current_cell = "cell "+ str(measurement_lst[measurement+1]) + " measure xx"

        cell_no = data["passive"][data["passive"][:,-2] == measurement_lst[measurement],0:3][0]
        current_cell = "day: " + str(int(np.real(cell_no[0])+1)) + ", cell: " + str(int(np.real(cell_no[1])+1)) + ", measurement: " + str(int(np.real(cell_no[2])+1))
        varNum.set(current_cell)
        
        #save decision for measurements to remove
        lis = []
        lis.append(var0.get())
        lis.append(var1.get())
        lis.append(var2.get())
        lis.append(var3.get())
        lis.append(var4.get())
        
        output.append(np.hstack((np.array(lis*np.arange(1,6)),int(np.real(measurement_lst[measurement-1])))))
               
        #uncheck all checkboxes
        var0.set(0)
        var1.set(0)
        var2.set(0)
        var3.set(0)
        var4.set(0)
        
        #set title/labels for plot and draw figure
        a.set_title ("passive rheology")    
        a.set_ylabel("PSD")
        a.set_xlabel("frequency [Hz]")
        a.legend([0,1,2,3,4])
        canvas.draw()

        measurement = measurement+1
            
    
    #create tk window and variables for checkboxes and labels
    root =  tk.Tk() 
    
    varNum = tk.StringVar()
    varNum.set("0")
    lblVar = tk.StringVar()
    var0 = tk.IntVar()
    var1 = tk.IntVar()
    var2 = tk.IntVar()
    var3 = tk.IntVar()
    var4 = tk.IntVar()
    #root.geometry('800x500')
    
    fig = Figure(figsize = (6,6))
    a = fig.add_subplot(111)
    #create labels, figure, button and checkboxes   
    lblNumber =  tk.Label(root,text = "0",textvariable = varNum)
    lblNumber.pack()
    
    canvas = FigureCanvasTkAgg(fig, master = root)
    canvas.get_tk_widget().pack()
    canvas.draw()
    
    bt1 = tk.Button(root,text = "next",command = inital)
    bt1.pack(pady = 10)  
    
    sub = tk.Frame(root)
    sub.pack()
    ck0 = tk.Checkbutton(sub,text = "0",variable = var0)
    ck1 = tk.Checkbutton(sub,text = "1",variable = var1)
    ck2 = tk.Checkbutton(sub,text = "2",variable = var2)
    ck3 = tk.Checkbutton(sub,text = "3",variable = var3)
    ck4 = tk.Checkbutton(sub,text = "4",variable = var4)
    lbl1 = tk.Label(root,textvariable = lblVar)
    ck0.pack(side = "left")
    ck1.pack(side = "left")
    ck2.pack(side = "left")
    ck3.pack(side = "left")
    ck4.pack(side = "left",padx = 5,pady = 5)
    lbl1.pack(pady = 5)
         
#    inital()
#    output = [[]]
#    k = 0
#    o = 0
    root.mainloop()
    #return list of measurements to remove
    output = np.vstack(output[1:])
    
    return output

def remove_chosen_passive(data,pas_list):
    "remove passive data that was removed from 'chose_passive'"   
    output = {}
    output["active"] = data["active"]*[1]
    output["passive"] = data["passive"]*[1]
    output["akz_Ta"] = data["akz_Ta"]*[1]
    output["msd"] = data["msd"]*[1]
    output["vanHove"] = data["vanHove"]*[1]
    
    if "passive_f" in data:
        output["passive_f"] = data["passive_f"]*[1]
    if "_description" in data:
        output["_description"] = data["_description"]
    if "_folder" in data:
        output["_folder"] = data["_folder"]
    
    for idx in range(len(pas_list)):      
        #check where pas_list is non zero and go through these repeats
        for repeat in pas_list[idx,:5][(pas_list[idx,:5]!=0)]:
            output["passive"] = np.delete(output["passive"],np.where(np.all(output["passive"][:,[3,-2]] == [repeat-1,pas_list[idx,-1]],axis = 1)),0)
            if "passive_f" in data:
                output["passive_f"] = np.delete(output["passive_f"],np.where(np.all(output["passive_f"][:,[3,-2]] == [repeat-1,pas_list[idx,-1]],axis = 1)),0)
                output["msd"] = np.delete(output["msd"],np.where(np.all(output["msd"][:,[3,-1]] == [repeat-1,pas_list[idx,-1]],axis = 1)),0)
                output["vanHove"] = np.delete(output["vanHove"],np.where(np.all(output["vanHove"][:,[3,-1]] == [repeat-1,pas_list[idx,-1]],axis = 1)),0)
                
    
    return output
        
        

#%%fit functions
def power_law_log(x,a,b):
    return np.log(a*x**b + 1)

def power_law_log2(x,a,b):
    return np.log(a*x**b)

def double_plaw(x, ca, cb, a, b):
    return np.log(ca*x**a + cb*x**b)
#beadsize = beadsize
#lambda f, a, k, z: G12prime(f, a, k, z, beadsize), f, a, k, z

def G_power(f, a, b, c):
    return np.log(a*f**b + c)

def G_power_2_para(f, a, b):
    return np.log(a*f**b)

def power_law_model(f, A, B, a, b, f0):
    return A*((i*f)/f0)**a + B*((i*f)/f0)**b

def G_prime(f, a, k, z):
    return np.log((k/(3*np.pi*bead_diameter))*((((2*np.pi*f*z)**a)*np.cos((np.pi*a)/2))+1))
    
def G_2prime(f, a, k, z):
    return np.log((k/(3*np.pi*bead_diameter))*((2*np.pi*f*z)**a)*np.sin((np.pi*a)/2))

def G12prime(f, a, k, z, d, length):
    """"fits model from Whylie Ahmed's paper to G' and G''
    G'(w) = k/ 6*pi *R *[(w*z)**a *cos((pi*a)/2)+1]
    G''(w) = k/ 6*pi *R *(w*z)**a *sin((pi*a)/2)
    alpha, kappa and zeta are fitted once for both G' and G''
    input: f: values of function, d: diameter of particle in m"""
    
    def G_prime_internal(f, a, k, z):
        return np.log((k/(3*np.pi*d))*((((2*np.pi*f*z)**a)*np.cos((np.pi*a)/2))+1))
    
    def G_2prime_internal(f, a, k, z):
        return np.log((k/(3*np.pi*d))*((2*np.pi*f*z)**a)*np.sin((np.pi*a)/2))
    
    extract1 = f[:int(length)]
    extract2 = f[int(length):]

    result1 = G_prime_internal(extract1, a, k, z)
    result2 = G_2prime_internal(extract2, a, k, z)
    
    return np.append(result1,result2)

def power2(f, a, b, c, l):
    
    extract1 = f[:int(l)]
    extract2 = f[int(l):]
    
    result1 = G_power(extract1, a, b, c)
    result2 = G_power(extract2, a, b, c)
    
    return np.append(result1, result2)


def G1_pot(f, A, B, alpha, beta):
    func = (A* (2 * np.pi * f)**alpha) * np.cos(alpha * np.pi/2) + (B* (2 * np.pi * f)**beta) * np.cos(beta * np.pi/2)
    func[func <0] = np.nan
    return np.log(func)

def G2_pot(f, A, B, alpha, beta):
    func = (A* (2 * np.pi * f)**alpha) * np.sin(alpha * np.pi/2) + (B* (2 * np.pi * f)**beta) * np.sin(beta * np.pi/2)
    func[func < 0] = np.nan
    return np.log(func)

def fit_new_eeff(f, Tcv, v, A, B, alpha, beta):
    return 1 + (Tcv * np.sin(np.pi*v/2)*(2*np.pi*f)**v)/(A * np.sin(np.pi*alpha/2)*(2*np.pi*f)**alpha + B * np.sin(np.pi*beta/2)*(2*np.pi*f)**beta)

def fit_new_eeff2(f, Tcv, v, A, alpha):
    return 1 + ((Tcv * np.sin(np.pi*v/2))/(A* np.sin(np.pi*alpha/2)) * (2*np.pi*f)**(v-alpha))

def springpot(f, A, B, alpha, beta, length):
    "fits springpot model to G' and G''"
    "G' (w) "
    extract1 = f[:int(length)]
    extract2 = f[int(length):]

    result1 = G1_pot(extract1, A, B, alpha, beta)
    result2 = G2_pot(extract2, A, B, alpha, beta) 
    return np.append(result1, result2)

def gaussian(x, a, mu, sig):
    I = a * np.exp(-(x-mu)**2/(2*sig**2))
    return I 

def fit_eeff(f, Ta, tau, alpha, zeta):
    return 1 + 1/((2*np.pi*f *zeta)**(3*alpha-1) *np.sin(np.pi*alpha/2)) * (Ta)/(1+(2*np.pi*f*tau)**2)


    

def fit_s_active(f, tau, kbTa, alpha, kappa, zeta):
    return np.log((2*kappa*zeta)/(2*np.pi*f*zeta)**(2*alpha) * (kbTa*const.k*310)/(1+(2*np.pi*f*tau)**2))

def gof(x,y,fit,func):
    ss_res = np.nansum((y-func(x,*fit))**2)
    ss_tot = np.nansum((y-np.nanmean(y))**2)
    r2_G12 = 1-(ss_res/ss_tot)
    return r2_G12


#%% van Hove and MSD 

def calcVH(reshaped_data, slopes):
    "calculates van Hove distribution for lagtimes 1, 0.1 and 0.01 sec in x and y"
    x1, x01, x001, y1, y01, y001 = [], [], [], [], [], [] 
    
    #get bins (move to middle of bin)     
    edges = vanhove([0,1], 1)[1][:-1]
    bins = edges + (edges[1] - edges[0])/2
    
    #loop through data, divide by slope and calculate van Hove for 1, 0.1 and 0.01 sec lagtime
    for idx, data in enumerate(reshaped_data):
        x1.append(vanhove(data[0,:]/slopes[idx][0],1)[0])
        x01.append(vanhove(data[0,:]/slopes[idx][0],0.1)[0])
        x001.append(vanhove(data[0,:]/slopes[idx][0],0.01)[0])
        
        y1.append(vanhove(data[1,:]/slopes[idx][1],1)[0])
        y01.append(vanhove(data[1,:]/slopes[idx][1],0.1)[0])
        y001.append(vanhove(data[1,:]/slopes[idx][1],0.01)[0])
       
    return np.hstack((np.repeat(range(int(len(np.vstack((x1)))/len(bins))),len(bins))[:,np.newaxis],            #repeat
                      np.tile(bins,int(len(np.vstack((x1)))/len(bins)))[:,np.newaxis], np.vstack((x1)),         #bins
                      np.vstack((x01)), np.vstack((x001)), np.vstack((y1)), np.vstack((y01)), np.vstack((y001)), #distribution (1, 0.1, 0.01)
                      np.repeat(np.real(identifier),len(np.vstack(x1)))[:,np.newaxis]))                         #identifier
    
       
def vanhove(input_data, lagtime, window_width = 20, move_to_0 = True):
    
    #calculate index from lagtime with acquisition rate of 50 kHz
    lag = int(lagtime*50000)
    
    #filter noise in fourier space
    # input_data = removeNoise(input_data)
    
    #smooth with window of window_size
    cum_sum = np.cumsum(np.insert(input_data,0,0))
    pos = (cum_sum[window_width:] - cum_sum[:-window_width])/window_width
    
    #calculate displacement for specific lagtime
    disp = np.subtract(pos[:-lag], pos[lag:])
    
    #use fixed bin size 
    global_bins = np.arange(-.5, 0.501, 20/2**13)
    
#    bi = np.arange((b[1]*10**14)/2, 0.5, (b[1]*10**14))
#    bounds = np.hstack(([-bi[::-1], bi]))
#    global_bins = bounds
    #calculate histogram as density
    vh = np.histogram(disp, bins = global_bins)[0]
    
    #if move_to_0 is active, center of of distribution will be shifted to 0 
    if move_to_0:
        #shift edges by half of edge distance to get middle of bins
        bin_mid = global_bins[:-1] + (global_bins[1] - global_bins[0])/2
        #find bin with lowest distance to center of mass
        count_mean = np.abs(np.sum(bin_mid * vh.T)/sum(vh) - bin_mid).argmin()
        #move center of mass to middle of global_bins
        shift = count_mean - int(len(global_bins-1)/2)
        if shift >= 0:
            vh = np.hstack((vh[shift:], np.repeat(0,shift)))
        else:
            vh = np.hstack((np.repeat(0, -shift), vh[:shift]))
    
    return vh[:,np.newaxis], global_bins

#out = calcVH(d,[slopes])
#plt.plot(out[:,1],out[:,2])

def MSD(reshaped_data, slopes):
    "calculates the mean squared displacement for given passive data" 
    
    msd = []
    #set lagtimes for which MSD is calculated
    lagtimes = np.unique(np.round([10**power for power in np.arange(0, 5.75, .125)]).astype(int))
    lagtimes[-3] = 5*50000
    #go through data and append into msd list
    for idx, data in enumerate(reshaped_data):
        # data = np.array([removeNoise(single_data) for single_data in data])
        temp_array = np.zeros([len(lagtimes),2])
        #calculate msd for every lagtime
        for lag_idx, lagtime in enumerate(lagtimes):
            disp = np.subtract((data[0:2,:]/np.array(slopes[idx])[:,np.newaxis])[:,:-lagtime], (data[0:2,:]/np.array(slopes[idx])[:,np.newaxis])[:,lagtime:])
            temp_array[lag_idx,:] = np.mean(disp**2,1)
        msd.append(temp_array)
       
    length_msd = int(len(np.vstack((msd)))/len(lagtimes))
    return np.vstack((np.repeat(range(length_msd),len(lagtimes)),                   #repeats
                      np.tile(lagtimes, length_msd), np.vstack((msd)).T,            #lagtimes and msd
                      np.repeat(np.real(identifier), len(np.vstack((msd)))))).T     #identifier
    
def mergeData(data_in1, data_in2, same_phase = False):
    data1 = copy.deepcopy(data_in1)
    data2 = copy.deepcopy(data_in2)
    data2["_description"] = {}
    interpolate_active(data1)
    interpolate_active(data2)
    max_nr = np.max(data1["active"][:,-2]) + 1
    if same_phase:
        max_phase = 0
    else:
        max_phase = np.max(data1["active"][:,-1]) + 1
    max_day = np.max(data1["active"][:,0]) + 1
    
    data1["_folder"].extend([data2["_folder"]])  
    
    data2["active"][:,-2] = data2["active"][:,-2] + max_nr
    data2["active"][data2["active"][:,-1] != 0,-1] = data2["active"][data2["active"][:,-1] != 0,-1] + max_phase
    data2["active"][:,0] = data2["active"][:,0] + max_day
                                                              
    data2["passive"][:,-2] = data2["passive"][:,-2] + max_nr
    data2["passive"][data2["passive"][:,-1] != 0,-1] = data2["passive"][data2["passive"][:,-1] != 0,-1] + max_phase
    data2["passive"][:,0] = data2["passive"][:,0] + max_day
    
    data2["passive_f"][:,-2] = data2["passive_f"][:,-2] + max_nr
    data2["passive_f"][data2["passive_f"][:,-1] != 0,-1] = data2["passive_f"][data2["passive_f"][:,-1] != 0,-1] + max_phase
    data2["passive_f"][:,0] = data2["passive_f"][:,0] + max_day
    
    data2["akz_Ta"][:,-1] = data2["akz_Ta"][:,-1] + max_nr
    data2["msd"][:,-1] = data2["msd"][:,-1] + max_nr
    data2["msd"][:,0] = data2["msd"][:,0] + max_day
    data2["vanHove"][:,-1] = data2["vanHove"][:,-1] + max_nr
    data2["vanHove"][:,0] = data2["vanHove"][:,0] + max_day
    data2["scans"][:,0] = data2["scans"][:,0] + max_day
    data2["raw_response"][:,-1] = data2["raw_response"][:,-1] + max_nr
    if "removedPassive" in data2:   
        data2["removedPassive"][:,-1] = data2["removedPassive"][:,-1] + max_nr
    else:
        data2["removedPassive"] = np.zeros([len(data2["akz_Ta"]),6])
        # data2["removedPassive"][:,-1] = range(max_nr, len(data2["akz_Ta"])+max_nr)
    if "excludeGrubbs" in data2:
        data2["excludeGrubbs"] = list(np.array(data2["excludeGrubbs"]) + max_nr)
    else: 
        data2["excludeGrubbs"] = []
    if "excludeList" in data2:
        data2["excludeList"] = list(np.array(data2["excludeList"]) + max_nr)
        data2["excludeManual"] = list(np.array(data2["excludeManual"]) + max_nr)
    else:
        data2["excludeList"] = []
        data2["excludeManual"] = []
    

    data_new = {"_description": data1["_description"], "_folder": data1["_folder"], "active":np.vstack((data1["active"], data2["active"])), 
                "passive":np.vstack((data1["passive"], data2["passive"])),
                "passive_f":np.vstack((data1["passive_f"], data2["passive_f"])), "akz_Ta":np.vstack((data1["akz_Ta"], data2["akz_Ta"])),
                "msd":np.vstack((data1["msd"], data2["msd"])), "vanHove":np.vstack((data1["vanHove"], data2["vanHove"])), 
                "power":np.vstack((data1["power"], data2["power"])), "scans":np.vstack((data1["scans"], data2["scans"])), 
                "raw_response":np.vstack((data1["raw_response"], data2["raw_response"])), "excludeGrubbs": data1["excludeGrubbs"] + data2["excludeGrubbs"],
                "excludeList": data1["excludeList"] + data2["excludeList"], "excludeManual": data1["excludeManual"] + data2["excludeManual"],
                "removedPassive": np.vstack((data1["removedPassive"], data2["removedPassive"]))}
    return data_new


def removeNoise(dat):
    fname = r'\\spacelord2.uni-muenster.de\users2\sebastian\Rheology\20200117\cell01\measure01\Passive_microrheology0002\Passive_microrheology00.lvb'
    
    rawdata = np.fromfile(fname, dtype = '>f8')
    dat = np.reshape(rawdata, [3, 500000])
    
   
    p = len(dat[1,:])
    
    # p = len(dat)
    ft = np.fft.fft(dat, p)
    f = 50000/p*(np.arange(0, p/2+1))
    a = ft[0:int(p/2)+1]

    b = a*[1]
    mean_value = np.mean(abs(a))
    threshold = mean_value
    b[20000:][abs(b[20000:]) > threshold] = 0;
    # b[abs(b) > threshold] = 0;
    filtered = np.fft.ifft(b, p)
    # plt.figure()
    # plt.plot(f, a)    
    
    # plt.plot(f, b)
    # plt.figure()
    # plt.plot(dat)
    # plt.plot(filtered)
    
    return filtered

def getNum(data, num):
    return data["df_ap"][data["df_ap"]["number"] == num]

def getPhase(data, phase, exclude = [], mean_per_cell = False, num_out = False, para = False):
    #get all data with specified phase
    phaseData = data["df_ap"][data["df_ap"]["phase"] == phase]
    #remove excluded datasets from data
    phaseData = phaseData[~np.isin(phaseData["number"], exclude)]
    
    numbers = phaseData["number"].drop_duplicates().to_numpy().astype(int)
    if para:
        paraOut = data["fitParameters"].iloc[numbers]
    #calculate a mean per cell for every separate cell
    if mean_per_cell:
        unique_cell = phaseData[["day","cell"]].drop_duplicates()
        one_cell, numbers  = [], []
        paraOut = pd.DataFrame()
        one_cell = pd.DataFrame()
        for cell in unique_cell.iloc:
            #reshape into 3D array, get mean
            numbers.append(phaseData[np.all(phaseData[["day", "cell"]] == cell, axis = 1)]["number"].drop_duplicates().to_numpy().astype(int))
            # one_cell.append(np.nanmean(np.reshape(phaseData[np.all(phaseData[["day", "cell"]] == cell, axis = 1)].to_numpy(), [-1, 16, 29]), 0))
            one_cell = one_cell.append(phaseData[np.all(phaseData[["day","cell"]] == cell, axis =1)].groupby("f").mean())
            if para:
                paraOut = paraOut.append(data["fitParameters"][np.isin(data["fitParameters"]["number"], numbers[-1])].mean(), ignore_index = True)
            
        # cols = ["day", "cell", "measure", "f", "response x", "response y", "response y scan", "PSD x", "PSD y", "eff E x", 
        #         "eff E y", "eff E y scan", "eff E y fit", "G' x","G' y", "G' y scan", "G'' x", "G'' y", "G'' y scan","det slope x", "det slope y", 
        #         "det slope x scan", "det slope y scan", "shift x", "shift y", "beadsize", "time", "number", "phase"]
        # phaseData = pd.DataFrame(np.vstack(one_cell), columns = cols)
        phaseData = one_cell
        # phaseData["f"] = phaseData.index
    
    if num_out:
        if para:
            return phaseData, paraOut
        return phaseData, numbers
    else:
        return phaseData


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped
