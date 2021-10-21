# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:03:04 2021

@author: Sebastian

program to analyse rheological data from optical tweezers measurements
"""



import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import scipy.io
import time




##Hardcoded variables
#unique identifier
identifier = 0
#frequencies of active measurements
frequency_passive = np.array([.1, .2, .5, 1, 2.2, 4.6, 10, 21.5, 46.4, 100, 215, 464, 1000, 2154, 4642, 10000])



#%% This part contains functions to read and pre-analyse in the data of active and passive rheology

#read data of active rheology measurement
def get_response(folder_path, repeat_number = 0):
    """
    Calculates response function of active microrheology experiments.
    
    Parameters
    ----------
    folder_path : str
        A folder that contains rheological data of an active rheology experiment.
    repeat_number : TYPE, optional
        An index for multiple repeats of the same measurement. The default is 0.

    Returns
    -------
    active_response : array
        DESCRIPTION.
    raw_response : array
        DESCRIPTION.

    """
    
    #Get paths of all active rheology experiments. Files are named 'Active_microrheologyXX.lvb'.
    response_paths=[folder_path + '/' + k for k in os.listdir(folder_path) if (k.startswith('Active_microrheology') and k.endswith('.lvb')) ]   
           
    #get frequency, responses in x and y, slopes in x and y, a list with all the trap positions, a list with times, the beadsize and the raw response
    frequency, response_x, response_y, slopes, trap_list, time_list, beadsize, raw_response = get_modulus(response_paths)
      
    #convert diameter to meter
    beadsize = beadsize * 10**-6
    
    #set repeat number and unique identifier
    repeat = np.ones(len(frequency)) * repeat_number
    ID = np.ones(len(frequency)) * identifier
  
    #calculate shear modulus from generalized Stokes-Einstein-Equation: G* = 1/(6pi*R*response)
    G_x = 1/(3*np.pi*beadsize*response_x)
    G_y = 1/(3*np.pi*beadsize*response_y)
  
    beads = np.repeat(beadsize, len(frequency))
    
    #measurement in that cell
    zero = np.zeros(len(frequency))

    #save as array with: day, cell, particle, repeat, frequency, response in x, response in y, G in x, G in y, detector slope in x, 
    #                                         detector slope in y, bead diameter in m, trap position,  time in seconds, unique identifier, phase
    active_response = np.transpose(np.vstack([zero, zero, zero, repeat, frequency, response_x, response_y, G_x, G_y, slopes[:,0], 
                                              slopes[:,1], beads, trap_list, time_list, ID, zero]))
    
    raw_response[:,-1] = ID[0]
    
    return active_response, raw_response


def get_modulus(file_path, power_law_fit = True):
    """
    
    Parameters
    ----------
    file_path : str
        Paths of active rheology measurements.
    power_law_fit : bool, optional
        Decision whether fits to the raw data are being calculated. The default is True.

    Returns
    -------
    None.

    """

    #define empty lists for buffer, frequency, number of channels (n_chan), acquisition rate, slopes, trap position, times and beadsize
    buffer,frequency,n_chan,acq_rate, slope_out, trap_list, time_list, beadsize = [], [], [], [], [], [], [], []
    #define emtpy lists for cuts of raw data to perform fit on and and a list that defines noise
    raw_cut, noise = [], []
    #define empty lists for resulting responses
    response_x, response_y = [],[]

    ###Hardcoded conversion values from voltage to force of the lunam in x and y direction. This is precalibrated.
    alpha_x = 4.68*10**-11
    alpha_y = 4.85*10**-11
    
    for file in file_path:
        #read in file as big endian
        rawdata = np.fromfile(file,dtype = '>f8')
        #read in metadata (same name, but ends in 'metadata.txt')
        metadata_path = file[:-6]+"_metadata" + file[-6:-4] + ".txt"

        table = np.loadtxt(metadata_path,dtype = str)
    
        #save values from metadata
        buffer.append(float(table[table[:,0] == "Buffer_size",1]))
        frequency.append(float(table[table[:,0] == "Frequency",1]))
        n_chan.append(int(table[table[:,0] == "Number_channels",1]))
        acq_rate.append(float(table[table[:,0] == "Acq_rate",1]))
        time_hms = str(table[table[:,0] == "Date_time",1])[-10:-2]
        time_list.append(int(time_hms[:2])*3600 + int(time_hms[3:5])*60 + int(time_hms[-2:]))
        slope_x = (float(table[table[:,0] == "Detector_slope_x",1]))
        slope_y = (float(table[table[:,0] == "Detector_slope_y",1]))
        trap_str = table[table[:,0] == "Trap_pos(px)",1][0]
        beadsize = float(table[table[:,0] == "radius(um)",1])
        trap = trap_str.split(';')
        trap_list.append(float(trap[0]) + 1j* float(trap[1]))
        slope_out.append([slope_x,slope_y])
        #reshape data according to metadata
        data = np.reshape(rawdata,[n_chan[-1],int(len(rawdata)/n_chan[-1])])
        
        #use x and y signal of lunam (force) and stationary detector (displacement)
        #rows of resulting array correspond to: Lunam in x: 0, in y: 1; detector in x: 2, in y: 3
        lunam_and_detector = data[[0,1,3,4],:]
        
        #calculate fourier transformation of detector and lunam signal
        fourier_lunam_detector = np.fft.fft(lunam_and_detector)

        if power_law_fit:
            #get amplitudes of lunam and detector
            amplitudes = np.abs(fourier_lunam_detector)[:,1:]
            amplitude_detector_y = amplitudes[3,:]
            
            #get frequencies of fourier transformed
            freq = np.fft.fftfreq(int(buffer[-1]), 1/acq_rate[-1])[1:]
            l = len(freq)//2
            freq_positive = freq[0:l]
    
            #get index of frequencies 0.5 and 5 times current frequency
            f0 = np.abs(freq_positive - frequency[-1]*0.5).argmin()
            f1 = np.abs(freq_positive - frequency[-1]*5).argmin()
            
            f_arg = np.abs(freq_positive[f0:f1] - frequency[-1]).argmin()
            current_f = np.abs(freq - frequency[-1]).argmin()

            range_wo_frequency = np.arange(f0, f1) != current_f
            try:
                # print(file)
                fit_pl, pcov = curve_fit(log_power_law, freq[f0:f1][range_wo_frequency], 
                            np.log((2/amplitude_detector_y.size)*amplitude_detector_y[f0:f1][range_wo_frequency]))#, maxfev = 10000)
                norm_d = (2/amplitude_detector_y.size)*amplitude_detector_y[f0:f1]/(np.exp(log_power_law(freq[f0:f1], *fit_pl)))
                sd = np.std(norm_d)
                mean = np.mean(norm_d)
                f_norm_d = norm_d[f_arg]
                noise.append(f_norm_d > mean + sd)
                
                raw_cut.append(np.vstack((freq[f0:f1], norm_d, np.repeat(f_arg, len(norm_d)), np.repeat(frequency[-1], len(norm_d)), np.repeat(0, len(norm_d)))).T)
                
            except:
                print("no fit for power law on raw data " + file)
                fit_pl = [1,1]
                noise.append(False)
                
            #active
    
        else:
            noise.append(0)
        
        #find maximum of detector 
        fourier_lunam_detector_max = np.argmax(np.abs(fourier_lunam_detector[:,1:]),1)+1
        
        #Calculate ratio of lunam and detector. Division of maximum of detector signal (direction: x = 2, y = 3) and maximum of lunam signal (x = 0, y = 1)
        ratio_x = fourier_lunam_detector[2, fourier_lunam_detector_max[0]]/fourier_lunam_detector[0, fourier_lunam_detector_max[0]]
        ratio_y = fourier_lunam_detector[3, fourier_lunam_detector_max[1]]/fourier_lunam_detector[1, fourier_lunam_detector_max[1]]
        
        prefactor_x = 1./(10**6*alpha_x*slope_x)
        prefactor_y = 1./(10**6*alpha_y*slope_y)
        
        response_x.append(ratio_x*prefactor_x)
        response_y.append(ratio_y*prefactor_y)
        

    return np.array(frequency), np.array(response_x), np.array(response_y), np.vstack(slope_out), trap_list, time_list, beadsize, np.vstack((raw_cut))


def get_fluctuations(folder_path): 
    """
    Calculates response function of active microrheology experiments.
    
    Parameters
    ----------
    folder_path : str
        Paths of active rheology measurements.

    Returns
    -------
    Array with data.

    """
    
    #find all folders with subfolders that contain passive data
    fluctuation_folders = [dirpath for dirpath, dirnames, filenames in os.walk(folder_path) if "Passive_microrheology" in dirpath]
    passive = []    
    idx_repeat = 0
    
    #go through every folder
    for single_folder in fluctuation_folders:
        #find all files of raw data individual passive measurements
        fluctuation_paths = [single_folder + '/' + k for k in os.listdir(single_folder) if k.startswith('Passive_microrheology') and k.endswith('.lvb')]
        
        if fluctuation_paths:
            #create empty variables to store data in
            f_b,PSDx_b,PSDy_b,t = [],[],[],[]
            det_cmplx, zero, repeat = [], [], []    
            time_list,ID, slopes = [],[], []
            
        #    data = read_raw(fluctuation_paths,new_prog)
            #read in data in big endian encoding
            rawdata = [np.fromfile(file,dtype = '>f8') for file in fluctuation_paths]
            
            #get metadata (same filename, with _metadataXX.txt)
            metadata_path = [file[:-6]+"_metadata" + file[-6:-4] + ".txt" for file in fluctuation_paths]
            for metadata in metadata_path:
                meta_table = np.loadtxt(metadata,dtype = str)
                n_chan = int(meta_table[meta_table[:,0] == "Number_channels",1])
                slope_x = (float(meta_table[meta_table[:,0] == "Detector_slope_x",1]))
                slope_y = (float(meta_table[meta_table[:,0] == "Detector_slope_y",1]))
                rate = (float(meta_table[meta_table[:,0] == "Acq_rate",1]))
                time_hms = str(meta_table[meta_table[:,0] == "Date_time",1])[-10:-2]
                time_list.append(int(time_hms[:2])*3600 + int(time_hms[3:5])*60 + int(time_hms[-2:]))
                slopes.append([slope_x, slope_y])
            
            #reshape data according to channel number
            data = [np.reshape(k,[n_chan,int(len(k)/n_chan)]) for k in rawdata]
            
            for idx in range(len(fluctuation_paths)):     
                          
                #calculate PSD for x and y
                psd_x = power_spectral_density(data[idx][0], rate)
                psd_y = power_spectral_density(data[idx][1], rate)            
      
                #bin PSD to frequencies to hardcoded frequencies
                binned_x = log_f(psd_x["f"],psd_x["PSD"],frequency_passive)
                binned_y = log_f(psd_y["f"],psd_y["PSD"],frequency_passive)
                f_b.append(frequency_passive)
                
                #append PSPs to lists
                PSDx_b.append(binned_x)
                PSDy_b.append(binned_y)
                
                #append time to list
                t.append(np.ones(len(f_b[idx]))*time_list[idx])
                
                #create array with zeros as long as measurement
                zero.append(np.zeros(len(f_b[idx])))
                
                #create list with detector slopes
                det_cmplx.append(np.ones(len(f_b[idx]))*(slopes[idx][0]+1j*slopes[idx][1]))
                
                #append to list with number of repeat
                repeat.append(np.ones(len(f_b[idx]))*(idx_repeat))
    
                #create array with identifier
                ID = np.concatenate([ID,(np.ones(len(f_b[idx]))*identifier)]) 
                
                #increase idx of the current repeat by 1
                idx_repeat += 1
                   
            #turn into arrays
            slopes = np.array(slopes)
            ID = np.array(ID)
            zero = np.hstack(zero)
    
            #array with day, cell, particle, measurement, binned frequency, binned PSD in x, binned PSD in y, slope in x, slope in y, time, identifier, phase
            passive.append(np.transpose(np.vstack([zero,zero,zero,np.hstack(repeat),np.hstack(f_b),np.hstack(PSDx_b),np.hstack(PSDy_b),np.hstack(np.real(det_cmplx)),np.hstack(np.imag(det_cmplx)) ,np.hstack(t),ID,zero])))
            
    #turn list into array
    passive = np.vstack((passive))
    
    return passive


def power_spectral_density(raw,srate):
    """
    Calculates power spectral density from passive microrheology data.
    Parameters
    ----------
    raw : array of floats
        Raw fluctuation data.
    srate : int
        Sampling rate.

    Returns
    -------
    PSD : dictionary
        Power spectral density.

    """    
    
    length = len(raw)
    
    delta_t = 1/ srate
    
    time = np.arange(0,length*delta_t,delta_t)
    T = max(time)
    
    #Fourier transformation of raw signal and multiplication with complex conjugate
    Y = np.fft.fft(raw,length)
    Pyy = Y*np.conj(Y)/ (length*srate)
    
    f = srate/length*(np.arange(0, length/2+1))
    Pyy_out = Pyy[0:int(length/2)+1]
     
    PSD = {"f":f, "PSD":Pyy_out,"T":T}
    
    return PSD


def log_f(x_in, y_in, f):
    """
    Bins data to specified frequencies.
    
    Parameters
    ----------
    x_in : TYPE
        Input raw frequencies.
    y_in : TYPE
        Input raw y-values.
    f : array of floats
        Array with frequencies data is binned to.

    Returns
    -------
    Array with data log-binned to frequencies f.
    
    """
    
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

#%%
def analyse_measurement(folder, idx_measure = 0,suppress = False):
    """
    Analyses active and passive rheology of one measurement and bins passive data.
    
    Parameters
    ----------
    folder : str
        Folder with active and passive measurement.
    idx_measure : int, optional
        Index for current measurement. The default is 0.
    suppress : bool, optional
        If results sould not be printed, set to True. The default is False.


    Returns
    -------
    dict with 4 keys, active and passive data, folder names and raw response data (for filtering)

    """
    
    global identifier
    
    #start timer to estimate time
    start = time.time()

    ###active
    #go through folder and analyse all active measurements
    response_folders = [dirpath for dirpath, dirnames, filenames in os.walk(folder) if "Active_microrheology" in dirpath]    
    active, raw_response = [get_response(folder,idx) for idx,folder in enumerate(response_folders)][-1]
    active = np.vstack(active)
    active[:,2] = idx_measure
    
    ###passive
    #get all passive measurements in current folder
    passive = get_fluctuations(folder)
    passive[:,2] = idx_measure
    
    #increase unique identifer by 1
    identifier = active[-1,-2]+1
    
    #create dictionary with all data
    data = {"active":active, "passive":passive, "_folder":str(folder), "raw_response":raw_response}    
           
    #stop timer and calculate time taken for function
    end = time.time()
    elapsed_time = end - start
    
    if suppress:   
        return
    return data, elapsed_time

#go through subfolders of measurements and run full analysis followed by plot_all
#input: folder of one cell
def analyse_cell(folder,idx_cell = 0):
    "analyses all measurements of one cell by calling analyse_measurements for each measurement"
    
    folder_list = [os.path.dirname(dirpath) for dirpath, dirnames, filenames in os.walk(folder) if "Active_microrheology" in dirpath]
    folder_list_unique = np.unique(folder_list)
    
    data = {"active":[], "passive":[], "_folder":[], "raw_response":[]}
    for idx_measure, current_folder in enumerate(list(dict.fromkeys(folder_list_unique))):
        print("measurement " + str(idx_measure + 1) + " of " + str(len(folder_list_unique)))
        temp, elapsed_time = analyse_measurement(current_folder,idx_measure)
        data["active"].append(temp["active"])
        data["passive"].append(temp["passive"])
        data["_folder"].append(temp["_folder"])
        data["raw_response"].append(temp["raw_response"])
    
    data["active"] = np.vstack(data["active"])
    data["passive"] = np.vstack(data["passive"])
    data["raw_response"] = np.vstack(data["raw_response"])
    data["active"][:,1] = idx_cell
    data["passive"][:,1] = idx_cell       
    
    return data, elapsed_time
    
#go through all measurements of one cell and save as list
#input: folder of one measurement day
def analyse_day(folder,idx_day = 0, progressBar = None, calc_vh_msd = True):
    "analyses all measurements of one day by calling analyse_cell for each cell"
    #doesn't work with new data with os.path.dirname(os.path.dirname(dirpath))
    folder_list = [os.path.dirname(os.path.dirname(dirpath)) for dirpath, dirnames, filenames in os.walk(folder) if "Active_microrheology" in dirpath]
    folder_list_unique = np.unique(folder_list)
    data = {"active":[], "passive":[], "_folder":[], "raw_response":[]}
    for idx_cell, current_folder in enumerate(folder_list_unique):
        print("cell " + str(idx_cell + 1) + " of " + str(len(folder_list_unique)))# + ". (Path: " + current_folder + ")")
        temp, elapsed_time = analyse_cell(current_folder, idx_cell)
        data["active"].append(temp["active"])
        data["passive"].append(temp["passive"])
        data["_folder"].append(temp["_folder"])
        data["raw_response"].append(temp["raw_response"])
        
    data["active"] = np.vstack(data["active"])
    data["passive"] = np.vstack(data["passive"])
    data["raw_response"] = np.vstack(data["raw_response"])
    data["active"][:,0] = idx_day
    data["passive"][:,0] = idx_day
    
    return data, elapsed_time

def analyse_multiple_days(folders):
    """
    
    Parameters
    ----------
    folders : list
        List with str of folders of multiple measurement days.

    Returns
    -------
    data : dict
        Dictionary with all pre-analysed data .

    """
    start = time.time()
    
    global identifier
    identifier = 0
    
    counter = 0
    
    for folder in folders:
        for subdir, dirs, files in os.walk(folder):
            for dirpath in dirs:
                if "Passive_microrheology" in dirpath:
                    counter += 1
    
    print("Raw approximate total time for analysis of " + str(counter) + " measurements: " + str(counter*1.1) + " seconds.")
        
    data = {"active":[], "passive":[], "_folder":[], "raw_response":[]}
    for idx, folder in enumerate(folders):
        print("day " + str(idx + 1) + " of " + str(len(folders)))
        temp, elapsed_time = analyse_day(folder,idx_day = idx)
        print("Approximated total time " + str(elapsed_time*counter) + " seconds. Time elapsed: " + str(time.time() - start) + " seconds")
        data["active"].append(temp["active"])
        data["passive"].append(temp["passive"])
        data["_folder"].append(temp["_folder"])
        data["raw_response"].append(temp["raw_response"])

    data["active"] = np.vstack(data["active"])
    data["passive"] = np.vstack(data["passive"])
    data["raw_response"] = np.vstack(data["raw_response"])

    data["_description"] = {"active": ["day","cell","measurement", "repeat", "frequency", "response in x", 
    "response in y", "G in x", "G in y", "detector slope in x", "detector slope in y", "beadsize", "trap position", 
    "timestamp", "identifier", "phase"],"passive": ["day", "cell", "particle", "measurement", 
    "binned frequency", "binned PSD in x", "binned PSD in y", "slope in x", "slope in y", "time", "identifier", "phase"]}
      
  
    return data

def assign_phase(data, phase):
    "assigns the phase specified by input to corresponding measurement according to day, cell, measurement (in this order)"
            
    for idx, phase_idx in enumerate(phase):                  
        data["active"][data["active"][:,-2] == idx,-1] = phase_idx
        data["passive"][data["passive"][:,-2] == idx,-1] = phase_idx
                
    #remove nan
    data["active"] = data["active"][~np.isnan(data["active"][:,6]),:]    
    data["passive"] = data["passive"][~np.isnan(data["passive"][:,6]),:]

    
def filter_bad(data, std_times = 1):
    """
    Filters bad measurements according to signal-to-noise ratio. 
    Result is added as new key to data dictionary.

    Parameters
    ----------
    data : dict
        Dictionary with data.
    std_times : int, optional
        Standard deviations data has to exceed mean value. The default is 1.

    Returns
    -------
    None.
    """
    
    #create empty list for output
    output = []
    
    #go through every index
    for number in range(int(max(data["raw_response"][:,4])) + 1): 
        #get current data
        current_raw = data["raw_response"][data["raw_response"][:,4] == number,:]
        
        #get current frequencies
        f = np.unique(current_raw[:,3])
        
        #create array with zeros 
        temp_zero = np.zeros(len(f))
        
        #go through every frequency
        for idx, frequency in enumerate(f):
            #get force signal for current frequency
            current_f = current_raw[current_raw[:,3] == frequency, :]
        
            #calculate mean and standard deviation of force signal
            std = np.std(current_f[:,1])        
            mean = np.mean(current_f[:,1])
            
            #check if force signal at current frequency exceeds mean signal by standard deviation*std_times
            if ((mean + std * std_times) < current_f[int(current_f[0,2]),1]):
                #if signal is above mean by std_times * std, change 0 to 1 in temp_zero
                temp_zero[idx] = True
        
        #add array with force above snr, frequency and amount of data to output list
        output.append(np.vstack((temp_zero, f, np.repeat(number, len(f)))).T)

    #stack output lists            
    output = np.vstack(output)
    
    #add output to data
    data["correct"] = output
   
    

#%%analysis

def analyse_ap(data, chosen_passive = None, skip_last = 3):
    """
    Calculates fit parameters of fractional Kelvin-Voigt model to shear moduli and 
    power law model parameters of the fit to the active energy.
    
    Parameters
    ----------
    data : dict
        Data with pre-analysed data from active microrheology and particle fluctuation analysis.
    chosen_passive : array, optional
        Array with specifications of passive data to remove from analysis. The default is None.
    skip_last : int, optional
        Cutoff for highest frequencies for the fit to the shear moduli. The default is 3.

    Returns
    -------
    Tuple with 2 arrays. First array contains sit parameters to shear moduli, second array fit parameters to active energy.
        array 0: c_alpha, c_beta, alpha, beta, R²
        array 1: e_0, nu, e_0 (shifted), nu (shifted), R², R² (shifted)
        
    """
    
    ###only fits parameters in y-direction
    
    #create empty lists
    passive_array = []
    spring = []
    active_E = []
    
    
    #apply filter for signal-to-noise ratio
    if not "correct" in data:
        filter_bad(data)

    
    #go through all measurements
    for idx, ID in enumerate(np.unique(data["active"][:,-2])):  
        
        #get frequencies and number of frequencies
        frequencies = np.real(np.unique(np.real(data["passive"][data["passive"][:,-2] == ID,4])))
        l = len(frequencies)
        
        #get current passive measurement
        current_passive_measure = data["passive"][data["passive"][:,-2] == ID,:] 
                
        #get slopes for conversion into length scale
        slope_x = ((current_passive_measure[:,7]*10**6)**2)[0]
        slope_y = ((current_passive_measure[:,8]*10**6)**2)[0]

        #calculate mean if there are multiple passive measurements and convert to length scale with slope
        if chosen_passive is not None:
            used_passive = ~np.isin(current_passive_measure[:,3], chosen_passive[chosen_passive[:,5] == ID, 0:5] - 1)    
            passive_mean_x = np.nanmean(np.reshape(np.real(current_passive_measure[:,5][used_passive])/slope_x,[int(len(current_passive_measure[used_passive])/l),l]),0)
            passive_mean_y = np.nanmean(np.reshape(np.real(current_passive_measure[:,6][used_passive])/slope_y,[int(len(current_passive_measure[used_passive])/l),l]),0)
            
        else:
            passive_mean_x = np.nanmean(np.reshape(np.real(current_passive_measure[:,5])/slope_x,[int(len(current_passive_measure)/l),l]),0)
            passive_mean_y = np.nanmean(np.reshape(np.real(current_passive_measure[:,6])/slope_y,[int(len(current_passive_measure)/l),l]),0)
            
        #append to list and calculate response; predicted_response = passive*pi*f/kBT
        passive_array.append(np.transpose(np.vstack((passive_mean_x,passive_mean_y))*np.pi*frequencies/(scipy.constants.k *310)))
        
        #get active microrheology data
        active_set = data["active"][data["active"][:,-2] == ID,:]
        #only consider values that reached signal-to-noise ratio
        corr_no_bad = data["correct"][data["correct"][:,2] == ID][:-skip_last, 0].astype(bool)[::-1]
        #fit fractional Kelvin-Voigt model to data
        spring.append(springpot_fit(active_set, corr_no_bad, skip_last))
        
        #calculate active energy and fit power law model 
        active_E.append(active_E_fit(active_set[::-1], passive_array[-1], spring[-1]))

        
    return np.vstack((spring)), np.vstack((active_E))
  

def springpot_fit(active_set, corr_no_bad, skip_last):
    """
    Fit fractional Kelvin-Voigt model to shear moduli G' and G'' at frequencies f. 

    Parameters
    ----------
    active_set : array
        Data of active microrheology..
    corr_no_bad : array, bool
        Array with measurements to be filtered.
    skip_last : TYPE
        Frequencies to cut off for fit (plateau reached at about 1000 Hz).

    Returns
    -------
    Array with fit parameters c_alpha, c_beta, alpha, beta, R²

    """
    
    #get ID
    ID = active_set[0,-2]

    ##save frequency and shear modulus in x and y, one single array for both G' and G''. Skip all frequencies that did not exceed snr-threshold
    correct_len = np.sum(corr_no_bad)
    x = np.real(np.hstack([active_set[:,4][skip_last:][corr_no_bad], active_set[:,4][skip_last:][corr_no_bad][np.abs(np.imag(active_set[:,8][skip_last:][corr_no_bad])) > 10**-10]]))
    y = np.hstack([np.log(np.abs(np.real(active_set[:,8][skip_last:][corr_no_bad]))), np.log(np.abs(np.imag(active_set[:,8][skip_last:][corr_no_bad][np.abs(np.imag(active_set[:,8][skip_last:][corr_no_bad])) > 10**-10])))])
  
    try:
        #fit of both moduli at the same time, yielding 4 parameters
        tmp_fit_no_bad, pcov = curve_fit(lambda f, A, B, alpha, beta: springpot(f, A, B, alpha, beta, correct_len), x, y, p0 = [11, 1, 0.3, 0.9], bounds = ([0, 0, -np.inf, -np.inf], np.inf), maxfev = 50000)
       
        #calculate R² value
        r_pot = gof(x, y, np.real(np.hstack((tmp_fit_no_bad, len(active_set[:,4][skip_last:])))), springpot)
        
        #switch alpha and beta if beta is smaller than alpha
        if tmp_fit_no_bad[2] > tmp_fit_no_bad[3]:
            tmp_fit_no_bad = [tmp_fit_no_bad[1], tmp_fit_no_bad[0], tmp_fit_no_bad[3], tmp_fit_no_bad[2]]
    except:
        #if fit is not possible, fill with zeros
        print("no fit for springpot model for dataset number: " + str(int(ID)))
        tmp_fit_no_bad = [0, 0, 0, 0]
        r_pot = 0
        
    #return four fit parameters (c_alpha, c_beta, alpha, beta) and R²-value in array
    return np.hstack((tmp_fit_no_bad, r_pot))



def active_E_fit(active_set, passive_set, shear):
    """
    Fit power law to active energy.

    Parameters
    ----------
    active_set : array
        Data of active microrheology.
    passive_array : array
        Data of passive measurement, converted to length.
    shear : array
        Fit parameters from shear modulus.    

    Returns
    -------
    Array with fit parameters of power law fit to active energy e_0, nu, e_0 (shifted), nu (shifted), R², R² of shifted

    """
    
    #get frequencies and number of frequencies
    frequencies = active_set[:,4]
    l = len(frequencies)
    
    #get beadsize
    beadsize = np.real(active_set[:,-5])[0]
    
    #calculate shear moduli from fit parameters of fractional Kelvin-Voigt model
    g = np.reshape(np.exp(springpot(np.tile(frequencies,2), *shear[:4], 16)),[2,16])
    g_star = g[0] + 1J*g[1]
    #true divide error because of nans in real part of g_star
    
    #calculate response according to shear moduli calculated from fit parameters
    response_fit = np.abs(np.imag(1/(3 * np.pi * beadsize * g_star)))
    
    #calculate active energy
    E_a = passive_set[:,1]/response_fit
    
    #exclude nan values for fit
    na_fit = ~np.isnan(np.log(E_a))
    
    #exclude 46.5 and 100 Hz (stage noise)
    na_fit[8:10] = False
    na_fit[-1] = False
    
    #if measurement is taken during mitosis, remove 0.1 Hz 
    if np.isin(active_set[0,-1], [2,3,4,5,6]):
        na_fit[0] = False
        
    #calculate shift of active and passive measurements at high frequencies (1000, 2154, 4642 Hz)
    shift = np.tile(np.mean(np.divide(np.abs(np.imag(active_set[:,5:7])), passive_set)[12:15],0), (l,1))
    
    #calculate active energy when passive measurement is shifted on top of active
    E_a_shift = np.real(shift[:,1]*passive_set[:,1]/response_fit)
    
    #remove nan and stage noise frequencies
    na_shift = ~np.isnan(E_a_shift)
    na_shift[8:10] = False
    na_shift[-1] = False
    
    
    try:
        #fit power law to active energy and shifted active energy
        tmp_pl_fit = curve_fit(power_law_log, frequencies[na_fit], np.log(E_a)[na_fit], maxfev = 50000)[0]
        tmp_pl_shift = curve_fit(power_law_log, frequencies[na_shift], np.log(E_a_shift[na_shift]), maxfev = 50000)[0]
        
        #calculate R² values for fits
        r_pl_fit = gof(frequencies[na_fit], np.log(E_a)[na_fit], tmp_pl_fit, power_law_log)
        r_pl_shift = gof(frequencies[na_shift], np.log(E_a_shift[na_shift]), tmp_pl_shift, power_law_log)
    except:
        #if fit is not possible, fill with zeros
        tmp_pl_fit = [0,0]
        tmp_pl_shift =  [0,0]
        r_pl_fit = 0
        r_pl_shift =0 
       
    return np.real(np.hstack((tmp_pl_fit, tmp_pl_shift, r_pl_fit, r_pl_shift)))
    

#%% fit functions

#power law with offset 1 for active energy
def power_law_log(x,a,b):
    return np.log(a*x**b + 1)

#power law fit
def log_power_law(x,a,b):
    return np.log(a*x**b)

def gof(x,y,fit,func):
    "calculates goodness of fit as R²"
    ss_res = np.nansum((y-func(x,*fit))**2)
    ss_tot = np.nansum((y-np.nanmean(y))**2)
    r2_G12 = 1-(ss_res/ss_tot)
    return r2_G12

#elastic modulus as from frational Kelvin-Voigt model
def G1_pot(f, A, B, alpha, beta):
    func = (A* (2 * np.pi * f)**alpha) * np.cos(alpha * np.pi/2) + (B* (2 * np.pi * f)**beta) * np.cos(beta * np.pi/2)
    func[func <0] = np.nan
    return np.log(func)

#viscous modulus as from frational Kelvin-Voigt model
def G2_pot(f, A, B, alpha, beta):
    func = (A* (2 * np.pi * f)**alpha) * np.sin(alpha * np.pi/2) + (B* (2 * np.pi * f)**beta) * np.sin(beta * np.pi/2)
    func[func < 0] = np.nan
    return np.log(func)

#fit of both moduli in one function
def springpot(f, A, B, alpha, beta, length):
    "fits springpot model to G' and G''"

    extract1 = f[:int(length)]
    extract2 = f[int(length):]

    result1 = G1_pot(extract1, A, B, alpha, beta)
    result2 = G2_pot(extract2, A, B, alpha, beta) 
    return np.append(result1, result2)
