#!/usr/bin/env python

# written by grey@christoforo.net

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LinearModel
from lmfit import Model
from scipy import constants

from pathlib import Path
home = str(Path.home())

db_name = 'JUN_18_OCL.db'
session = "JUN '18 OCL" # TODO: read this from the files
fullpath = home + '/' + db_name 


conn = sqlite3.connect('file:' + fullpath + '?mode=ro', uri=True)
res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

tables = []
for name in res:
  tables.append(name[0])

if len(tables) == 1:
  tableName = tables[0]
else:
  print('The tables available in the database are:')
  print(tables)
  tableName = input("Type one here --> ")
  
queryString = "select * from {:};".format(tableName)

df = pd.read_sql_query(queryString, conn)

# TODO: read from the files
camPhotonsPerCount = 5.7817


# time on x axis temperature plots
enable_this_section = True
if enable_this_section:
  
  # filter out non temperature data
  doi = df.loc[~df['temperature'].isnull()]
  
  # filter out beam off data
  doi = doi.loc[doi['avgBeamCurrent'] < 20]
  
  # filter out bad spot fits
  # filter bad cam fits
  doi = doi.loc[doi['camSpotAmplitude'] > 0]
  
  # ignore day one data
  doi = doi.loc[doi['trigger_id'] >= 1843]
  doi = doi.loc[doi['trigger_id'] <= 2275]
  
  # filter out bad ruby peak fits
  #doi = doi.loc[~doi['aHeight'].isnull()]
  #doi = doi.loc[~doi['bHeight'].isnull()]
  #doi = doi.loc[doi['aHeight'].astype(float) > 0]
  #doi = doi.loc[doi['bHeight'].astype(float) > 0]
  
  # unique samples
  temperatureSamples = doi['sample_name'].unique()
  
  PperC = {}
  PerCTemp = {}
  for sample in temperatureSamples:
    sampleRows = doi.loc[doi['sample_name'] == sample]
    forcedCamCharge = np.array(sampleRows['camCharge'])[:5].mean() # so that we don't get messed up by weird interactions between current measurement and heater  
    x = np.array(sampleRows['timestamp'].astype(float))
    y1 = np.array(sampleRows['temperature'])
    x = (x - x[0])/60
    y2 = np.array(sampleRows['photonsPerProtonBlur'])
    camCharge = np.ones(len(y1)) * forcedCamCharge
    #y2 = np.array(sampleRows['camSpotAmplitude']) / camCharge
    PperC[sample] = y2
    PerCTemp[sample] = y1
    
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'b-')
    ax1.set_xlabel('Time [min]')
    ax1.xaxis.grid()
    ax1.set_title('Luminescence Efficiency and Temperature Vs Time'+'|'+sample+'|'+session)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Sample Temperature [$^\circ$C]', color='b')
    #ax1.yaxis.grid(color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r-')
    ax2.set_ylabel('Photons per Proton', color='r')
    ax2.tick_params('y', colors='r')
    ax2.yaxis.grid(color='r')
    #ax2.grid()
    
    #plt.title(sample)
    fig.tight_layout()
    
    # temp vs spectro stuff
    enable_this_section = False
    if enable_this_section:
      y2 = np.array(sampleRows['bCen'].astype(float)) - np.array(sampleRows['aCen'].astype(float))
      fig, ax1 = plt.subplots()
      ax1.plot(x, y1, 'b-')
      ax1.set_xlabel('Time [min]')
      ax1.xaxis.grid()
      ax1.set_title('Ruby Peak Spacing Vs Temperature'+'|'+sample+'|'+session)
      # Make the y-axis label, ticks and tick labels match the line color.
      ax1.set_ylabel('Sample Temperature [$^\circ$C]', color='b')
      #ax1.yaxis.grid(color='b')
      ax1.tick_params('y', colors='b')
      
      ax2 = ax1.twinx()
      ax2.plot(x, y2, 'r-')
      ax2.set_ylabel('Ruby emission peak spacing [nm]', color='r')
      ax2.tick_params('y', colors='r')
      ax2.yaxis.grid(color='r')
      #ax2.grid()
      
      #plt.title(sample)
      fig.tight_layout()  

  # temp on x axis temp plots
  for sample in ['HV1', 'HVC1.2','HV10']:
    plt.figure()
    plt.plot(PerCTemp[sample],PperC[sample],label=sample)
    
    plt.xlabel('Temperature [$^\circ$C]')
    plt.ylabel('Photons per Proton')
    plt.title('Luminescence Efficiency Vs Temperature'+'|'+sample+'|'+session)
    plt.tight_layout()
    #plt.legend()
    plt.grid()

# for peak vs charge seen plots
enable_this_section = False
if enable_this_section:
  
  # filter out temperature data
  doi = df.loc[df['temperature'].isnull()]
  
  # filter out beam off data
  doi = doi.loc[doi['avgBeamCurrent']>20]
  
  # filter bad cam fits
  doi = doi.loc[doi['camSpotAmplitude'] > 0]
  
  # filter day 1
  doi = doi.loc[doi['trigger_id'] > 6000] # ignore day one data
  
  samples = doi['sample_name'].unique()
  camPeak = {}
  camCharge = {}
  camTime = {}
  plt.figure()
  for sample in samples:
    sampleRow = doi.loc[doi['sample_name'] == sample]
    camCharge[sample] = np.array(sampleRow['camCharge'])
    camPeak[sample] = np.array(sampleRow['camSpotAmplitude'])
    camTime[sample] = np.array(sampleRow['t_camExposure'])
    plt.plot(camCharge[sample]/camTime[sample],camPeak[sample]/camTime[sample],linestyle = 'None',marker='o',label=sample,markeredgewidth=0.0)
    
  
  plt.xlabel('Faraday Cup Current [nA]')
  plt.ylabel('Photons per Second (peak, form Gaus. fit)')
  plt.title('Camera Photon Capture Rate Vs Current'+'|'+session)
  plt.tight_layout()
  plt.legend()
  plt.grid()


enable_this_section = False
if enable_this_section:
  # filter out temperature data
  doi = df.loc[df['temperature'].isnull()]
  
  # filter out beam off data
  doi = doi.loc[doi['avgBeamCurrent']>20]
  
  # filter out bad ruby peak fits
  doi = doi.loc[doi['aHeight'].astype(float) > 0]
  doi = doi.loc[doi['bHeight'].astype(float) > 0]
  
  # filter day 1
  doi = doi.loc[doi['trigger_id'] > 6000] # ignore day one data
  
  samples = doi['sample_name'].unique()
  
  sPeak = {}
  sCharge = {}
  sTime = {}
  plt.figure()
  for sample in samples:
    sampleRow = doi.loc[doi['sample_name'] == sample]
    sCharge[sample] = np.array(sampleRow['spectroCharge'])
    sPeak[sample] = np.array(sampleRow['bHeight'].astype(float))
    sTime[sample] = np.array(sampleRow['t_spectrumExposure'])
    plt.plot(sCharge[sample]/sTime[sample],sPeak[sample]/sTime[sample],linestyle = 'None',marker='o',label=sample,markeredgewidth=0.0)
  
  plt.xlabel('Faraday Cup Current [nA]')
  plt.ylabel('Spectrometer Ruby Peak Counts per Second')
  plt.title('Spectrometer Ruby Peak Vs Current'+'|'+session)
  plt.ylim((0, 100000))
  plt.tight_layout()
  plt.legend()
  plt.grid()


# peak found by max after big blur
enable_this_section = False
if enable_this_section:
  
  # filter out temperature data
  doi = df.loc[df['temperature'].isnull()]
  
  # filter out beam off data
  doi = doi.loc[doi['avgBeamCurrent']>20]
  
  # filter bad cam fits
  doi = doi.loc[doi['camSpotAmplitude'] > 0]
  
  # filter day 1
  doi = doi.loc[doi['trigger_id'] > 6000] # ignore day one data
  
  samples = doi['sample_name'].unique()
  camPeak = {}
  camCharge = {}
  camTime = {}
  plt.figure()
  for sample in samples:
    sampleRow = doi.loc[doi['sample_name'] == sample]
    camCharge[sample] = np.array(sampleRow['camCharge'])
    camPeak[sample] = np.array(sampleRow['sampleBlurPeak'])
    camTime[sample] = np.array(sampleRow['t_camExposure'])
    plt.plot(camCharge[sample]/camTime[sample],camPeak[sample]/camTime[sample],linestyle = 'None',marker='o',label=sample,markeredgewidth=0.0)
  
  plt.xlabel('Faraday Cup Current [nA]')
  plt.ylabel('Photons per Second (peak, max() after median filt.)')
  plt.title('Camera Peak Photons Vs Current'+'|'+session)
  plt.tight_layout()
  plt.legend()
  plt.grid()


# peak found by integration of big blur image
enable_this_section = False
if enable_this_section:
  
  # filter out temperature data
  doi = df.loc[df['temperature'].isnull()]
  
  # filter out beam off data
  doi = doi.loc[doi['avgBeamCurrent']>20]
  
  # filter bad cam fits
  doi = doi.loc[doi['camSpotAmplitude'] > 0]
  
  # filter day 1
  doi = doi.loc[doi['trigger_id'] > 6000] # ignore day one data
  
  samples = doi['sample_name'].unique()
  photonVol = {}
  camCharge = {}
  camTime = {}
  plt.figure()
  for sample in samples:
    sampleRow = doi.loc[doi['sample_name'] == sample]
    camCharge[sample] = np.array(sampleRow['camCharge'])
    photonVol[sample] = np.array(sampleRow['sampleBlurVol'])
    camTime[sample] = np.array(sampleRow['t_camExposure'])
    plt.plot(camCharge[sample]/camTime[sample],photonVol[sample]/camTime[sample],linestyle = 'None',marker='o',label=sample,markeredgewidth=0.0)
  
  plt.xlabel('Faraday Cup Current [nA]')
  plt.ylabel('Photons per Second (total seen by camera after median filt.)')
  plt.title('Camera Total Photons Vs Current'+'|'+session)
  plt.tight_layout()
  plt.legend()
  plt.grid()


# photon emission vs proton flux
enable_this_section = True
if enable_this_section:
  # filter out beam off data
  doi = df.loc[df['avgBeamCurrent']>20]
  
  # filter bad cam fits
  doi = doi.loc[doi['camSpotAmplitude'] > 0]
  
  # filter setup runs
  doi = doi.loc[doi['trigger_id'] >= 1143] # ignore setup data
  doi = doi.loc[doi['trigger_id'] <= 1343] # ignore setup data
  
  # TODO: read these from the files
  samplePhotonsPerCamPhoton = 5326.5557712833215
  mod = LinearModel()
  samples = doi['sample_name'].unique()
  sampleCupFraction = {}
  camCharge = {}
  camTime = {}
  sampleGausVol = {}
  usefulProtons = {}
  pa = {}
  pb = {}
  slopes = {}
  colors = {}
  
  fig, ax = plt.subplots()
  for sample in samples:
    sampleRow = doi.loc[doi['sample_name'] == sample]
    camCharge[sample] = np.array(sampleRow['camCharge']) * 1e-9
    sampleGausVol[sample] = np.array(sampleRow['sampleGausVol'])
    sampleCupFraction[sample] = np.array(sampleRow['sampleCupFraction'])
    camTime[sample] = np.array(sampleRow['t_camExposure'])
    usefulProtons[sample] = (camCharge[sample]/constants.e * sampleCupFraction[sample]).round()
    x = usefulProtons[sample]/camTime[sample]
    y = sampleGausVol[sample]/camTime[sample]*samplePhotonsPerCamPhoton
    p = ax.plot(x,y,linestyle = 'None',marker='o',alpha=0.2)
    color = p[0].get_color()
    colors[sample] = color
    guesses = mod.guess(y, x=x)
    fitResult  = mod.fit(y, guesses, x=x)
    slope = fitResult.best_values['slope']
    slopes[sample] = slope
    y2 = fitResult.best_fit
    ax.plot(x,y2,color,label=sample)
    
    x_mid = (x.max() + x.min())/2
    y_mid = mod.eval(x=x_mid,**fitResult.best_values)
    xp2 = x.max()
    yp2 = mod.eval(x=xp2,**fitResult.best_values)
  
    
    pa[sample] = (x_mid,y_mid)
    pb[sample] = (xp2,yp2)
    
  ax.set_xlabel('Proton Flux Through Sample [proton/s]')
  ax.set_ylabel('Photon Emission From Sample [photon/s]')
  ax.set_title('Photons per Proton'+'|'+session)
  ax.legend()
  ax.grid()
  fig.tight_layout()
  
  # annotate the plot with text on the lines now
  for sample in samples:
    pat = ax.transData.transform(pa[sample])
    pbt = ax.transData.transform(pb[sample])
    tslope = (pat[1] - pbt[1])/(pat[0] - pbt[0])
    trans_angle = np.rad2deg(np.arctan(tslope))
    ax.annotate("{:.0f}".format(slopes[sample]), xy=(pa[sample][0], pa[sample][1]),  xycoords='data',
                  xytext=(0, 0), textcoords='offset points',rotation=trans_angle,ha='center',va='bottom',color=colors[sample],weight='bold', rotation_mode='anchor')
    


# light scan boxplot
enable_this_section = True
if enable_this_section:
  # only fixed exposure light scans
  doi = df.loc[df['trigger_id'] >= 1481]
  doi = doi.loc[doi['trigger_id'] <= 1534]
  
  # only good peak fits
  doi = doi.loc[doi['aHeight'].astype(float) > 0]
  doi = doi.loc[doi['bHeight'].astype(float) > 0]
  
  doi = doi.loc[doi['sample_name'] != 'HV10'] # remove HV10 it's too small
  
  samples = doi['sample_name'].unique()
  trigger = {}
  pPerP = []
  fig, ax = plt.subplots()
  for sample in samples:
    sampleRow = doi.loc[doi['sample_name'] == sample]
    pPerP.append(np.array(sampleRow['bHeight'].astype(float)))
  
  ax.boxplot(pPerP, labels=samples, notch=False, showfliers=False, showmeans=True, meanline=True)
  ax.yaxis.grid()
  for i in range(len(samples)):
    y = pPerP[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    ax.plot(x, y, 'm.', alpha=0.1)
  #plt.xlabel('Trigger Number')
  ax.set_ylabel('Ruby Emission Peak [spectrometer counts]')
  ax.set_title(session+ ' | 375nm LED')
  fig.tight_layout() 


# PperP boxplot
enable_this_section = True
if enable_this_section:
  # filter out beam off data
  doi = df.loc[df['avgBeamCurrent']>20]
  
  # filter bad cam fits
  doi = doi.loc[doi['camSpotAmplitude'] > 0]
  
  # filter setup runs
  doi = doi.loc[doi['trigger_id'] >= 1143] # ignore setup data
  doi = doi.loc[doi['trigger_id'] <= 1343] # ignore setup data
  
  samples = doi['sample_name'].unique()
  trigger = {}
  pPerP = []
  fig, ax = plt.subplots()
  for sample in samples:
    sampleRow = doi.loc[doi['sample_name'] == sample]
    pPerP.append(np.array(sampleRow['photonsPerProtonGaus']))
  
  ax.boxplot(pPerP, labels=samples, notch=False, showfliers=False, showmeans=True, meanline=True)
  ax.yaxis.grid()
  for i in range(len(samples)):
    y = pPerP[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    ax.plot(x, y, 'm.', alpha=0.1)
  #plt.xlabel('Trigger Number')
  ax.set_ylabel('Photons per Proton')
  ax.set_title(session)
  fig.tight_layout()



# sigma stuff
enable_this_section = False
if enable_this_section:
  
  # filter out temperature data
  doi = df.loc[df['temperature'].isnull()]
  
  # filter out beam off data
  doi = doi.loc[doi['avgBeamCurrent']>20]
  
  # filter bad cam fits
  doi = doi.loc[doi['camSpotAmplitude'] > 0]
  
  # filter day 1
  doi = doi.loc[doi['trigger_id'] > 6000] # ignore day one data
  
  x = np.array(doi['trigger_id'])
  y1 = np.array(doi['avgBeamCurrent'])
  y2A = np.array(doi['sigmaA'])
  y2B = np.array(doi['sigmaB'])
  
  # pick out the larger and smaller sigmas
  y2H = np.maximum(y2A,y2B)
  y2L = np.minimum(y2A,y2B)
  
  
  fig, ax1 = plt.subplots()
  ax1.plot(x, y1, 'b',linestyle = 'None',marker='o')
  ax1.set_xlabel('Trigger Number')
  ax1.xaxis.grid()
  ax1.set_title('Spot Standard Deviation vs Beam Current'+'|'+session)
  # Make the y-axis label, ticks and tick labels match the line color.
  ax1.set_ylabel('Beam Current [nA]', color='b')
  #ax1.yaxis.grid(color='b')
  ax1.tick_params('y', colors='b')
  
  ax2 = ax1.twinx()
  ax2.plot(x, y2H, 'r-',label='$^\sigma$A')
  ax2.plot(x, y2L, 'r:',label='$^\sigma$B')
  ax2.set_ylabel('Spot standard deviations [pixels]', color='r')
  ax2.tick_params('y', colors='r')
  ax2.yaxis.grid(color='r')
  ax2.set_ylim((100, 200))
  ax2.legend()
  #ax2.grid()
  
  #plt.title(sample)
  fig.tight_layout()


plt.show()

print("Done")
