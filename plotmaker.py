#!/usr/bin/env python

# written by grey@christoforo.net

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from lmfit.models import LinearModel
from lmfit import Model
from scipy import constants

from pathlib import Path
home = str(Path.home())

db_name = 'FEB_18_OCL_new.db'
session = "FEB '18 OCL"
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


# time on x axis temperature plots

# filter out non temperature data
doi = df.loc[~df['temperature'].isnull()]

# filter out beam off data
doi = doi.loc[doi['avgBeamCurrent'] > 20]

# filter out bad spot fits
doi = doi.loc[~doi['camSpotAmplitude'].isnull()]

# ignore day one data
doi = doi.loc[doi['trigger_id'] > 6000]

# filter out bad ruby peak fits
#doi = doi.loc[~doi['aHeight'].isnull()]
#doi = doi.loc[~doi['bHeight'].isnull()]
doi = doi.loc[doi['aHeight'].astype(float) > 0]
doi = doi.loc[doi['bHeight'].astype(float) > 0]

# unique samples
temperatureSamples = doi['sample_name'].unique()

PperC = {}
PerCTemp = {}
for sample in temperatureSamples:
  sampleRows = doi.loc[doi['sample_name'] == sample]
  forcedCamCharge = np.array(sampleRows['camCharge'])[:5].mean() # so that we don't get messed up by weird interactions between current measurement and heater  
  x = np.array(sampleRows['timestamp'])
  y1 = np.array(sampleRows['temperature'])
  x = (x - x[0])/60
  #y2 = np.array(sampleRows['camSpotAmplitude']) / np.array(sampleRows['camCharge'])
  camCharge = np.ones(len(y1)) * forcedCamCharge
  y2 = np.array(sampleRows['camSpotAmplitude']) / camCharge
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
  ax2.set_ylabel('Photons (peak) per nC', color='r')
  ax2.tick_params('y', colors='r')
  ax2.yaxis.grid(color='r')
  #ax2.grid()
  
  #plt.title(sample)
  fig.tight_layout()
  
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
plt.figure()
sample = 'HV1'
plt.plot(PerCTemp[sample],PperC[sample],label=sample)
sample = 'HVC1.2'
plt.plot(PerCTemp[sample],PperC[sample],label=sample)

plt.xlabel('Temperature [$^\circ$C]')
plt.ylabel('Photons (peak) per nC')
plt.title('Luminescence Efficiency Vs Temperature'+'|'+session)
plt.tight_layout()
plt.legend()
plt.grid()

# for peak vs charge seen plots

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
  

plt.xlabel('Current Through Sample [nA]')
plt.ylabel('Photons per Second (peak, form Gaus. fit)')
plt.title('Camera Photon Capture Rate Vs Current'+'|'+session)
plt.tight_layout()
plt.legend()
plt.grid()



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

plt.xlabel('Current Through Sample [nA]')
plt.ylabel('Spectrometer Ruby Peak Counts per Second')
plt.title('Spectrometer Ruby Peak Vs Current'+'|'+session)
plt.ylim((0, 100000))
plt.tight_layout()
plt.legend()
plt.grid()


# peak found by max after big blur

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

plt.xlabel('Current Through Sample [nA]')
plt.ylabel('Photons per Second (peak, max() after median filt.)')
plt.title('Camera Peak Photons Vs Current'+'|'+session)
plt.tight_layout()
plt.legend()
plt.grid()


# peak found by integration of big blur image

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

plt.xlabel('Current Through Sample [nC]')
plt.ylabel('Photons per Second(total seen by camera after median filt.)')
plt.title('Camera Total Photons Vs Current'+'|'+session)
plt.tight_layout()
plt.legend()
plt.grid()


# photon emission vs proton flux

# filter out temperature data
doi = df.loc[df['temperature'].isnull()]

# filter out beam off data
doi = doi.loc[doi['avgBeamCurrent']>20]

# filter day 1
doi = doi.loc[doi['trigger_id'] > 6000] # ignore day one data

samplePhotonsPerCamPhoton = 5326.5557712833215

samples = doi['sample_name'].unique()
sampleCupFraction = {}
camCharge = {}
camTime = {}
sampleGausVol = {}
usefulProtons = {}
plt.figure()
for sample in samples:
  sampleRow = doi.loc[doi['sample_name'] == sample]
  camCharge[sample] = np.array(sampleRow['camCharge'])
  sampleGausVol[sample] = np.array(sampleRow['sampleGausVol'])
  sampleCupFraction[sample] = np.array(sampleRow['sampleCupFraction'])
  camTime[sample] = np.array(sampleRow['t_camExposure'])
  usefulProtons[sample] = round(camCharge[sample]/constants.e * sampleCupFraction[sample])
  plt.plot(usefulProtons[sample]/camTime[sample],sampleGausVol[sample]/camTime[sample],linestyle = 'None',marker='o',label=sample,markeredgewidth=0.0)

plt.xlabel('Current Through Sample [nC]')
plt.ylabel('Photons per Second(total seen by camera after median filt.)')
plt.title('Camera Total Photons Vs Current'+'|'+session)
plt.tight_layout()
plt.legend()
plt.grid()



# PperP boxplot

# filter out temperature data
doi = df.loc[df['temperature'].isnull()]

# filter out beam off data
doi = doi.loc[doi['avgBeamCurrent']>20]

# filter bad cam fits
doi = doi.loc[doi['camSpotAmplitude'] > 0]

# filter day 1
doi = doi.loc[doi['trigger_id'] > 6000] # ignore day one data

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
ax2.plot(x, y2A, 'r-',label='$^\sigma$A')
ax2.plot(x, y2B, 'r:',label='$^\sigma$B')
ax2.set_ylabel('Spot standard deviations [pixels]', color='r')
ax2.tick_params('y', colors='r')
ax2.yaxis.grid(color='r')
ax2.set_ylim((100, 200))
ax2.legend()
#ax2.grid()

#plt.title(sample)
fig.tight_layout()

#samples = doi['sample_name'].unique()
##photonVol = {}
##camCharge = {}
##camEff = {}
#trigger = {}
#pPerP = {}
#plt.figure()
#for sample in samples:
  #sampleRow = doi.loc[doi['sample_name'] == sample]
  #trigger[sample] = np.array(sampleRow['trigger_id'])
  #pPerP[sample] = np.array(sampleRow['photonsPerProtonGaus'])
  ##camEff[sample] = camPeak[sample] / camCharge[sample]
  #plt.plot(trigger[sample],pPerP[sample],linestyle = 'None',marker='o',label=sample,markeredgewidth=0.0)

#plt.xlabel('Trigger Number')
#plt.ylabel('Photons per Proton (from Gaus. fit integration)')
#plt.title('Photons per Proton'+'|'+session)
#plt.tight_layout()
#plt.legend()
#plt.grid()


  
#marker=mark,
#  markerfacecolor='None',
#  markeredgecolor=color,
#  linestyle = 'None',
 # label=`i`

plt.show()

#df.loc

# get first table
print("Done")
