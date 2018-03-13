#!/usr/bin/env python

# written by grey@christoforo.net

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from pathlib import Path
home = str(Path.home())

db_name = 'FEB_18_OCL.db'
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
  sampleRows = sampleRows.loc[sampleRows['trigger_id'] > 6000] # ignore day one data
  x = np.array(sampleRows['timestamp'])
  y1 = np.array(sampleRows['temperature'])
  x = (x - x[0])/60
  y2 = np.array(sampleRows['camSpotAmplitude']) / np.array(sampleRows['camCharge'])
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
camEff = {}
plt.figure()
for sample in samples:
  sampleRow = doi.loc[doi['sample_name'] == sample]
  camCharge[sample] = np.array(sampleRow['camCharge'])
  camPeak[sample] = np.array(sampleRow['camSpotAmplitude'])
  camEff[sample] = camPeak[sample] / camCharge[sample]
  plt.plot(camCharge[sample],camPeak[sample],linestyle = 'None',marker='o',label=sample,markeredgewidth=0.0)

plt.xlabel('Charge Through Sample [nC]')
plt.ylabel('Photons (peak)')
plt.title('Camera Peak Photons Vs Charge'+'|'+session)
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
sEff = {}
plt.figure()
for sample in samples:
  sampleRow = doi.loc[doi['sample_name'] == sample]
  sCharge[sample] = np.array(sampleRow['spectroCharge'])
  sPeak[sample] = np.array(sampleRow['bHeight'].astype(float))
  camEff[sample] = sPeak[sample].astype(float) / sCharge[sample].astype(float)
  plt.plot(sCharge[sample],sPeak[sample],linestyle = 'None',marker='o',label=sample,markeredgewidth=0.0)

plt.xlabel('Charge Through Sample [nC]')
plt.ylabel('Spectrometer Ruby Peak Counts')
plt.title('Spectrometer Ruby Peak Vs Charge'+'|'+session)
plt.ylim((0, 60000))
plt.tight_layout()
plt.legend()
plt.grid()
  
#marker=mark,
#  markerfacecolor='None',
#  markeredgecolor=color,
#  linestyle = 'None',
 # label=`i`

plt.show()

#df.loc

# get first table
print("Done")
