#!/usr/bin/env python3

# written by grey@christoforo.net

import os
import tempfile
import argparse
from glob import glob
import numpy as np
#import mpmath
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy import interpolate
from io import StringIO
from datetime import datetime
import csv
import array
import h5py
import time
import dataset

from lmfit.models import LorentzianModel, LinearModel, VoigtModel, GaussianModel

class Object(object):
    pass

parser = argparse.ArgumentParser(description='Peek at beam diagnostic data in hdf5 files')
parser.add_argument('--save-image', dest='saveImage', action='store_true', default=False, help="Save data .pgm images to /tmp/pgms/")
parser.add_argument('--save-report', dest='saveReport', action='store_true', default=False, help="Save analysis report .pdfs to /tmp/pdfs/")
parser.add_argument('--draw-plot', dest='drawPlot', action='store_true', default=False, help="Draw data plot or each file processed")
parser.add_argument('--h5-backend', type=argparse.FileType('ab'), help="Save/append analysis data to this HDF5 file")
parser.add_argument('--create-spreadsheet', type=argparse.FileType('x'), help="Create this spreadsheet using HDF5 backend file given in the --h5-backend argument")
parser.add_argument('--use-parameter-files', dest='pFiles', type=argparse.FileType('r'), nargs='*', help="Read additional timestamed parameters from these files")
parser.add_argument('input', type=argparse.FileType('r'), nargs='+', help="File(s) to process")
args = parser.parse_args()

if args.h5_backend is not None:
    filename,file_ext = os.path.splitext(args.h5_backend.name)
    if file_ext != '.h5':
        print("Error: HDF5 file name must end in .h5")
        exit(1)
    args.h5_backend.close()
    
    if os.path.getsize(args.h5_backend.name) == 0:
        newFile = False
    else:
        newFile = True

    be = h5py.File(args.h5_backend.name)
    #beRoot = be.get('/')
else:
    be = None
        
    #csvWriter = csv.DictWriter(args.csv_out,fieldnames=fieldNames)
    #csvWriter.writeheader()
    #args.csv_out.flush()
    
def spreadsheetWrite(key,value):
    if newFile:
        nCols = len(ws.columns)
        ws.append({nCols+1:key})
    
    ws.append({nCols+1:value})

def visitFunction(name, obj):
    root = obj.file.get('/')
    sampleName = root.attrs.get('sample_name')
    
    # TODO: remove this hack
    stageSample = root['data/linearstage/standa'].attrs.get('Current_sample')
    if sampleName != stageSample:
        sampleName = stageSample
        
    beSample = be.require_dataset('Sample Name')

    
    trigger = root.attrs.get('trigger_id')
    now = time.gmtime(root.attrs.get('timestamp'))
    titleString = str(trigger) + '|' + sampleName + '|' + time.strftime("%a, %d %b %Y %H:%M:%S", now)
    
    print(name)
    for key, val in obj.attrs.items():
        print('    ' + str(key) + ': ' + str(val))
    
    if type(obj) is h5py._hl.dataset.Dataset:
        print(obj.name+' <-- dataset')
        if 'Manta camera' in obj.parent.attrs.values():  # camera plot
            fig = plt.figure()
            camData = obj[:]
            ax = plt.matshow(camData,fignum=fig.number,origin='lower')
            ax.axes.xaxis.tick_bottom()
            plt.title('Camera|'+titleString)
            plt.colorbar()
            
            print("Camera Average:",np.average(camData),"[counts]")
            

        elif ('Thorlabs spectrometer' in obj.parent.attrs.values()) and ('spectra' in obj.name) and ('y_data' in obj.name):  # spectrometer plot
            parent = obj.parent
            xPlot = parent.get('x_values')[:]
            xlen = len(xPlot)
            yPlot = parent.get('y_data')[0:xlen]  # TODO doubcle chek this length
            y_scale = parent.get('y_scale')[0:xlen]
            #y = y/y_scale # TODO: check scaling
            
            # wavelength range overwhich we'll fit
            fitRange = (685, 705) # nm
            lMask = xPlot >= fitRange[0]
            uMask = xPlot <= fitRange[1]
            
            x = xPlot[lMask & uMask]
            y = yPlot[lMask & uMask]
            yMean = np.average(y)
            
            mod = LinearModel()
            lPars = mod.guess(y, x=x)
            
            mod = LorentzianModel(prefix='A_')
            A_zPars = mod.guess(y-yMean, x=x)
            A_zPars['A_center'].value = A_zPars['A_center'].value - 1
            
            mod = LorentzianModel(prefix='B_')
            B_zPars = mod.guess(y-yMean, x=x)
            B_zPars['B_center'].value = B_zPars['B_center'].value + 1
            
            pars = lPars + A_zPars + B_zPars
            
            mod = LinearModel() + LorentzianModel(prefix='A_') + LorentzianModel(prefix='B_')
                     
            result  = mod.fit(y, pars, x=x)
            #print(result.fit_report(min_correl=0.25))
            print("Peak A @",A_zPars['A_center'])
            
            plt.figure()
            plt.plot(xPlot,yPlot, marker='.',)
            plt.plot(x, result.best_fit, 'r-')
            
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Spectrometer Counts')
            plt.title('Emission Spectrum|'+titleString)
            plt.tight_layout()
            plt.grid()
            
        elif ('PicoScope 4264, python' in obj.parent.attrs.values()) and ('wavefront' in obj.name) and ('y_data' in obj.name):
            parent = obj.parent
            x = parent.get('x_data')[:]
            totalDuration = x[-1] - x[0]
            y = abs(parent.get('y_data')[:])
            plt.figure()
            currentAverage = np.average(y)
            plt.plot(x*1000, y*1e9, marker='.', label='Data')
            plt.plot((x[0]*1000,x[-1]*1000), (currentAverage*1e9,currentAverage*1e9), 'r--', label='{:.0f} ms Average = {:.0f} [nA]'.format(totalDuration*1e3,currentAverage*1e9))
            
            
            
            print("Current Average:",currentAverage*1e9,"[nA]")
            
            plt.title('Beam Current|' + titleString)
            
            plt.xlabel('Time Since Trigger Event [ms]')
            plt.ylabel('Beam Current [nA]')
            plt.grid()
            plt.legend()
            
            
            
            
            #print(obj)
        #if(len(obj[:].shape) == 2):
            #plt.matshow(obj[:])
            #plt.colorbar()
            #plt.show()
        #if(len(obj[:].shape) == 1):
            #dim = obj.len()
            #if(obj.attrs.get('pvname') and
               #obj.attrs['pvname'].find('CCS1') == 0):
                #dim = 3600
            #plt.plot(obj[0:dim])
            #plt.show()


# loop through each file in the input
for f in args.input:
    f.close()
    fullPath = f.name
    fileName = os.path.basename(f.name)
    print('Processing', fullPath, '...')
    f = h5py.File(fullPath, 'r')
    
    root = f.get('/')
    
    print('/')
    # top level attributes
    for key, val in root.attrs.items():
        print('    ' + str(key) + ': ' + str(val))
    #for key,val in root.items():
    #    obj = 
    #    print(key,val)
    f.visititems(lambda obj, name: visitFunction(obj, name))
    plt.show()
    print("")
    print("")
    newFile = False
    
if args.xlsx_out is not None:
    wb.save(args.xlsx_out.name)
    
    
