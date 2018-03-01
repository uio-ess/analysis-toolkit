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

from lmfit.models import LorentzianModel, LinearModel, VoigtModel, GaussianModel

class Object(object):
    pass

parser = argparse.ArgumentParser(description='Peek at beam diagnostic data in hdf5 files')
parser.add_argument('--save-image', dest='saveImage', action='store_true', default=False, help="Save data .pgm images to /tmp/pgms/")
parser.add_argument('--save-report', dest='saveReport', action='store_true', default=False, help="Save analysis report .pdfs to /tmp/pdfs/")
parser.add_argument('--draw-plot', dest='drawPlot', action='store_true', default=False, help="Draw data plot or each file processed")
parser.add_argument('--csv-out', type=argparse.FileType('w'), help="Save analysis data to csv file")
parser.add_argument("--do-not-fit", dest='dontFit', action='store_true', default=False, help="Do not attempt to fit the data to a 2D gaussian")
parser.add_argument('--use-parameter-files', dest='pFiles', type=argparse.FileType('r'), nargs='*', help="Read additional timestamed parameters from these files")
parser.add_argument('input', type=argparse.FileType('rb'), nargs='+', help="File(s) to process")
args = parser.parse_args()

def visitFunction(name, obj):
    root = obj.file.get('/')
    sampleName = root.attrs.get('sample_name')
    
    # TODO: remove this hack
    stageSample = root['data/linearstage/standa'].attrs.get('Current_sample')
    if sampleName != stageSample:
        sampleName = stageSample

    
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
            # TODO: fit to dual lorentzian
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
            plt.title('Emission Spectra|'+titleString)
            plt.tight_layout()
            plt.grid()
            
        elif ('PicoScope 4264, python' in obj.parent.attrs.values()) and ('wavefront' in obj.name) and ('y_data' in obj.name):
            parent = obj.parent
            x = parent.get('x_data')[:]
            y = parent.get('y_data')[:]
            plt.figure()
            currentAverage = np.average(y)
            plt.plot(x*1000, y*1e9, marker='.', label='Data')
            plt.plot((x[0]*1000,x[-1]*1000), (currentAverage*1e9,currentAverage*1e9), 'r--', label='Average = {:.0f} [nA]'.format(currentAverage*1e9))
            
            
            
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