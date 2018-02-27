#!/usr/bin/env python3

# written by grey@christoforo.net

import os
import tempfile
import argparse
from glob import glob
import numpy as np
#import mpmath
import matplotlib.pyplot as plt
from sdds import SDDS as ssds
#from math import sqrt
from scipy import optimize as opt
from scipy import interpolate
from io import StringIO
from datetime import datetime
import csv
import array
import h5py

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
    print(name)
    for key, val in obj.attrs.items():
        print('    ' + str(key) + ': ' + str(val))
    
    if type(obj) is h5py._hl.dataset.Dataset:
        print(obj)
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
    f.visititems(lambda obj, name: visitFunction(obj, name))
    print("")
    print("")