#!/usr/bin/env python
# written by grey@christoforo.net

import argparse
import os
from toolkit import analyzer

parser = argparse.ArgumentParser(description='Peek at beam diagnostic data in hdf5 files')
parser.add_argument('--draw-plots', dest='drawPlots', action='store_true', default=False, help="Draw analysis plots or each file processed")
parser.add_argument('--database', default=':memory:', help="Save/append analysis data to this sqlite database file")
parser.add_argument('--freeze-file', dest='freezeObj', type=argparse.FileType('x'), help="Freeze/draw data to a .csv file")
parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Dump everything to the terminal during analysis")
parser.add_argument('input', type=argparse.FileType('r'), nargs='+', help="File(s) to process")
pArgs = parser.parse_args()

if pArgs.freezeObj is not None:
  filename,file_ext = os.path.splitext(pArgs.freezeObj.name)
  if file_ext != '.csv':
    print("Error: Freeze file name must end in .csv")
    exit(1)
    
if pArgs.database != ':memory:':
  filename,file_ext = os.path.splitext(pArgs.database)
  if file_ext != '.db':
    print("Error: Database file name must end in .db")
    exit(1)

#========= start print override stuff =========
import builtins as __builtin__
systemPrint = __builtin__.print
def print(*args, **kwargs): # overload the print() function
  if pArgs.verbose:
    return systemPrint(*args, **kwargs) # now do the print for real
  else:
    return  # or not
__builtin__.print = print
#========= end print override stuff =========

a = analyzer(files = pArgs.input, database=pArgs.database, drawPlots=pArgs.drawPlots, freezeObj = pArgs.freezeObj)
a.processFiles()