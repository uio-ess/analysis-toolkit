#!/usr/bin/env python
# written by grey@christoforo.net

# note: to increase the open file limit under linux use: ulimit -n 2048
# note: select files by file number (in bash): ./cli.py --database ./db.db $(find $(seq -f "./%016g-*.h5" 1843 1845) 2> /dev/null)

import argparse
import os
from toolkit import analyzer

parser = argparse.ArgumentParser(description='Peek at beam diagnostic data in hdf5 files')
parser.add_argument('--draw-plots', dest='drawPlots', action='store_true', default=False, help="Draw analysis plots or each file processed")
parser.add_argument('--filter-current', dest='do_sw_current_filter', action='store_true', default=False, help="Do software filter on current waveform")
parser.add_argument('--cam-pv', dest='cam_pv', type=str, default="", help="Camera image PV base for real time fitting mode (maybe CAM1:)")
parser.add_argument('--no-spot-fit', dest='fitSpot', action='store_false', default=True, help="Do not fit camera data to 2D gaussian")
parser.add_argument('--database', default=':memory:', help="Save/append analysis data to this sqlite database file")
parser.add_argument('--freeze-file', dest='freezeObj', type=argparse.FileType('x'), help="Freeze/draw data to a .csv file")
parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Dump everything to the terminal during analysis")
parser.add_argument('input', type=argparse.FileType('r'), nargs='*', help="File(s) to process")
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

a = analyzer(files = pArgs.input, verbose=pArgs.verbose, database=pArgs.database, drawPlots=pArgs.drawPlots, freezeObj = pArgs.freezeObj, fitSpot = pArgs.fitSpot, do_sw_current_filter = pArgs.do_sw_current_filter, cam_pv=pArgs.cam_pv)

if pArgs.cam_pv == "":
  a.processFiles()
else:
  print("Realtime camera fit mode activated")
  a.realtimeFit()