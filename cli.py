#!/usr/bin/env python
# written by grey@christoforo.net

import argparse
import os
from toolkit import analyzer

parser = argparse.ArgumentParser(description='Peek at beam diagnostic data in hdf5 files')
parser.add_argument('--draw-plots', dest='drawPlots', action='store_true', default=False, help="Draw analysis plots or each file processed")
parser.add_argument('--database', default=':memory:', help="Save/append analysis data to this sqlite database file")
parser.add_argument('--freeze-file', dest='freezeObj', type=argparse.FileType('x'), help="Freeze/draw data to a .csv file")
parser.add_argument('input', type=argparse.FileType('r'), nargs='+', help="File(s) to process")
args = parser.parse_args()

if args.freezeObj is not None:
  filename,file_ext = os.path.splitext(args.freezeObj.name)
  if file_ext != '.csv':
    print("Error: Freeze file name must end in .csv")
    exit(1)
    
if args.database != ':memory:':
  filename,file_ext = os.path.splitext(args.database)
  if file_ext != '.db':
    print("Error: Database file name must end in .db")
    exit(1)
    
a = analyzer(files = args.input, database=args.database, drawPlots=args.drawPlots, freezeObj = args.freezeObj)
a.processFiles()

