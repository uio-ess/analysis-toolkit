# analysis-toolkit
a tool for peeking at beam diagnostic data in a hdf5 file

## CLI interface usage
```
$ ./cli.py --help
usage: cli.py [-h] [--draw-plots] [--no-spot-fit] [--database DATABASE]
              [--freeze-file FREEZEOBJ] [-v]
              [input [input ...]]

Peek at beam diagnostic data in hdf5 files

positional arguments:
  input                 File(s) to process

optional arguments:
  -h, --help            show this help message and exit
  --draw-plots          Draw analysis plots or each file processed
  --no-spot-fit         Do not fit camera data to 2D gaussian
  --database DATABASE   Save/append analysis data to this sqlite database file
  --freeze-file FREEZEOBJ
                        Freeze/draw data to a .csv file
  -v, --verbose         Dump everything to the terminal during analysis
```
