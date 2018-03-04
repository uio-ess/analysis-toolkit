import h5py
import dataset
import datafreeze
import os
import time
import matplotlib.pylab as plt
import numpy as np
from lmfit.models import LorentzianModel, LinearModel, VoigtModel, GaussianModel


class analyzer:
  """
  Analyzes HDF5 sample files and enters sample data into the sample database
  for the ESS proton beam imaging system
  """

  def __init__(self, files, database = ':memory:', drawPlots = False, freezeObj = None):
    self.files = files
    self.drawPlots = drawPlots
    self.db = dataset.connect('sqlite:///'+database)
    self.t = None  # the database table we'll be using here
    self.freezeObj = freezeObj
    
    self.sd = {}  # sample dictionary
    
  def processFiles(self):
    # loop through each file in the input
    for f in self.files:
      f.close()
      fullPath = f.name
      fileName = os.path.basename(f.name)
      print('Processing', fullPath, '...')
      f = h5py.File(fullPath, 'r')
    
      root = f.get('/')
      print('/')
      
      session = "FEB '18 OCL"
      #session = root.attrs.get('session')  # TODO: insert session attr into sample HDF5 files
      self.sd['session'] = session
      self.t = self.db[str(hash(session))]
      # self.t.drop()
      
      # print top level attributes
      for key, val in root.attrs.items():
        print('\t' + str(key) + ': ' + str(val))
        
      attr = 'sample_name'
      self.sd[attr] = root.attrs.get(attr)
      stageSample = root['data/linearstage/standa'].attrs.get('Current_sample')  # TODO: remove this hack
      if stageSample != self.sd[attr]:
        self.sd[attr] = stageSample      
      
      attr = 'timestamp'
      self.sd[attr] = root.attrs.get(attr)
      
      attr = 'trigger_id'
      self.sd[attr] = root.attrs.get(attr)
       
      f.visititems(self.visitor)
      
      self.t.insert(self.sd, ensure=True)
      
      print("")
      print("")
      self.drawPlots and plt.show()
    print("Done processing all files.")
    
    if self.freezeObj is not None:
      result = self.t.all()
      datafreeze.freeze(result, format='csv', fileobj=self.freezeObj)
      print('Sooo cold! Data frozen to {:}'.format(self.freezeObj.name))
      self.freezeObj.close()
  
  def visitor(self, name, obj):
    root = obj.file.get('/')

    titleString = str(self.sd['trigger_id']) + '|' + self.sd['sample_name'] + '|' + self.sd['session']
    
    print(name)
    for key, val in obj.attrs.items():
        print('    ' + str(key) + ': ' + str(val))
    
    if type(obj) is h5py._hl.dataset.Dataset:
      print(obj.name+' <-- dataset')
      if 'Manta camera' in obj.parent.attrs.values():  # camera plot
        # TODO: break these handlers out into their own functions
        camData = obj[:]
        camAvg = np.average(camData)
        print("Camera Average:",camAvg,"[counts]")
        self.sd['camAvg'] = camAvg        
        
        if self.drawPlots:
          fig = plt.figure()
          ax = plt.matshow(camData,fignum=fig.number,origin='lower')
          ax.axes.xaxis.tick_bottom()
          plt.title('Camera|'+titleString)
          plt.colorbar()

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
        aAmp = result.params['A_amplitude'].value
        aCen = result.params['A_center'].value
        bAmp = result.params['B_amplitude'].value
        bCen = result.params['B_center'].value
        print("Peak A: {:.0f} counts @ {:.2f} [nm]".format(aAmp, aCen))
        print("Peak B: {:.0f} counts @ {:.2f} [nm]".format(bAmp, bCen))
        self.sd['aAmp'] = aAmp
        self.sd['aCen'] = aCen
        self.sd['bAmp'] = bAmp
        self.sd['bCen'] = bCen
        
        if self.drawPlots:
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
        print("Current Average:",currentAverage*1e9,"[nA]")
        self.sd['avgBeamCurrent'] = currentAverage
        
        if self.drawPlots:
          plt.plot(x*1000, y*1e9, marker='.', label='Data')
          plt.plot((x[0]*1000,x[-1]*1000), (currentAverage*1e9,currentAverage*1e9), 'r--', label='{:.0f} ms Average = {:.0f} [nA]'.format(totalDuration*1e3,currentAverage*1e9))
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
    
    
    