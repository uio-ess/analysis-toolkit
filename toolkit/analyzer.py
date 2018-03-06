import h5py
import dataset
import datafreeze
import os
import io
import re
import time
import hashlib
import matplotlib.pylab as plt
import numpy as np
from lmfit.models import LorentzianModel, LinearModel, VoigtModel, GaussianModel
from scipy import signal
from scipy import constants
from scipy import optimize as opt
from scipy import interpolate

class analyzer:
  """
  Analyzes HDF5 sample files and enters sample data into the sample database
  for the ESS proton beam imaging system
  """
  
  camPhotonsPerCount = 5.7817 # TODO: read this from the data when it's ready

  def __init__(self, files, database = ':memory:', drawPlots = False, freezeObj = None, fitSpot = True):
    self.files = files
    self.drawPlots = drawPlots
    self.fitSpot = fitSpot
    self.db = dataset.connect('sqlite:///'+database)
    self.t = None  # the database table we'll be using here
    self.freezeObj = freezeObj
    
    self.sd = {}  # sample dictionary
    
  def processFiles(self):
    # loop through each file in the input
    for f in self.files:
      self.sd = {}  # initialize sample dictionary
      try:
        f.close()
        fullPath = f.name
        fileName = os.path.basename(f.name)
        print('Processing', fullPath, '...')        
        self.processOneFile(f)
      except:
        print("Failed to process {:}".format(f.name))
      self.drawPlots and plt.show()
    print("Done processing all files.")
    
    # dump the results to a csv file if the user asked for it
    if self.freezeObj is not None:
      if self.files == []:  # we were called without processing new files
        if len(self.db.tables) == 1:
          tableName = self.db.tables[0]
        else:
          print('The tables available in the database are:')
          print(self.db.tables)
          tableName = input("Type one here --> ")
        self.t = self.db.create_table(tableName, primary_id='int_hash', primary_type=self.db.types.bigint)
      result = self.t.all()
      datafreeze.freeze(result, format='csv', fileobj=self.freezeObj)
      print('Sooo cold! Data frozen to {:}'.format(self.freezeObj.name))
      self.freezeObj.close()
      
  def processOneFile(self, f):
    f = h5py.File(f.name, 'r')
  
    root = f.get('/')
    print('/')
    
    session = "FEB '18 OCL"
    #session = root.attrs.get('session')  # TODO: insert session attr into sample HDF5 files
    self.sd['session'] = session
    safeSession = re.sub(r'\W+', '', session) # allow this to become a table name
    self.t = self.db.create_table(safeSession, primary_id='int_hash', primary_type=self.db.types.bigint)
    #self.t = self.db[str(hash(session))]
    # self.t.drop()
    
    # form row hash/title string out of sample name, trigger count and session
    attr = 'sample_name'
    self.sd[attr] = root.attrs.get(attr)
    try:
      stageSample = root['data/linearstage/standa'].attrs.get('Current_sample')  # TODO: remove this hack
    except:
      stageSample = None
    if (stageSample is not None) and (stageSample != self.sd[attr]):
      self.sd[attr] = stageSample
    
    attr = 'trigger_id'
    self.sd[attr] = int(root.attrs.get(attr)) # hopefully nothing gets mangled by the int cast here...
    
    self.titleString = str(self.sd['trigger_id']) + '|' +\
      self.sd['sample_name'] + '|' + self.sd['session']    
    
    # this is the hash we use for the uniqueness test when adding/updating the database
    self.sd['int_hash'] = int.from_bytes(hashlib.blake2s(self.titleString.encode(),digest_size=6).digest(),byteorder='big')
    
    # now immediately write the unfinished row to the database so we have something on the file in case we fail later
    self.t.upsert(self.sd, ['int_hash'], ensure=True)

    # other things i'm interested in here
    #iWants = ('sample_name','session', ...)
    iWants = ('experiment_description', 'sub_experiment', 'timestamp')
    for thingIWant in iWants:
      attribute = root.attrs.get(thingIWant)
      if type(attribute) is np.int64:
        attribute = int(attribute) # hopefully nothing is mangled here...
      self.sd[thingIWant] = attribute

    # print top level attributes
    for key, val in root.attrs.items():
      print('\t{:}--> {:}'.format(key,val))    
    
    # walk through the HDF5 file here, in no particular order...
    f.visititems(self.visitor)
    
    # now we'll do some analysis that had to wait until we'd read the entire file
    try:
      self.postAnalysis()
    except:
      print("Failed during postAnalysis()")
    
    # store what we've learned in our database
    self.t.upsert(self.sd, ['int_hash'], ensure=True)
    #self.t.insert(self.sd, ensure=True)
    
    print("")
    print("")    

  def postAnalysis(self):
    # compute the charge seen by the sample during data collection
    
    # integration range for camera
    intRange = (0, self.t_camExposure) # seconds
    lMask = self.currentX >= intRange[0]
    uMask = self.currentX <= intRange[1]
    
    x = self.currentX[lMask & uMask]
    y = self.currentY[lMask & uMask]
    
    camCharge = np.trapz(y,x=x)  #accuracy issues with trapz? TODO: compare to MATLAB's quadgk
    self.sd['camCharge'] = camCharge * 1e9
    
    # integration range for spectrometer
    intRange = (0, self.t_spectrumExposure) # seconds
    lMask = self.currentX >= intRange[0]
    uMask = self.currentX <= intRange[1]
    
    x = self.currentX[lMask & uMask]
    y = self.currentY[lMask & uMask]
    
    spectroCharge = np.trapz(y,x=x)  # accuracy issues with trapz? TODO: compare to MATLAB's quadgk
    self.sd['spectroCharge'] = spectroCharge * 1e9
    
  def camAnalysis(self, camData):
    camData = signal.medfilt2d(camData.astype(np.float32),kernel_size=3) * self.camPhotonsPerCount
    camMax = camData.max()
    camAvg = camData.mean()
    print("Camera Maximum:",camMax,"[photons]")
    self.sd['camMax'] = float(camMax)
    
    if self.drawPlots:
      # for the image
      fig = plt.figure()
      ax = plt.matshow(camData,fignum=fig.number,origin='lower')
      ax.axes.xaxis.tick_bottom()
      plt.title('Camera|' + self.titleString)
      plt.colorbar(label='Photons')
    
    if self.fitSpot:
      xRes = camData.shape[1]
      yRes = camData.shape[0]
      camData1D = camData.reshape([camData.size])
      
      # this forms our initial guesses for the fit
      params = analyzer.moments(camData)
      
      # [amplitude, peakX, peakY, sigmaX, sigmaY, theta(rotation angle), avg (background offset level)]
      initial_guess = (camMax-camAvg, params[2], params[1], params[4], params[3], 0, camAvg)
      
      # Create x and y grid
      xv = np.linspace(0, xRes-1, xRes)
      yv = np.linspace(0, yRes-1, yRes)
      x, y = np.meshgrid(xv, yv)
      
      try:
        fitResult = opt.curve_fit(analyzer.twoD_Gaussian, (x, y), camData1D, p0=initial_guess, maxfev=300, full_output=True)
        popt = fitResult[0]
        pconv = fitResult[1]
        infodict = fitResult[2]
        mesg = fitResult[3]
        #print(infodict)
        #print(mesg)
        fitFail = False
      except:
        fitFail = True
        popt = initial_guess
        
      # the fit parameters in photons
      amplitude = popt[0]
      theta = popt[5]
      peakPos = (popt[1],popt[2])
      peakX = peakPos[0]
      peakX = peakPos[1]
      sigma = (popt[3],popt[4])
      sigmaX = sigma[0]
      sigmaY = sigma[1]
      baseline = popt[6]
      peak = amplitude+baseline
      
      volume = abs(2 * constants.pi * amplitude * sigmaX * sigmaY)
      
      if fitFail:
        print('Camera spot 2D gaussian fit failure')
      else:
        self.sd['camSpotAmplitude'] = amplitude
        self.sd['camSpotVolume'] = volume
  
      print("Camera Spot Height: {:.0f} [photons]".format(amplitude))
      print("Camera Spot Volume: {:.0f} [photon*pixel^2]".format(volume))    
    
      if self.drawPlots:
        # for the spot fit analysis
        fitSurface1D = analyzer.twoD_Gaussian((x, y), *popt)
        fitSurface2D = fitSurface1D.reshape([yRes,xRes])      
        
        # let's make some evaluation lines
        nPoints = 100
        nSigmas = 4 # line length, number of sigmas to plot in each direction
        rA = np.linspace(-nSigmas*sigma[0],nSigmas*sigma[0],nPoints) # radii (in polar coords for line A)
        AX = rA*np.cos(theta-np.pi/4) + peakPos[0] # x values for line A
        AY = rA*np.sin(theta-np.pi/4) + peakPos[1] # y values for line A
      
        rB = np.linspace(-nSigmas*sigma[1],nSigmas*sigma[1],nPoints) # radii (in polar coords for line B)
        BX = rB*np.cos(theta+np.pi/4) + peakPos[0] # x values for line B
        BY = rB*np.sin(theta+np.pi/4) + peakPos[1] # y values for line B    
      
        f = interpolate.interp2d(xv, yv, camData) # linear interpolation for data surface
      
        lineAData = np.array([float(f(px,py)) for px,py in zip(AX,AY)])
        lineAFit = np.array([float(analyzer.twoD_Gaussian((px, py), *popt)) for px,py in zip(AX,AY)])
      
        lineBData = np.array([float(f(px,py)) for px,py in zip(BX,BY)])
        lineBFit = np.array([float(analyzer.twoD_Gaussian((px, py), *popt)) for px,py in zip(BX,BY)])
      
        residuals = lineBData - lineBFit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((lineBData - np.mean(lineBData)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        fig, axes = plt.subplots(2, 2,figsize=(8, 6), facecolor='w', edgecolor='k')
        fig.suptitle('Camera|' + self.titleString, fontsize=10)
        axes[0,0].imshow(camData, cmap=plt.cm.copper, origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))
        if len(np.unique(fitSurface2D)) is not 1: # this works around a bug in contour()
          axes[0,0].contour(x, y, fitSurface2D, 3, colors='w')
        else:
          print('Warning: contour() bug avoided')
        axes[0,0].plot(AX,AY,'r') # plot line A
        axes[0,0].plot(BX,BY,'g') # plot line B
        axes[0,0].set_title("Image Data")
        axes[0,0].set_ylim([y.min(), y.max()])
        axes[0,0].set_xlim([x.min(), x.max()])
      
        axes[1,0].plot(rA,lineAData,'r',label='Data')
        axes[1,0].plot(rA,lineAFit,'k',label='Fit')
        axes[1,0].set_title('Red Line Cut')
        axes[1,0].set_xlabel('Distance from center of spot [pixels]')
        axes[1,0].set_ylabel('Magnitude [photons]')
        axes[1,0].grid(linestyle='--')
        handles, labels = axes[1,0].get_legend_handles_labels()
        axes[1,0].legend(handles, labels)        
      
        axes[1,1].plot(rB,lineBData,'g',label='Data')
        axes[1,1].plot(rB,lineBFit,'k',label='Fit')
        axes[1,1].set_title('Green Line Cut')
        axes[1,1].set_xlabel('Distance from center of spot [pixels]')
        axes[1,1].set_ylabel('Magnitude [photons]')
        axes[1,1].grid(linestyle='--')
        handles, labels = axes[1,1].get_legend_handles_labels()
        axes[1,1].legend(handles, labels)           
      
        axes[0,1].axis('off')
        
        logMessages = io.StringIO()
        print("Green Line Cut R^2 =", r2, file=logMessages)
        peak = amplitude+baseline
        print("Peak =", peak, file=logMessages)
        print("====Fit Parameters====", file=logMessages)
        print("Amplitude =", amplitude, file=logMessages)
        print("Center X =", peakPos[0], file=logMessages)
        print("Center Y =", peakPos[1], file=logMessages)
        print("Sigma X =", sigma[0], file=logMessages)
        print("Sigma Y =", sigma[1], file=logMessages)
        print("Rotation (in rad) =", theta, file=logMessages)
        print("Baseline =", baseline, file=logMessages)
        print("", file=logMessages)
        logMessages.seek(0)
        messages = logMessages.read()      
        
        axes[0,1].text(0,0,messages)

  # calculates a 2d gaussian's height, x, y position and x and y sigma values from surface height data
  def moments(data):
      """Returns (height, x, y, width_x, width_y)
      the gaussian parameters of a 2D distribution by calculating its
      moments """
      total = data.sum()
      X, Y = np.indices(data.shape)
      x = (X*data).sum()/total
      y = (Y*data).sum()/total
      col = data[:, int(y)]
      width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
      row = data[int(x), :]
      width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
      height = data.max()
      return height, x, y, width_x, width_y
    
  def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    returns a 1d vector representation of the height at position xy of a 2d gaussian surface where
    amplitude = gaussian peak height
    xo,yo is the peak's position
    sigma_x, sigma_y are the x and y standard deviations
    theta is the rotaion angle of the gaussian
    and
    offset is the surface's height offset from zero
    """
    x = xy[0]
    y = xy[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()  
      
  def currentAnalysis(self, x, y):
    y = abs(y)
    totalDuration = x[-1] - x[0]
    currentAverage = y.mean()
    self.sd['avgBeamCurrent'] = currentAverage*1e9 # in nanoamps
    print("Current Average:",currentAverage*1e9,"[nA]")
    
    # store these away for postAnalysis()
    self.currentX = x
    self.currentY = y
    
    if self.drawPlots:
      plt.figure()
      plt.plot(x*1000, y*1e9, marker='.', label='Data')
      plt.plot((x[0]*1000,x[-1]*1000), (currentAverage*1e9,currentAverage*1e9), 'r--', label='{:.0f} ms Average = {:.0f} [nA]'.format(totalDuration*1e3,currentAverage*1e9))
      plt.title('Beam Current|' + self.titleString)
      
      plt.xlabel('Time Since Trigger Event [ms]')
      plt.ylabel('Beam Current [nA]')
      plt.grid()
      plt.legend()           
      
  def spectAnalysis(self, xPlot, yPlot, y_scale):
    #y = y/y_scale # TODO: check scaling
    
    # wavelength range overwhich we'll fit
    fitRange = (685, 705) # nm
    lMask = xPlot >= fitRange[0]
    uMask = xPlot <= fitRange[1]
    
    x = xPlot[lMask & uMask]
    y = yPlot[lMask & uMask]
    y_shift = y - y[0]
    
    # fomulate our guesses here
    mod = LinearModel()
    lPars = mod.guess(y, x=x)
    lPars['intercept'].value = y[0]
    lPars['slope'].value = 0
    
    # try playing with this if the fits are bad
    sigmaGuessScale = 2 
    amplitudeGuessScale = 3
    
    mod = LorentzianModel(prefix='A_')
    A_zPars = mod.guess(y_shift, x=x)
    A_zPars['A_center'].value = A_zPars['A_center'].value - 1
    A_zPars['A_sigma'].value = A_zPars['A_sigma'].value/sigmaGuessScale
    A_zPars['A_amplitude'].value = A_zPars['A_amplitude'].value/amplitudeGuessScale * 0.65
    
    mod = LorentzianModel(prefix='B_')
    B_zPars = mod.guess(y_shift, x=x)
    B_zPars['B_center'].value = B_zPars['B_center'].value + 0.4
    B_zPars['B_sigma'].value = B_zPars['B_sigma'].value/sigmaGuessScale
    B_zPars['B_amplitude'].value = B_zPars['B_amplitude'].value/amplitudeGuessScale
  
    # these are our guesses
    guesses = lPars + A_zPars + B_zPars
    
    # this is our fit model
    mod = LinearModel() + LorentzianModel(prefix='A_') + LorentzianModel(prefix='B_')
             
    result  = mod.fit(y, guesses, x=x)
    #print(result.fit_report(min_correl=0.25)) # for fit analysis
    
    if self.drawPlots:
      plt.figure()
      plt.plot(xPlot,yPlot, marker='.',label='Data')
      plt.plot(x, result.best_fit, 'r-',label='Ruby Emission Fit')
      
      # for guess analysis:
      #plt.plot(x, y, 'bo')
      #plt.plot(x, result.init_fit, 'k--')
      #plt.plot(x, result.best_fit, 'r-')
      #plt.show()      
      
      plt.xlabel('Wavelength [nm]')
      plt.ylabel('Spectrometer Counts')
      plt.title('Emission Spectrum|' + self.titleString)
      plt.tight_layout()
      plt.legend()
      plt.grid()    
    
    R2 = 1 - result.residual.var() / np.var(y)
    R2Threshold = 0.8 # anything lower than this we'll consider a failed fit
      
    if result.success and (R2 > R2Threshold):
      aHeight = result.params['A_height'].value
      aCen = result.params['A_center'].value
      bHeight = result.params['B_height'].value
      bCen = result.params['B_center'].value
      print("Peak A: {:.0f} counts @ {:.2f} [nm]".format(aHeight, aCen))
      print("Peak B: {:.0f} counts @ {:.2f} [nm]".format(bHeight, bCen))
      self.sd['aHeight'] = aHeight
      self.sd['aCen'] = aCen
      self.sd['bHeight'] = bHeight
      self.sd['bCen'] = bCen
    else:
      self.sd['aHeight'] = None
      self.sd['aCen'] = None
      self.sd['bHeight'] = None
      self.sd['bCen'] = None      
      print("WARNING: Bad ruby peak fit.")
    
  def visitor(self, name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print('\t{:}--> {:}'.format(key,val))
        if val == 'Manta camera':
          self.t_camExposure = obj.attrs['CAM1:det1:AcquireTime_RBV']  # TODO: change this to capture_time
          #self.sd['t_camExposure'] = self.t_camExposure
        elif val == 'Thorlabs spectrometer':
          self.t_spectrumExposure = obj.attrs['CCS1:det1:AcquireTime_RBV']  # TODO: change this to capture_time
          #self.sd['t_spectrumExposure'] = self.t_spectrumExposure
        elif val == 'LairdTech temperature regulator':
          self.sd['temperature'] = obj.attrs['LT59:Temp1_RBV']

    if type(obj) is h5py._hl.dataset.Dataset:
      print(obj.name+' <-- dataset')
      if 'Manta camera' in obj.parent.attrs.values():  # camera plot
        try:
          camData = obj[:]
          self.camAnalysis(camData)
        except:
          print("Failed during camera image analysis.")

      elif ('Thorlabs spectrometer' in obj.parent.attrs.values()) and ('spectra' in obj.name) and ('y_data' in obj.name):  # spectrometer plot
        try:
          parent = obj.parent
          xPlot = parent.get('x_values')[:]
          xlen = len(xPlot)
          yPlot = parent.get('y_data')[0:xlen]  # TODO doubcle check this length
          y_scale = parent.get('y_scale')[0:xlen]
        
          self.spectAnalysis(xPlot, yPlot, y_scale)
        except:
          print("Failed during spectrum analysis.")
          
      elif ('PicoScope 4264, python' in obj.parent.attrs.values()) and ('wavefront' in obj.name) and ('y_data' in obj.name):
        try:
          parent = obj.parent
          x = parent.get('x_data')[:]
          y = parent.get('y_data')[:]
          self.currentAnalysis(x,y)
        except:
          print("Failed during current analysis.")
