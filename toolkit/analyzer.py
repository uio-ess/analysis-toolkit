import h5py
import dataset
import datafreeze
import os
import io
import re
import cv2
import time
import hashlib
import matplotlib.pylab as plt
import numpy as np
from lmfit.models import LorentzianModel, LinearModel, VoigtModel, GaussianModel
import scipy
from lmfit import Model
from scipy import signal
from scipy import constants
from scipy import optimize as opt
from scipy import interpolate

class Object(object):
  pass

class analyzer:
  """
  Analyzes HDF5 sample files and enters sample data into the sample database
  for the ESS proton beam imaging system
  """

  def __init__(self, files, database = ':memory:', drawPlots = False, freezeObj = None, fitSpot = True, verbose = False, do_sw_current_filter = False, cam_pv = ""):
    self.full_paths = []
    self.drawPlots = drawPlots
    self.fitSpot = fitSpot
    self.verbose = verbose
    self.db = dataset.connect('sqlite:///'+database)
    self.t = None  # the database table we'll be using here
    self.freezeObj = freezeObj
    self.do_sw_current_filter = do_sw_current_filter
    self.cam_pv = cam_pv
    
    self.sd = {}  # sample dictionary
    
    # close all the files and store their names. because we'll open them later with h5py
    for f in files:
      f.close()
      self.full_paths = self.full_paths + [f.name]
  
  def realtimeFit(self):
    if self.cam_pv == "":
      raise(ValueError("You must provide the PV name base for the camera image (something like CAM1)"))
    import epics
    title_string = 'Live'

    while True:
      cam_data = epics.caget("{:}:image1:ArrayData".format(self.cam_pv))
      xRes = epics.caget("{:}:det1:SizeX_RBV".format(self.cam_pv))
      yRes = epics.caget("{:}:det1:SizeY_RBV".format(self.cam_pv))
      camData = np.reshape(cam_data,(yRes,xRes))
      #camData = cam_data.reshape(yRes,xRes)
      #camData = np.rot90(camData)

      processed = self.process_cam_image(camData, title_string, draw_plots = self.drawPlots)

      if processed['saturated'] == False:
        print("Median filtered image peak: {:.0f} [counts]".format(processed['blur_peak']))
        print("Median filtered image volume: {:.3g} [counts]".format(processed['blur_volume']))
      if processed['good_fit']:
        totalVolume = 2 * constants.pi * processed['amplitude'] * processed['sigmaX'] * processed['sigmaY']
        print("Gaussian spot amplitude seen by camera: {:.0f} [counts]".format(processed['amplitude']))
        print("Gaussian spot volume seen by camera: {:.3g} [counts]".format(totalVolume))
        print("Gaussian spot sigma X : {:.2f} [pixels]".format(processed['sigmaX']))
        print("Gaussian spot sigma Y : {:.2f} [pixels]".format(processed['sigmaY']))
        print("Gaussian spot position X : {:.2f} [pixels]".format(processed['peakPosX']))
        print("Gaussian spot position Y : {:.2f} [pixels]".format(processed['peakPosY']))
        print("Gaussian spot rotation : {:.4f} [radians]".format(processed['theta']))
      
      self.drawPlots and plt.show()
      print("Done")
      print("Done")
      print('\033[H')
      #print(chr(27) + "[2J")
      time.sleep(0.5)
      #os.system('clear')
    
  def processFiles(self):
    # loop through each file in the input
    for full_path in self.full_paths:
      file_name = os.path.basename(full_path)
      print('Processing {:}...'.format(full_path))
      self.verbose and analyzer.printContents(full_path)
      try:
        self.processOneFile(full_path)
      except:
        print("WARNING: Failed to process {:}".format(file_name))
      self.drawPlots and plt.show()
    print("Done processing all files.")
    
    # dump the results to a csv file if the user asked for it
    if self.freezeObj is not None:
      if self.full_paths == []:  # we were called without processing new files
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

  def process_cam_image(self, data, title_string, draw_plots = False):
    """process a matrix of camera data"""
    ret = {}
    # values in pixels and counts
    ret['good_fit'] = False
    ret['saturated'] = True
    ret['sigmaX'] = 0
    ret['sigmaY'] = 0
    ret['peakPosX'] = 0
    ret['peakPosY'] = 0
    ret['amplitude'] = 0
    ret['theta'] = 0
    ret['blur_volume'] = 0
    ret['blur_peak'] = 0
    ret['background'] = 0

    if draw_plots:
      # plot the image
      fig = plt.figure()
      ax = plt.matshow(data, fignum=fig.number)
      ax.axes.xaxis.tick_bottom()
      plt.title('RAW Camera|' + title_string)
      plt.colorbar(label='Counts')

    cameraBits = 12  # bit depth of the camera
    nCameraValues = 2**cameraBits  # and so there are this many unique values a pixel can take on
    maxCamValue = nCameraValues - 1  # and so the maximum value a pixel can take on is this
    xRes = data.shape[0]  # image x resolution
    yRes =  data.shape[1]  # image y resolution
    nPix = xRes * yRes # number of pixels in camera image

    # corner method of finding background
    #cornerDim = 50
    #corner = camData[:cornerDim,-cornerDim:] # take corner of the image
    #background = corner.mean()

    # histogram method of finding background (better probs)
    # take a histogram of counts in image and call the bin with the most counts the background
    bins = np.bincount(data.ravel(),minlength=nCameraValues) # maybe gaussian blur the image first?
    background = bins.argmax().astype(np.int16)
    ret['background'] = background

    #print("Camera background found to be {:} counts.".format(background))

    # camData with background offset subtracted away
    bg0 = data.copy() - background  

    #camMax = camData.max()
    #camAvg = camData.mean()
    #print("Camera Maximum:",camMax * photons_per_count,"[photons]")
    #self.sd['camMax'] = float(camMax * photons_per_count)

    # global auto-threshold the image (for finding the substrate)
    blur = data.copy() # copy camera data so we don't mess up the origional
    blur = cv2.medianBlur(src=blur, ksize=5) # median filter here
    bg0_blur = blur - background  # camData with background offset subtracted away

    # detect camera saturation
    saturation_percent_of_max = 5  # any pixel within this percent of max is considered saturated
    saturation_percent_of_pixels = 1  # if any more than this percent of total pixels in ROI are saturated, then we set the saturated flag for the image
    sat_thresh = np.int16(round((1-(saturation_percent_of_max / 100)) * maxCamValue))  # any pixel over this value is saturated
    nSatPix = np.count_nonzero(blur.ravel() >= sat_thresh)  # number of saturated pixels
    cameraSaturated = True
    if nSatPix < nPix*saturation_percent_of_pixels / 100:
      cameraSaturated = False
      ret['saturated'] = False
    else:
      print("WARNING: Image saturation detected. >{:}% of pixels are within {:}% of their max value".format(saturation_percent_of_pixels, saturation_percent_of_max))

    cantFindSubstrate = True
    if (not cameraSaturated):
      ret['blur_peak'] = bg0_blur.max()
      ret['blur_volume'] = bg0_blur.sum()

      #self.sd['blurVolume'] = bg0_blur.sum() * photons_per_count
      #self.sd['blurAmplitude'] = bg0_blur.max() * photons_per_count
      
      #print("Median filtered image peak: {:.0f} [photons]".format(ret['blur_peak']))
      #print("Median filtered image volume: {:.0f} [photons]".format(ret['blur_volume']))
        
      # global auto-threshold the image (for finding the substrate)
      thresh_copy = data.copy() # copy camera data so we don't mess up the origional
      thresh_copy = (thresh_copy/nCameraValues*(2**8-1)).round()  # re-normalize to be 8 bit
      thresh_copy = thresh_copy.astype(np.uint8) # information loss here, though only for thresh finding just because the big (k=15 filter makes) cv2 puke for 16bit numbers :-P
      thresh_blur = cv2.medianBlur(src=thresh_copy, ksize=15) # big filter here    
      tret,thresh = cv2.threshold(thresh_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # global auto-threshold
      nz = cv2.findNonZero(thresh) # thresh --> bool
      if nz is not None:
        mar = cv2.minAreaRect(nz) # find our substrate, minimum area rectangle = ((center_x,center_y),(width,height), angle)
      else: # this gets hit when the camera data is dark
        mar = ((0,0),(0,0), 0)
        
      # caculate the ROI
      boxScaleFactor = 0.70 # reduce ROI by this factor to prevent substrate edge effects
      smaller = (mar[0],tuple([x*boxScaleFactor for x in mar[1]]), mar[2]) # scale box width and height
      box = cv2.boxPoints(smaller).astype(np.int0) # find new ROI corners
      
      # show user the new ROI
      whereIsROI = cv2.drawContours(thresh_blur, [box], 0, 127, 3)
      if draw_plots:
        # for the ROI image
        fig = plt.figure()        
        ax = plt.matshow(whereIsROI, fignum=fig.number, cmap='gray', vmin = 0, vmax = 255)
        ax.axes.xaxis.tick_bottom()
        plt.title('Camera ROI|' + title_string)
      
      # did we find the substrate?
      approxSubstrateArea = mar[1][0] * mar[1][1]  # compute substrate area
      substrateAreaFactor = 0.8    
      if nPix * substrateAreaFactor > approxSubstrateArea: # test if the ROI is huge
        cantFindSubstrate = False
        # now we'll crop the blurred camera data
        ROI = np.full(data.shape, np.NaN)  # ROI starts as NaNs TODO: check shape, maybe bg0_blur
        mask = np.zeros_like(data)  # now we'll build up a mask for the ROI TODO: check shape, maybe bg0_blur
        cv2.drawContours(mask, [box], 0, 255, -1)  # fill the ROI box with white in the mask array
        ROI[mask == 255] = bg0_blur[mask == 255]  # now copy the blurred image data from the ROI region into the ROI array 
      else:
        print("WARNING: Can't find the substrate")  # and quit if it's too big    

    if self.fitSpot and (not cantFindSubstrate) and (not cameraSaturated):
      # let's make some initial guesses for the 2d gaussian fit
      twoDG_model = Model(analyzer.twoD_Gaussian, independent_vars=['x','y', 'offset'])
      guesses = twoDG_model.make_params()      
      # the raw image moment calculation forms the basis of our guesses
      m = cv2.moments(bg0_blur)
      
      data_sum = m['m00']
      
      # "center of mass"
      guesses['yo'].value = m['m10']/data_sum
      guesses['xo'].value = m['m01']/data_sum
      
      #angle = 0.5 * np.arctan(2 * m['mu11'] / (m['mu20'] - m['mu02']))
      #guesses['theta'].value = abs(angle - constants.pi/4) - constants.pi/4# lol, wtf, check this against more examples
      guesses['theta'].value = 0  # can't really get angle guess right
      
      guesses['sigma_y'].value = np.sqrt(m['mu20']/m['m00'])
      guesses['sigma_x'].value = np.sqrt(m['mu02']/m['m00'])
      
      # take an average of the points around the peak to find amplitude
      xc = round(guesses['xo'].value)
      yc = round(guesses['yo'].value)
      guesses['amplitude'].value = bg0_blur[xc,yc]
      
      # Create x and y grid
      xv = np.linspace(0, xRes-1, xRes) #TODO: check if yRes needs to be used here
      yv = np.linspace(0, yRes-1, yRes) #TODO: check if xRes needs to be used here
      x, y = np.meshgrid(xv, yv, indexing='ij')
      #x, y = np.meshgrid(xv, yv)
      
      # here we fit the camera data to a 2D gaussian model
      fitFail = True
      try:
        fitResult = twoDG_model.fit(ROI, x=x, y=y, offset=0, params=guesses, nan_policy='omit', fit_kws={'maxfev': 500})
        if fitResult.success:
          fitFail = False
      except:
        pass
          
      if fitFail:
        print('Camera spot 2D gaussian fit failure \a')
      else:  # do these things when the fit does not fail
        # the fit parameters in counts and pixels
        ret['sigmaX'] = fitResult.params['sigma_x'].value
        ret['sigmaY'] = fitResult.params['sigma_y'].value
        ret['peakPosX'] = fitResult.params['xo'].value
        ret['peakPosY'] = fitResult.params['yo'].value
        ret['amplitude'] = fitResult.params['amplitude'].value
        ret['theta'] = fitResult.params['theta'].value

        if draw_plots:
          theta = fitResult.params['theta'].value
          amplitude = fitResult.params['amplitude'].value
          peakPos = (fitResult.params['xo'].value, fitResult.params['yo'].value)
          sigma = (fitResult.params['sigma_x'].value, fitResult.params['sigma_y'].value)
          fitSurface2D = twoDG_model.eval(x=x, y=y, offset=0, params=fitResult.params)
          
          # let's make some evaluation lines
          nPoints = 100
          nSigmas = 4 # line length, number of sigmas to plot in each direction
          rA = np.linspace(-nSigmas*sigma[0], nSigmas*sigma[0], nPoints) # radii (in polar coords for line A)
          AX = rA*np.cos(theta+np.pi/2) + peakPos[0] # x values for line A
          AY = rA*np.sin(theta+np.pi/2) + peakPos[1] # y values for line A
        
          rB = np.linspace(-nSigmas*sigma[1],nSigmas*sigma[1],nPoints) # radii (in polar coords for line B)
          BX = rB*np.cos(theta) + peakPos[0] # x values for line B
          BY = rB*np.sin(theta) + peakPos[1] # y values for line B
                
          f = interpolate.RectBivariateSpline(xv, yv, data) # linear interpolation for data surface
        
          lineAData = f.ev(AX,AY)
          lineAFit = twoDG_model.eval(x=AX, y=AY, offset=background, params=fitResult.params)
        
          lineBData = f.ev(BX,BY)
          lineBFit = twoDG_model.eval(x=BX, y=BY, offset=background, params=fitResult.params)
        
          residuals = lineBData - lineBFit
          ss_res = np.sum(residuals**2)
          ss_tot = np.sum((lineBData - np.mean(lineBData)) ** 2)
          r2 = 1 - (ss_res / ss_tot)
          
          fig, axes = plt.subplots(2, 2,figsize=(8, 6), facecolor='w', edgecolor='k')
          fig.suptitle('Camera|' + self.titleString, fontsize=10)
          axes[0,0].matshow(data, cmap=plt.cm.copper)
          axes[0,0].contour(y, x, mask, 3, colors='yellow', alpha=0.2)
          #thresh
          #whereIsROI2 = cv2.drawContours(camData, [box], 0, 50, 3)
          #axes[0,0].matshow(whereIsROI2, cmap=plt.cm.copper)  # 
          ax.axes.xaxis.tick_bottom()

          if len(np.unique(fitSurface2D)) is not 1: # this works around a bug in contour()
            axes[0,0].contour(y, x, fitSurface2D, 3, colors='gray', alpha=0.5)
          else:
            print('Warning: contour() bug avoided')
          
          axes[0,0].plot(AY,AX,'r', alpha=0.5) # plot line A
          axes[0,0].plot(BY,BX,'g', alpha=0.5) # plot line B
          axes[0,0].set_title("Image Data")
          axes[0,0].set_ylim([xv.max(), xv.min()])
          axes[0,0].set_xlim([yv.min(), yv.max()])
          axes[0,0].xaxis.tick_bottom()
        
          axes[1,0].plot(rA, lineAData, 'r', label='Data')
          axes[1,0].plot(rA, lineAFit, 'k', label='Fit')
          axes[1,0].set_title('Red Line Cut')
          axes[1,0].set_xlabel('Distance from center of spot [pixels]')
          axes[1,0].set_ylabel('Magnitude [counts]')
          axes[1,0].grid(linestyle = '--')
          handles, labels = axes[1,0].get_legend_handles_labels()
          axes[1,0].legend(handles, labels)        
        
          axes[1,1].plot(rB,lineBData,'g',label='Data')
          axes[1,1].plot(rB,lineBFit,'k',label='Fit')
          axes[1,1].set_title('Green Line Cut')
          axes[1,1].set_xlabel('Distance from center of spot [pixels]')
          axes[1,1].set_ylabel('Magnitude [counts]')
          axes[1,1].yaxis.set_label_position("right")
          axes[1,1].grid(linestyle='--')
          handles, labels = axes[1,1].get_legend_handles_labels()
          axes[1,1].legend(handles, labels)
          
        
          axes[0,1].axis('off')
          axes[0,1].set_title('Fit Details')
          
          logMessages = io.StringIO()
          print("Green Line Cut R^2 = {:0.8f}".format(r2), file=logMessages)
          print("Peak = {:+011.5f} [counts]\n".format(amplitude+background), file=logMessages)
          print("     ====Fit Parameters====", file=logMessages)
          print("Amplitude = {:+011.5f} [counts]".format(amplitude), file=logMessages)
          print("Center X  = {:+011.5f} [pixels]".format(peakPos[1]), file=logMessages)
          print("Center Y  = {:+011.5f} [pixels]".format(peakPos[0]), file=logMessages)
          print("Sigma X   = {:+011.5f} [pixels]".format(sigma[1]), file=logMessages)
          print("Sigma Y   = {:+011.5f} [pixels]".format(sigma[0]), file=logMessages)
          print("Rotation  = {:+011.5f} [radians]".format(theta), file=logMessages)
          print("Baseline  = {:+011.5f} [counts]".format(background), file=logMessages)
          logMessages.seek(0)
          messages = logMessages.read()      
          
          axes[0,1].text(0,0,messages, family='monospace')
      ret['fit_fail'] = fitFail
      return(ret)
    

  def fetchRootAttributes(self, root):
    """read root attributes from the file and create a database table and line in the table that we'll populate later"""
    session = root.attrs['session']
    self.sd['session'] = session
    safeSession = re.sub(r'\W+', '', session) # allow this to become a table name
    self.t = self.db.create_table(safeSession, primary_id='int_hash', primary_type=self.db.types.bigint)
    
    # form row hash/title string out of sample name, trigger count and session
    attr = 'sample_name'
    self.sd[attr] = root.attrs[attr]
    
    attr = 'trigger_id'
    self.sd[attr] = int(root.attrs[attr]) # hopefully nothing gets mangled by the int cast here...
    
    self.titleString = str(self.sd['trigger_id']) + '|' + self.sd['sample_name'] + '|' + self.sd['session']    
    
    # this is the hash we use for the uniqueness test when adding/updating the database
    self.sd['int_hash'] = int.from_bytes(hashlib.blake2s(self.titleString.encode(),digest_size=6).digest(),byteorder='big')
    
    # now immediately write the unfinished row to the database so we have something on the file in case we fail later
    self.t.upsert(self.sd, ['int_hash'], ensure=True)

    # other things i'm interested in here
    iWants = ('experiment_description', 'sub_experiment', 'timestamp')
    for thingIWant in iWants:
      if (thingIWant == 'timestamp') and ('File creation  time' in root.attrs):
        attribute = root.attrs['File creation  time'] # catch poorly named timestamp TODO: remove this old attribute name
      else:
        attribute = root.attrs[thingIWant]
      if type(attribute) is np.int64:
        attribute = int(attribute) # hopefully nothing is mangled here...
      self.sd[thingIWant] = attribute
      
    attr = 'nominal_beam_current'
    self.sd[attr] = root.attrs[attr] * 1e9
    
  def processOneFile(self, full_path):
    """called to handle processing of the data in a single file"""
    f = h5py.File(full_path, 'r')  # open to file for reading
    
    self.sd = {}  # initialize sample dictionary, here's where we store stuff we're gonna write to the database
    
    root_group = f['/']
    self.fetchRootAttributes(root_group)

    if 'images/CAM1' in f['/data']:
      cam_group = f['/data/images/CAM1']
      self.camAnalysis(cam_group)
    else:
      print('WARNING: Did not find camera data')      

    if 'spectra/CCS1' in f['/data']:
      spect_group = f['/data/spectra/CCS1']
      self.spectAnalysis(spect_group)
    else:
      print('WARNING: Did not find spectrometer data')

    if 'oscope/ps4264py' in f['/data']:
      current_group = f['/data/oscope/ps4264py']
      self.currentAnalysis(current_group)
    else:
      print('WARNING: Did not find oscilloscope data for current measurement')

    # record temperature
    if 'temperature/ECAT' in f['/data']:
      self.sd['temperature'] = f['/data/temperature/ECAT'].attrs['temperature']
    
    # now we'll do some analysis that had to wait until we'd done the above analysis
    try:
      self.postAnalysis()
    except:
      print("WARNING: Failed during postAnalysis()")
    
    # store what we've learned in our database
    self.t.upsert(self.sd, ['int_hash'], ensure=True)
    
    print("\n")
    
  def printContents(full_path):
    """ prints the contents of the file to the terminal
    """
    f = h5py.File(full_path, 'r')
    root = f.get('/')
    #print('/')

    # print top level attributes
    for key, val in root.attrs.items():
      print('/{:}--> {:}'.format(key,val))
    
    def printVisitor(name, obj):
      """prints out what it visits in an hdf5 file"""
      if type(obj) is h5py._hl.dataset.Dataset:
        print_name = "\u001b[07m{:}\u001b[0m".format(obj.name)  # highlight datasets with inversted colors
        print("{:}--> DATASET with shape {:} of {:}".format(print_name, obj.shape, obj.dtype))
      else:
        print_name = obj.name

      for key, val in obj.attrs.items():
        print('{:}/{:}--> {:}'.format(print_name, key, val))
    
    # walk through the HDF5 file here, and print out what we find...
    f.visititems(printVisitor)
    
    print('\n')

  def postAnalysis(self):
    """ analysis that depends on multiple data sources i.e camera and beam current data
    """
    
    #check that we have enough current data
    if self.t_camExposure >= self.currentX[-1]:
      print("WARNING: We only captured current data for {:} ms, but the camera exposed for {:} ms".format(self.currentX[-1]*1000,self.t_camExposure*1000))
      print("Skipping several final analysis steps")
    else:
      # integration range for camera
      intRange = (0, self.t_camExposure) # seconds
      lMask = self.currentX >= intRange[0]
      uMask = self.currentX <= intRange[1]
      
      x = self.currentX[lMask & uMask]
      y = self.currentY[lMask & uMask]
      
      camCharge = np.trapz(y, dx=self.dx) * -1  # volume under the current mesuremet curve while the camera was exposing
      
      self.sd['camCharge'] = camCharge * 1e9 # store camera charge as nC
      print('Nanocoulombs during camera exposure: {:}[nC]'.format(self.sd['camCharge']))
      print('Average current during camera exposure ({:}[ms]): {:}[nA]'.format(self.t_camExposure*1000,camCharge/self.t_camExposure*1e9))
      
      protons = round(camCharge/constants.e)
      
      if 'gaussianVolume' in self.sd:
        # gaus fit takes gaussian out to infinity and thus should not be impacted by sample size
        self.sd['photonsPerProtonGaus'] = self.sd['gaussianVolume'] * self.sd['samplePhotonsPerCamPhoton'] / protons
        print("Photons per proton from gaussian fit: {:.0f}".format(self.sd['photonsPerProtonGaus']))
        
      if 'blurVolume' in self.sd:
        # this will be messed up for small samples since many of the protons caught by the cup will not make light since they miss the sample
        self.sd['photonsPerProtonBlur'] = self.sd['blurVolume'] * self.sd['samplePhotonsPerCamPhoton'] / protons
        print("Photons per proton from median filtered image: {:.0f}".format(self.sd['photonsPerProtonBlur']))

    # integration range for spectrometer
    if self.t_spectrumExposure >= self.currentX[-1]:
      print("WARNING: We only captured current data for {:} ms, but the spectrometer exposed for {:} ms".format(self.currentX[-1]*1000,self.t_spectrumExposure*1000))
      print("Skipping final spectrometer analysis")
    else:
      intRange = (0, self.t_spectrumExposure) # seconds
      lMask = self.currentX >= intRange[0]
      uMask = self.currentX <= intRange[1]
      
      x = self.currentX[lMask & uMask]
      y = self.currentY[lMask & uMask]
      
      spectroCharge = np.trapz(y, dx=self.dx) * -1  # volume under the current mesuremet curve while the spectrometer was exposing
      self.sd['spectroCharge'] = spectroCharge * 1e9
    
  def camAnalysis(self, cam_group):
    """ fetches and analyzes camera data"""
    
    if 'Capture Time' in cam_group.attrs: # TODO: remove this old attr name
      self.t_camExposure = cam_group.attrs['Capture Time'] # TODO: remove this old attr name
    if 'acquire_duration' in cam_group.attrs:
      self.t_camExposure = cam_group.attrs['acquire_duration']
    self.sd['t_camExposure'] = self.t_camExposure
    
    if 'photons_per_count' in cam_group.attrs:
      photons_per_count = cam_group.attrs['photons_per_count']
    else:
      photons_per_count = 5.7817
      print('WARNING: using default photons_per_count value')
    
    if 'lens_transmission' in cam_group.attrs:
      lens_transmission = cam_group.attrs['lens_transmission']
    else:
      lens_transmission = 0.95
      print('WARNING: using default lens_transmission value')
    
    if 'f_number' in cam_group.attrs:
      f_number = cam_group.attrs['f_number']
    else:
      f_number = 2.8
      print('WARNING: using default f_number value')

    if 'focal_length' in cam_group.attrs:
      focal_length = cam_group.attrs['focal_length']
    else:
      focal_length = 50
      print('WARNING: using default focal_length value')

    if 'distance_to_target' in cam_group.attrs:
      distance_to_target = cam_group.attrs['distance_to_target']
    else:
      distance_to_target = 1120
      print('WARNING: using default distance_to_target value')    
    
    # NOTE: how valid is this math for non-point sources?
    # see https://en.wikipedia.org/wiki/Lambert%27s_cosine_law
    
    self.sd['f_number'] = f_number
    aperture_diameter = focal_length / f_number  # camera aperture diameter
    aperture_area = constants.pi * (aperture_diameter / 2) ** 2 # camera aperture area
    solid_angle = aperture_area / distance_to_target ** 2  # solid angle of camera aperture
    
    assumed_emission_steradians = 4 * constants.pi  # TODO: consider 2pi emission
    
    # for every photon the camera sees, this many photons were generated at the sample
    self.sd['samplePhotonsPerCamPhoton'] = assumed_emission_steradians / solid_angle / lens_transmission
    print("For every photon the camera sees, the sample generated {:.0f}".format(self.sd['samplePhotonsPerCamPhoton']))

    camData = cam_group['data'][:]

    processed_cam = self.process_cam_image(camData, self.titleString, draw_plots=self.drawPlots)

    if processed_cam['saturated'] == False:
      self.sd['blurVolume'] = processed_cam['blur_volume'] * photons_per_count
      self.sd['blurAmplitude'] = processed_cam['blur_peak'] * photons_per_count
      print("Median filtered image peak: {:.0f} [photons]".format(self.sd['blurAmplitude']))
      print("Median filtered image volume: {:.0f} [photons]".format(self.sd['blurVolume']))

    if processed_cam['good_fit']:
      totalVolume = 2 * constants.pi * processed_cam['amplitude'] * photons_per_count * processed_cam['sigmaX'] * processed_cam['sigmaY']
      self.sd['gaussianAmplitude'] = processed_cam['amplitude'] * photons_per_count
      print("Gaussian spot amplitude seen by camera: {:.0f} [photons] (that's {:.0f} photons per second)".format(self.sd['gaussianAmplitude'],self.sd['gaussianAmplitude']/self.t_camExposure))
      print("Gaussian spot amplitude generated by sample: {:.0f} [photons] (that's {:.0f} photons per second)".format(self.sd['gaussianAmplitude']*self.sd['samplePhotonsPerCamPhoton'],self.sd['gaussianAmplitude']*self.sd['samplePhotonsPerCamPhoton']/self.t_camExposure))
      self.sd['gaussianVolume'] = totalVolume
      print("Gaussian spot volume seen by camera: {:.0f} [photons]  (that's {:.3g} photons per second)".format(totalVolume,totalVolume/self.t_camExposure))
      print("Gaussian spot volume generated by sample: {:.3g} [photons] (that's {:.3g} photons per second)".format(totalVolume*self.sd['samplePhotonsPerCamPhoton'],totalVolume*self.sd['samplePhotonsPerCamPhoton']/self.t_camExposure))
      self.sd['sigmaA'] = processed_cam['sigmaX']
      self.sd['sigmaB'] = processed_cam['sigmaY']


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
    
  def twoD_Gaussian(x,y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    returns a 1d vector representation of the height at position xy of a 2d gaussian surface where
    amplitude = gaussian peak height
    xo,yo is the peak's position
    sigma_x, sigma_y are the x and y standard deviations
    theta is the rotaion angle of the gaussian
    and
    offset is the surface's height offset from zero
    """
    #x = xy[0]
    #y = xy[1]
    #xo = float(xo)
    #yo = float(yo)
    #xmg,ymg = np.meshgrid(x, y)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    #return g.ravel()
    return g
      
  def currentAnalysis(self, current_group):
    
    raw_data = current_group['y_data'][:]
    if 'voltage_scale' in current_group['y_data'].attrs:
      t0 = current_group['y_data'].attrs['t0']
      t_end = current_group['y_data'].attrs['t_end']
      x = np.linspace(t0,t_end,len(raw_data))
      y = np.empty(raw_data.size, dtype=type(current_group['y_data'].attrs['voltage_scale']))
      np.multiply(raw_data, current_group['y_data'].attrs['voltage_scale'], y)
      np.subtract(y, current_group['y_data'].attrs['voltage_offset'], y)
      np.multiply(y, current_group['y_data'].attrs['current_scale'], y)          
    else: # TODO: remove old format current analysis
      y = raw_data
      t0 = current_group['y_data'].attrs['t0']
      t_end = current_group['y_data'].attrs['t_end']
      x = np.linspace(t0,t_end,len(y))
    
    #y = y * -1 # make the current positive
    #y = abs(y)
    totalDuration = x[-1] - x[0]
    currentAverage = y.mean()
    self.sd['avgBeamCurrent'] = currentAverage*1e9 # in nanoamps
    
    print("Current Average:",currentAverage*1e9,"[nA]")
    
    # store these away for postAnalysis()
    self.currentX = x
    self.currentY = y
    self.dx = current_group.attrs['Actual Sampling Interval']
    
    if self.do_sw_current_filter:
      # for filtering
      f_sample = 1/self.dx
      f_niq = f_sample/2
    
      filt_freq = 1000 # Hz for low pass cutoff
    
      # the Buterworth filter
      N  = 2    # Filter order
      Wn = filt_freq / f_niq # for Cutoff frequency
      B, A = scipy.signal.butter(N, Wn, output='ba')
     
      # apply the filter
      yf = scipy.signal.filtfilt(B,A, y)
      self.currentYf = yf
    
    #FFT = abs(scipy.fft(y))
    #dx = self.dx
    #freqs = scipy.fftpack.fftfreq(y.size, dx)
    
    if self.drawPlots:
      plt.figure()
      plt.plot(x*1000, y*1e9, marker='.', label='Data')
      if self.do_sw_current_filter:
        plt.plot(x*1000, yf*1e9, marker='.', label='Filtered in software with {:} Hz lowpass'.format(filt_freq), color = 'yellow')
      plt.plot((x[0]*1000,x[-1]*1000), (currentAverage*1e9,currentAverage*1e9), 'r--', label='{:.0f} ms Average = {:.0f} [nA]'.format(totalDuration*1e3,currentAverage*1e9))
      plt.title('Beam Current|' + self.titleString)
      
      plt.xlabel('Time Since Trigger Event [ms]')
      plt.ylabel('Beam Current [nA]')
      plt.grid()
      legend = plt.legend()
      frame = legend.get_frame()
      frame.set_facecolor('gray')
      
      #plt.figure()
      #plt.plot(freqs/1000,20*scipy.log10(FFT),'x')
      #plt.title('Beam Current FFT|' + self.titleString)
      
      #plt.xlabel('Frequency [kHz]')
      #plt.ylabel('Intensity [arb]')
      #plt.xlim(xmin=0)
      #plt.grid()
      
  def spectAnalysis(self, spect_group):
    """analyze the spectrometer data"""
    
    if 'Capture Time' in spect_group.attrs: # TODO: remove this old attr name
      self.t_spectrumExposure = spect_group.attrs['Capture Time'] # TODO: remove this old attr name
    if 'acquire_duration' in spect_group.attrs:
      self.t_spectrumExposure = spect_group.attrs['acquire_duration']
    self.sd['t_spectrumExposure'] = self.t_spectrumExposure    
    
    xPlot = spect_group['x_values'][:]
    y_scale = spect_group['y_scale'][:]
    xlen = len(xPlot)
    
    yPlot = spect_group['y_values'][0:xlen]
    
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
    
  