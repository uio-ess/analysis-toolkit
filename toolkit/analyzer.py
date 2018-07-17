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
  
  samplePhotonsPerCamPhoton = None

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
        print("WARNING: Failed to process {:}".format(f.name))
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
    
    session = root.attrs.get('session')
    self.sd['session'] = session
    safeSession = re.sub(r'\W+', '', session) # allow this to become a table name
    self.t = self.db.create_table(safeSession, primary_id='int_hash', primary_type=self.db.types.bigint)
    #self.t = self.db[str(hash(session))]
    # self.t.drop()
    
    # form row hash/title string out of sample name, trigger count and session
    attr = 'sample_name'
    self.sd[attr] = root.attrs.get(attr)
    
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
      if (thingIWant == 'timestamp') and ('File creation  time' in root.attrs):
        attribute = root.attrs.get('File creation  time') # catch poorly named timestamp
      else:
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
      print("WARNING: Failed during postAnalysis()")
    
    # store what we've learned in our database
    self.t.upsert(self.sd, ['int_hash'], ensure=True)
    #self.t.insert(self.sd, ensure=True)
    
    print("")
    print("")    

  def postAnalysis(self):
    """ analysis that depends on multiple data sources i.e camera and beam current data
    """
    
    # integration range for camera
    intRange = (0, self.t_camExposure) # seconds
    lMask = self.currentX >= intRange[0]
    uMask = self.currentX <= intRange[1]
    
    x = self.currentX[lMask & uMask]
    y = self.currentY[lMask & uMask]
    
    camCharge = np.trapz(y,x=x) * -1  #accuracy issues with trapz? TODO: compare to MATLAB's quadgk
    self.sd['camCharge'] = camCharge * 1e9
    
    
    protons = round(camCharge/constants.e)
    
    if 'gaussianVolume' in self.sd:
      # gaus fit takes gaussian out to infinity ans thus should not be impacted by sample size
      self.sd['photonsPerProtonGaus'] = self.sd['gaussianVolume'] * self.samplePhotonsPerCamPhoton / protons
      print("Photons per proton from gaussian fit: {:.0f}".format(self.sd['photonsPerProtonGaus']))
      
    if 'blurVolume' in self.sd:
      # this will be messed up for small samples since many of the protons caught by the cup will not make light since they miss the sample
      self.sd['photonsPerProtonBlur'] = self.sd['blurVolume'] * self.samplePhotonsPerCamPhoton / protons
      print("Photons per proton from median filtered image: {:.0f}".format(self.sd['photonsPerProtonBlur']))

    # integration range for spectrometer
    intRange = (0, self.t_spectrumExposure) # seconds
    lMask = self.currentX >= intRange[0]
    uMask = self.currentX <= intRange[1]
    
    x = self.currentX[lMask & uMask]
    y = self.currentY[lMask & uMask]
    
    spectroCharge = np.trapz(y,x=x) * -1  # accuracy issues with trapz? TODO: compare to MATLAB's quadgk
    self.sd['spectroCharge'] = spectroCharge * 1e9
    
  def camAnalysis(self, camData, photons_per_count, lens_transmission, f_number, focal_length, distance_to_target):
    """ focal_length and distance_to_target must be of the same units
    """
    # NOTE: how valid is this math for non-point sources?
    # see https://en.wikipedia.org/wiki/Lambert%27s_cosine_law
    
    aperture_diameter = focal_length / f_number  # camera aperture diameter
    aperture_area = constants.pi * (aperture_diameter / 2) ** 2 # camera aperture area
    solid_angle = aperture_area / distance_to_target ** 2  # solid angle of camera aperture
    
    assumed_emission_steradians = 4 * constants.pi  # TODO: consider 2pi emission
    
    # then for every photon seen by the camera, the sample emitted this many photons
    self.samplePhotonsPerCamPhoton = assumed_emission_steradians / solid_angle / lens_transmission
    
    if self.drawPlots:
      # for the image
      fig = plt.figure()
      ax = plt.matshow(camData, fignum=fig.number)
      ax.axes.xaxis.tick_bottom()
      plt.title('RAW Camera|' + self.titleString)
      plt.colorbar(label='Counts')
    
    cameraBits = 12  # bit depth of the camera
    nCameraValues = 2**cameraBits  # and so there are this many unique values a pixel can take on
    maxCamValue = nCameraValues - 1  # and so the maximum value a pixel can take on is this
    xRes = camData.shape[0]  # image x resolution
    yRes =  camData.shape[1]  # image y resolution
    nPix = xRes * yRes # number of pixels in camera image
    
    # corner method of finding background
    #cornerDim = 50
    #corner = camData[:cornerDim,-cornerDim:] # take corner of the image
    #background = corner.mean()
    
    # histogram method of finding background (better probs)
    # take a histogram of counts in image and call the bin with the most counts the background
    bins = np.bincount(camData.ravel(),minlength=nCameraValues) # maybe gaussian blur the image first?
    background = bins.argmax().astype(np.int16)
    
    print("Camera background found to be {:} counts.".format(background))
    
    # camData with background offset subtracted away
    bg0 = camData.copy() - background  

    #camMax = camData.max()
    #camAvg = camData.mean()
    #print("Camera Maximum:",camMax * photons_per_count,"[photons]")
    #self.sd['camMax'] = float(camMax * photons_per_count)
    
    # global auto-threshold the image (for finding the substrate)
    blur = camData.copy() # copy camera data so we don't mess up the origional
    blur = cv2.medianBlur(src=blur, ksize=5) # median filter here
    bg0_blur = blur - background  # camData with background offset subtracted away
    
    # detect camera saturation
    saturation_percent_of_max = 5  # any pixel within this percent of max is considered saturated
    saturation_percent_of_pixels = 4  # if any more than this percent of total pixels in ROI are saturated, then we set the saturated flag for the image
    sat_thresh = np.int16(round((1-(saturation_percent_of_max / 100)) * maxCamValue))  # any pixel over this value is saturated
    nSatPix = np.count_nonzero(blur.ravel() >= sat_thresh)  # number of saturated pixels
    cameraSaturated = True
    if nSatPix < nPix*saturation_percent_of_pixels / 100:
      cameraSaturated = False
    else:
      print("WARNING: Image saturation detected. >{:}% of pixels are within {:}% of their max value".format(saturation_percent_of_max, saturation_percent_of_pixels))
    
    if (not cameraSaturated):
      self.sd['blurVolume'] = bg0_blur.sum() * photons_per_count
      self.sd['blurAmplitude'] = bg0_blur.max() * photons_per_count
      
      print("Median Filtered Image Peak: {:.0f} [photons]".format(self.sd['blurAmplitude']))
      print("Median Filtered Image Volume: {:.0f} [photons]".format(self.sd['blurVolume']))
        
      # global auto-threshold the image (for finding the substrate)
      thresh_copy = camData.copy() # copy camera data so we don't mess up the origional
      thresh_copy = (thresh_copy/nCameraValues*(2**8-1)).round()  # re-normalize to be 8 bit
      thresh_copy = thresh_copy.astype(np.uint8) # information loss here, though only for thresh finding just because the big (k=15 filter makes) cv2 puke for 16bit numbers :-P
      thresh_blur = cv2.medianBlur(src=thresh_copy, ksize=15) # big filter here    
      ret,thresh = cv2.threshold(thresh_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # global auto-threshold
      nz = cv2.findNonZero(thresh) # thresh --> bool
      mar = cv2.minAreaRect(nz) # find our substrate, minimum area rectangle = ((center_x,center_y),(width,height), angle)
      
      # caculate the ROI
      boxScaleFactor = 0.70 # reduce ROI by this factor to prevent substrate edge effects
      smaller = (mar[0],tuple([x*boxScaleFactor for x in mar[1]]), mar[2]) # scale box width and height
      box = cv2.boxPoints(smaller).astype(np.int0) # find new ROI corners
      
      # show user the new ROI
      whereIsROI = cv2.drawContours(thresh_blur, [box], 0, 127, 3)
      if self.drawPlots:
        # for the ROI image
        fig = plt.figure()        
        ax = plt.matshow(whereIsROI, fignum=fig.number, cmap='gray', vmin = 0, vmax = 255)
        ax.axes.xaxis.tick_bottom()
        plt.title('Camera ROI|' + self.titleString)
      
      # did we find the substrate?
      cantFindSubstrate = True
      approxSubstrateArea = mar[1][0] * mar[1][1]  # compute substrate area
      substrateAreaFactor = 0.8    
      if nPix * substrateAreaFactor > approxSubstrateArea: # test if the ROI is huge
        cantFindSubstrate = False
        # now we'll crop the blurred camera data
        ROI = np.full(camData.shape, np.NaN)  # ROI starts as NaNs
        mask = np.zeros_like(camData)  # now we'll build up a mask for the ROI
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
      xv = np.linspace(0, xRes-1, xRes)
      yv = np.linspace(0, yRes-1, yRes)
      x, y = np.meshgrid(xv, yv,indexing='ij')
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
        print('Camera spot 2D gaussian fit failure')
      else:  # do these things when the fit does not fail
        # the fit parameters in photons
        amplitude = fitResult.params['amplitude'].value
        theta = fitResult.params['theta'].value
        peakPos = (fitResult.params['xo'].value, fitResult.params['yo'].value)
        peakX = peakPos[0]
        peakY = peakPos[1]
        sigma = (fitResult.params['sigma_x'].value,fitResult.params['sigma_y'].value)
        sigmaX = sigma[0]
        sigmaY = sigma[1]
        
        totalVolume = abs(2 * constants.pi * amplitude * photons_per_count * sigmaX * sigmaY)        
        self.sd['gaussianAmplitude'] = amplitude * photons_per_count
        print("Gaussian Spot Amplitude: {:.0f} [photons]".format(self.sd['gaussianAmplitude']))
        self.sd['gaussianVolume'] = totalVolume
        print("Gaussian Spot Volume: {:.0f} [photons]".format(totalVolume))
        self.sd['sigmaA'] = sigmaX
        self.sd['sigmaB'] = sigmaY
    
      if self.drawPlots:
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
        
        #xResCam = CamData.shape[0]
        #yResCam = CamData.shape[1]        
      
        f = interpolate.RectBivariateSpline(xv, yv, camData) # linear interpolation for data surface
      
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
        axes[0,0].matshow(camData, cmap=plt.cm.copper)
        ax.axes.xaxis.tick_bottom()
        #axes[0,0].imshow(camData, cmap=plt.cm.copper, 
        #          extent=(yv.min(), yv.max(), xv.min(), xv.max()))
        if len(np.unique(fitSurface2D)) is not 1: # this works around a bug in contour()
          axes[0,0].contour(y, x, fitSurface2D, 3, colors='gray')
        else:
          print('Warning: contour() bug avoided')
        axes[0,0].plot(AY,AX,'r') # plot line A
        axes[0,0].plot(BY,BX,'g') # plot line B
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
        axes[1,1].grid(linestyle='--')
        handles, labels = axes[1,1].get_legend_handles_labels()
        axes[1,1].legend(handles, labels)           
      
        axes[0,1].axis('off')
        
        logMessages = io.StringIO()
        print("Green Line Cut R^2 =", r2, file=logMessages)
        peak = amplitude+background
        print("Peak =", peak, file=logMessages)
        print("====Fit Parameters====", file=logMessages)
        print("Amplitude =", amplitude, file=logMessages)
        print("Center X =", peakPos[0], file=logMessages)
        print("Center Y =", peakPos[1], file=logMessages)
        print("Sigma X =", sigma[0], file=logMessages)
        print("Sigma Y =", sigma[1], file=logMessages)
        print("Rotation (in rad) =", theta, file=logMessages)
        print("Baseline =", background, file=logMessages)
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
      
  def currentAnalysis(self, x, y):
    #y = y * -1 # make the current positive
    #y = abs(y)
    totalDuration = x[-1] - x[0]
    currentAverage = y.mean()
    self.sd['avgBeamCurrent'] = currentAverage*1e9 # in nanoamps
    print("Current Average:",currentAverage*1e9,"[nA]")
    
    # store these away for postAnalysis()
    self.currentX = x
    self.currentY = y
    
    #FFT = abs(scipy.fft(y))
    #dx = x[1]-x[0]
    #freqs = scipy.fftpack.fftfreq(y.size, dx)
    
    if self.drawPlots:
      plt.figure()
      plt.plot(x*1000, y*1e9, marker='.', label='Data')
      plt.plot((x[0]*1000,x[-1]*1000), (currentAverage*1e9,currentAverage*1e9), 'r--', label='{:.0f} ms Average = {:.0f} [nA]'.format(totalDuration*1e3,currentAverage*1e9))
      plt.title('Beam Current|' + self.titleString)
      
      plt.xlabel('Time Since Trigger Event [ms]')
      plt.ylabel('Beam Current [nA]')
      plt.grid()
      plt.legend()
      
      #plt.figure()
      #plt.plot(freqs/1000,20*scipy.log10(FFT),'x')
      #plt.title('Beam Current FFT|' + self.titleString)
      
      #plt.xlabel('Frequency [kHz]')
      #plt.ylabel('Intensity [arb]')
      #plt.xlim(xmin=0)
      #plt.grid()      
      
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
        val = str(val)  # make sure we're comparing two str objects below
        if val == 'Manta camera':
          if 'Capture Time' in obj.attrs: # TODO: remove this old attr name
            self.t_camExposure = obj.attrs['Capture Time'] # TODO: remove this old attr name
          if 'acquire_duration' in obj.attrs:
            self.t_camExposure = obj.attrs['acquire_duration']         
          self.sd['t_camExposure'] = self.t_camExposure
        elif val == 'Thorlabs spectrometer':
          if 'Capture Time' in obj.attrs: # TODO: remove this old attr name
            self.t_camExposure = obj.attrs['Capture Time'] # TODO: remove this old attr name
          if 'acquire_duration' in obj.attrs:
            self.t_spectrumExposure = obj.attrs['acquire_duration']
          self.sd['t_spectrumExposure'] = self.t_spectrumExposure
        elif val == 'LairdTech temperature regulator': # TODO: remove this old equipment
          self.sd['temperature'] = obj.attrs['LT59:Temp1_RBV'] # TODO: remove this old equipment
        elif val == 'm-ethercat with EL3318':
          self.sd['temperature'] = obj.attrs['temperature']

    if type(obj) is h5py._hl.dataset.Dataset:
      print(obj.name+' <-- dataset')
      if 'Manta camera' in obj.parent.attrs.values():  # camera plot
        try:
          camData = obj[:]
          
          if 'photons_per_count' in obj.parent.attrs:
            photons_per_count = obj.parent.attrs['photons_per_count']
          else:
            photons_per_count = 5.7817
            print('WARNING: using default photons_per_count value')
          
          if 'lens_transmission' in obj.parent.attrs:
            lens_transmission = obj.parent.attrs['lens_transmission']
          else:
            lens_transmission = 0.95
            print('WARNING: using default lens_transmission value')
          
          if 'f_number' in obj.parent.attrs:
            f_number = obj.parent.attrs['f_number']
          else:
            f_number = 2.8
            print('WARNING: using default f_number value')

          if 'focal_length' in obj.parent.attrs:
            focal_length = obj.parent.attrs['focal_length']
          else:
            focal_length = 50
            print('WARNING: using default focal_length value')

          if 'distance_to_target' in obj.parent.attrs:
            distance_to_target = obj.parent.attrs['distance_to_target']
          else:
            distance_to_target = 1120
            print('WARNING: using default distance_to_target value')
          
          self.camAnalysis(camData, photons_per_count, lens_transmission, f_number, focal_length, distance_to_target)
        except:
          print("Failed during camera image analysis.")

      elif ('Thorlabs spectrometer' in obj.parent.attrs.values()) and ('spectra' in obj.name) and ('y_values' in obj.name):  # spectrometer plot
        try:
          parent = obj.parent
          xPlot = parent.get('x_values')[:]
          xlen = len(xPlot)
          yPlot = parent.get('y_values')[0:xlen]  # TODO doubcle check this length
          y_scale = parent.get('y_scale')[0:xlen]
        
          self.spectAnalysis(xPlot, yPlot, y_scale)
        except:
          print("Failed during spectrum analysis.")
          
      elif ('PicoScope 4264, python' in obj.parent.attrs.values()) and ('ps4264py' in obj.name) and ('y_data' in obj.name):
        try:
          raw_data = obj.value
          t0 = obj.attrs['t0']
          t_end = obj.attrs['t_end']
          x = np.linspace(t0,t_end,len(raw_data))
          dataI = np.empty(raw_data.size, dtype=type(obj.attrs['voltage_scale']))
          np.multiply(raw_data, obj.attrs['voltage_scale'], dataI)
          np.subtract(dataI, obj.attrs['voltage_offset'], dataI)
          np.multiply(dataI, obj.attrs['current_scale'], dataI)          
          self.currentAnalysis(x,dataI)
        except:
          print("Failed during current analysis.")
