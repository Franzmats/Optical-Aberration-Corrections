from Complementary import createLegalStructFieldName, h5_LV_attributes_to_struct,  h5_LV_MF_format_extract_parameters,shorten_group_name, zernfun, zernikeFittype
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d, griddata, NearestNDInterpolator, SmoothBivariateSpline
from scipy.optimize import curve_fit, leastsq
from scipy.stats import median_abs_deviation
import cmath
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os

# AVERAGE_PHASE get mean/median phase for the whole pupil or phase from single point
# input: 
# out.phase_fit (x,y,pupil) as output from fit_phase
# param.average.method: 'mean'/'median': take mean of all pixels in
# pupil scan, 'single' (default): use phase of single pixel as
# result. For fitted result, average is not necessary
#  param.average.pixel: pixel coordinates (x,y) for reference pixel in single' mode
# output: 
#  out.phase_average(pupil): vector containing the resulting phase in
#  each pupil 
   
def average_phase(param,out):
    
    print('### Choosing phase in each pupil ###')
    phase_fit=out['phase_fit']
    averageMethod=param['average']['method']
    ref=param['average']['pixel']

    phase_average=np.zeros(phase_fit.shape[0])
    for i in range(phase_fit.shape[0]):

        match averageMethod:
            case 'mean':
                phase_average[i]=np.mean(phase_fit[i,:,:].flatten())
            case 'median':
                phase_average[i]=np.median(phase_fit[i,:,:].flatten())
            case 'sum':
                phase_average[i]=np.sum(phase_fit[i,:,:].flatten())
            case 'single':
                phase_average[i]=phase_fit[i,ref[0],ref[1]]
            case _:
                print('No averageMethod provided! Tacking single pixel value')
                phase_average[i]=phase_fit[i,ref[0],ref[1]]
            
    out['phase_average']=phase_average
    return out



#FIT_PHASE Polynomial fit of the phases igoring outliers based on mad_sensitivity
# input: 
#   out['phase_unwrapped'] (x,y,pupil) as output from retrieve_phase
#   param['phase.mad']: sensitivity for discarding data based on median
#   absolute deviation
# output: 
#   out.phase_fit(x,y,pupil): 2D linear phase fit result
#   out.points_to_keep: points not discarded by median absolute
#   deviation of fit
#   out.phase_flat(x,y,pupil): raw data with linear term subtracted

def fit_phase(data, param,out):

    print('### Polynomial fitting of phase in pupils ###')

    x=data['x']
    y=data['y']
    im=data['im']
    phase_unwrapped=out['phase_unwrapped']
    phase_raw=out['phase_raw']
    mad_sensitivity=param['phase']['mad']
    x_0=param['phase']['ref'][0]
    y_0=param['phase']['ref'][1]

    phase_poly_int=np.zeros((im.shape[3],3,1))
    phase_poly_int_error = np.zeros((im.shape[3], im.shape[1], im.shape[0]))
    points_to_keep= np.zeros((im.shape[3], im.shape[1], im.shape[0]))
    phase_fit = np.zeros((im.shape[3], im.shape[1], im.shape[0]))
    
    # EOD scan meshgrid
    x_mesh, y_mesh = np.meshgrid(np.arange(phase_unwrapped.shape[2]), np.arange(phase_unwrapped.shape[1]))
    x_mesh = x_mesh.T
    y_mesh = y_mesh.T

    def poly11(x, a, b, c):
        return  a + b*x[0] + c*x[1] 

    for i in range(phase_unwrapped.shape[0]):
     
        phase_2_fit = phase_unwrapped[i,:,:]

        # perform the polynomial fit, where the parameter '_' represent the values of the covariance matrix, which are ignored
        popt, _ = curve_fit(poly11, (x_mesh.flatten(), y_mesh.flatten()), phase_2_fit.flatten())

        # create a lambda function to evaluate the fitted polynomial
        pol= lambda x, y: poly11((x, y), *popt)
        

         # Calculate fit error pixel by pixel
        phase_poly_int_error[i,:,:] = pol(x_mesh, y_mesh) - phase_2_fit

        # Points to keep based on median absolute deviation
        points_to_keep[i,:,:] = abs(phase_poly_int_error[i,:,:]) < median_abs_deviation(phase_poly_int_error[i,:,:], axis=None) * mad_sensitivity

        # perform the polynomial fit again considering the outliers out
        popt, _ = curve_fit(poly11, (x_mesh.flatten(), y_mesh.flatten()), phase_2_fit.flatten()*points_to_keep[i,:,:].flatten())
        pol= lambda x, y: poly11((x, y), *popt)
        phase_poly_int[i,:,:]=popt.reshape(phase_poly_int.shape[1],phase_poly_int.shape[2])
        phase_fit[i,:,:]=pol(x_mesh, y_mesh)


    out['phase_fit'] = phase_fit
    out['points_to_keep'] = points_to_keep
    out['phase_poly_fit'] = phase_poly_int
    out['phase_flat'] = phase_unwrapped - phase_fit


    if param['debug']:
        seg_rng=param['seg_rng']
        im_size=param['im_size']
        x_mesh2 = im_size*np.linspace(-1,1, x_mesh.shape[0])
        y_mesh2 = im_size*np.linspace(-1,1, y_mesh.shape[1])

        x_Mesh2, y_Mesh2 = np.meshgrid(x_mesh2, y_mesh2)

        ## Show data by pupil segment
        seg_idx = 1
        phase_axis_lim = [-2, 2]
      #___________________________________________________________________________________
        plt.figure(3)
        fig=plt.figure(3,figsize=(param['figure_position'][2]/100,param['figure_position'][3]/100),dpi=100)
        fig.suptitle('Phase fit by pupil segment', fontsize=24, fontweight='bold')
      #___________________________________________________________________________________
      # SLM pupil segments
        plt.subplot(3,2,1)
        plt.plot(x, y, 's')
        plt.plot(x[seg_idx], y[seg_idx], '*')
        plt.axis('square')
        plt.grid(True)
        plt.xlabel(r'$x_{slm}$')
        plt.ylabel(r'$y_{slm}$')
        plt.title('Pupil segmentation')
        
      #___________________________________________________________________________________
     # Raw phase
        plt.subplot(3, 2, 2)
        plt.imshow(phase_raw[seg_idx,:,:],origin='lower')
        plt.plot(y_0, x_0, 'ws')
        plt.clim(phase_axis_lim)
        plt.colorbar()
        plt.title('Raw phase')
        
      #___________________________________________________________________________________
     # Unwrapped phase
        plt.subplot(3,3,5)
        plt.imshow(phase_unwrapped[seg_idx,:,:],origin='lower')

        plt.title('Unwrapped phase')
        x_outliers = x_mesh[np.logical_not(points_to_keep[seg_idx,:,:])]
        y_outliers = y_mesh[np.logical_not(points_to_keep[seg_idx,:,:])]

        plt.plot(y_outliers, x_outliers, 'rs')
        plt.contour(phase_unwrapped[seg_idx,:,:], colors='k')
        plt.clim(phase_axis_lim)

      #___________________________________________________________________________________
     # Fit result
        plt.subplot(3,3,6)
        phase_2_fit = phase_unwrapped[seg_idx,:,:]
        popt, _ = curve_fit(poly11, (x_mesh.flatten(), y_mesh.flatten()), phase_2_fit.flatten())
        pol= lambda x, y: poly11((x, y), *popt)
        plt.imshow(pol(x_mesh,y_mesh),origin='lower')
       
        plt.contour(phase_unwrapped[seg_idx,:,:], colors='0.7') 
        plt.contour(pol(x_mesh,y_mesh), colors='k')
        plt.clim(*phase_axis_lim)
        plt.title('Fit result')
        
      #___________________________________________________________________________________
     # Line plots
        plt.subplot(6, 3, 7)
        plt.plot(phase_raw[seg_idx, int(x_0),:])
        plt.plot(phase_unwrapped[seg_idx, int(x_0),:])
        plt.ylim(phase_axis_lim)
        al = plt.legend(['Raw', 'Unwrapped'])
        #al['FontSize']=4
        #al['Position']=[0.1315, 0.6415, 0.0911, 0.0273]
      #___________________________________________________________________________________
        plt.subplot(6, 3, 10)
        plt.plot(phase_raw[seg_idx, :, int(y_0)])
        plt.plot(phase_unwrapped[seg_idx,:, int(y_0)])
        plt.ylim(phase_axis_lim)
        
      #___________________________________________________________________________________
       # Measurements
        plt.subplot(3, 4, 9)
        plt.imshow(im[:, :, 0, seg_idx],origin='lower')
        plt.plot(param['average']['pixel'][0], param['average']['pixel'][1], 'xr')

        plt.subplot(3, 4, 10)
        plt.imshow(im[:, :, 1, seg_idx],origin='lower')

        plt.plot(param['average']['pixel'][0], param['average']['pixel'][1], 'xr')

        plt.subplot(3, 4, 11)
        plt.imshow(im[:, :, 2, seg_idx],origin='lower')

        plt.plot(param['average']['pixel'][0], param['average']['pixel'][1], 'xr')

        plt.subplot(3, 4, 12)
        plt.imshow(im[:, :, 3, seg_idx],origin='lower')
        plt.plot(param['average']['pixel'][0], param['average']['pixel'][1], 'xr')

        

      #___________________________________________________________________________________
        plt.figure(4)
        fig=plt.figure(4,figsize=(param['figure_position'][2]/100,param['figure_position'][3]/100),dpi=100)
        fig.suptitle('Overall phase recovery')
        plt.clf()

        for i in range(phase_unwrapped.shape[0]):
            yas=plt.imshow(np.fliplr(phase_unwrapped[i,:,:].T), extent=[x[i]-im_size, x[i]+im_size, y[i]-im_size, y[i]+im_size],origin='lower', vmin=-2, vmax=2)
            plt.contour(x[i]+x_mesh2, y[i]+y_mesh2, np.fliplr(phase_unwrapped[i,:,:].T), colors='r')
            plt.contour(x[i]+x_mesh2, y[i]+y_mesh2, np.fliplr(phase_fit[i,:,:].T), colors='k')
            plt.plot(x_Mesh2[~points_to_keep[i,:,:].astype(bool)].ravel() + x[i], y_Mesh2[~points_to_keep[i,:,:].astype(bool)].ravel() + y[i], 'r.')
            
        plt.title('Overall retrieved phase', fontsize=24, fontweight='bold')
        cbar=plt.colorbar(yas)
        cbar.set_label('Phase (\u03C0)')
        #plt.show()

      #___________________________________________________________________________________
      # Subtract linear term for single pupil fit
        idx1_rng = np.arange(-5, 6) + np.ceil(x_mesh.shape[0] / 2).astype(int)  # The (-5:5) range should be a parameter
        idx2_rng = np.arange(-5, 6) + np.ceil(x_mesh.shape[1] / 2).astype(int)
        phase_poly_zero_tilt = np.zeros((len(x), len(idx1_rng), len(idx2_rng)))
        phase_poly1 =np.zeros((1,len(x)))
        for i1 in range(len(idx1_rng)):
            for i2 in range(len(idx2_rng)):
                for i in range(im.shape[3]):
                    phase_2_fit = phase_unwrapped[i,:,:]
                    popt, _ = curve_fit(poly11, (x_mesh.flatten(), y_mesh.flatten()), phase_2_fit.flatten())
                    pol= lambda x, y: poly11((x, y), *popt)
                    phase_poly1[0,i] =pol(x_mesh[idx1_rng[i1], idx2_rng[i2]], y_mesh[idx1_rng[i1], idx2_rng[i2]])

                popt, _ = curve_fit(lambda xy, a, b: a*xy[:,0] + b*xy[:,1], np.column_stack([x[seg_rng], y[seg_rng]]), phase_poly1[0,seg_rng])
                phase_seg_fit = popt[0]*x + popt[1]*y
                phase_poly_zero_tilt[:,i1,i2] = phase_poly1 - phase_seg_fit
      #___________________________________________________________________________________
        plt.figure(5)
        fig=plt.figure(5,figsize=(param['figure_position'][2]/100,param['figure_position'][3]/100),dpi=100)
        fig.suptitle('Recovered phase after tilt subtraction')
        plt.clf()

        for i in seg_rng:
            ru=plt.imshow(
                phase_poly_zero_tilt[i,:,:], extent=[x[i]-im_size, x[i]+im_size, y[i]-im_size, y[i]+im_size],origin='lower', vmin=-0.3, vmax=0.3)
            plt.contour(x[i]+im_size*np.linspace(-1,1,phase_poly_zero_tilt.shape[1]), y[i]+im_size*np.linspace(-1,1,phase_poly_zero_tilt.shape[2]),phase_poly_zero_tilt[i,:,:], colors='black')


        cbar = plt.colorbar(ru)
        cbar.set_label('Phase (\u03C0)')
        plt.title('Recovered phase after px by px tilt subtraction', fontsize=24, fontweight='bold')
        #plt.show()

      #____________________________________________________________________

    return out

        

def generate_filenamePrefix(param):

    str=param['save']['filenamePrefix']

    # Interpolation method
    match param['interpolation']['method']:
        case 'zernike':
            str=[str, '{} order_{:.0f}'.format(param['interpolation']['method'], param['interpolation']['FitOrder'])]
        case 'thin_plate_smoothing_spline':
            str=[str,'{} smth_{:.2g}'.format(param['interpolation']['method'], param['interpolation']['smoothing'])]
        case _:
            str=[str,'{}'.format(param['interpolation']['method'])]
    
    # Average method
    str=[str,'avg_{}'.format(param['average']['method'])]

    # Cut method
    if param['interpolation']['cut']['on']:
        str=[str,"cut_{:.2f}".format(param['interpolation']['cut']['Factor']+1)]
    
    # Zero minimum
    if param['save']['filenamePrefix']:
        str=[str, 'zeromin']
    
    param['save']['filenamePrefix']=str
    


# phase_unwrap_regular
# Unwraps 2D phase map and keeps the average phase of the region were the
# pixel of coordinates 'ind_keep' is. 

def phase_unwrap_regular(phase_in, ind_keep):

    #Unwrap phase
    phase_unwrapped = np.unwrap(np.unwrap(phase_in, axis=0), axis=1)
    #Compute correction
    unwrap_correction=phase_unwrapped-phase_in

    #Initial value for phase offset
    phase_offset_0=unwrap_correction[int(ind_keep[0]),int(ind_keep[1])]

    #Compute output
    phase_out=phase_unwrapped-phase_offset_0
    phase_correction_final=phase_out-phase_in

    return phase_out #, phase_correction_final

def poly_fitfunction_from_order(order):

    match order:
        case 1:
            fitFunction='poly11'
        case 2:
            fitFunction='poly22'
        case 3:
            fitFunction='poly33'
        case 4:
            fitFunction='poly44'
        case 5:
            fitFunction='poly55'
        case 6:
            fitFunction='poly66'
        case 7:
            fitFunction='poly77'
        case 8:
            fitFunction='poly88'


# PUPIL_SEGMENTATION_INIT initializes sets default values in param
# struct. Needs to be called first, as previously defined values will
# be overwritten
# input: 
#   data: pupil segmentation measurement data
# output: 
#   param: parameters for retrieving the phase to be displayed on the SLM
#   out: empty struct to be filled with phase output arrays

def pupil_segmentation_init(data):

    print('### Initializing parameters ###')
    out=dict()
    param=dict()

    # General
    param['im_size']=15 #S ize of small images on SLM axes
    extra_points_end=0 # Pupils segments to be ignored at the end of the measurement
    extra_points_beginning=0 # Pupil segments to be ignored at the beginning of the measurement
    param['seg_rng'] = range(extra_points_beginning , data['im'].shape[3] - extra_points_end)
    

    # Debugging
    param['debug']=0
    param['figure_position']=[150, 50, 2150, 1250]

    # Phase recovery 
    param['phase'] = dict()
    param['phase']['ref'] = [np.floor(np.shape(data['im'])[0]/2+1), np.floor(np.shape(data['im'])[1]/2+1)]
    param['phase']['int']=0 # Plot intensity of beam
    param['phase']['mad']=4 # Mad sensitivity to discard outliers from polyfit

    # Phase averaging
    param['average'] = dict()
    param['average']['method']='median'
    param['average']['pixel']=param['phase']['ref'] # Image index for phase calculation

    # Unwrapping over pupils
    param['unwrap'] = dict()
    param['unwrap']['method']='none'
    param['unwrap']['interpMethod']='nearest'

    # SLM interpolation
    param['interpolation'] = dict()
    param['interpolation']['phase_choice']='zero_defocus' # 'direct', 'zero_tilt', 'zero_defocus'

    # Extend region to fit 
    param['interpolation']['extend']=dict()
    param['interpolation']['extend']['method']='direct' # 'repeat outer (last)', 'repeat outer (first)'
    param['interpolation']['extend']['OuterDiameterLength']=24
    param['interpolation']['extend']['OuterDiameterExtension']=1.3

    # Avoid phase wrap
    param['interpolation']['zerominimum']=1

    # Interpolation
    param['interpolation']['method']='zernike' # 'poly', 'scattered', 'thin_plate_smoothing_spline'
    param['interpolation']['FitOrder']=6 # Up to 10 possible for zernike; up to 8 posible for poly
    param['interpolation']['smoothing']=5e-4
    param['interpolation']['orientation']=dict()
    param['interpolation']['orientation']['invertcenterx']=0
    param['interpolation']['orientation']['invertcentery']=0
    param['interpolation']['orientation']['flipxy']=0
    param['interpolation']['orientation']['transpose']=0
    param['interpolation']['orientation']['flipx']=0
    param['interpolation']['orientation']['flipy']=0

    # Cut beam region
    param['interpolation']['cut']=dict()
    param['interpolation']['cut']['on']=1
    param['interpolation']['cut']['Factor']=0.3 #beam cut at (1+cutFactor)*R_max
    param['interpolation']['cut']['extrapolate']=1

    # Saving
    param['save']=dict()
    param['save']['on']=0
    param['save']['dirname']=data['dirname']
    param['save']['filenamePrefix']='phase_slm'

    #     param.save.filenamePrefix = sprintf('phase_slm_%s_order%.0f_%s_cut%.2f',...
    #     param.interpolation.method, param.average.method, param.interpolation.cut.Factor+1);

    return param,out 


def read_pupil_segmentation(filename, dirname):

    filename_full = dirname+"/"+filename

    with h5py.File(filename_full, 'r') as f:
        
        parameters_info = f.get('/parameters')

        # Display datasets
        print("Datasets in /data:")
        for key in f['/data'].keys():
            print(key)
        
        print("Datasets in /parameters:")
        for key in parameters_info.keys():
            print(key)

        
            
        data = dict()
        data['p'] = h5_LV_MF_format_extract_parameters(parameters_info)
        data['im'] = np.transpose(np.array(f['/data/images'])) # index1,index2: pixels of image, index3: pre-known phase, index4: pupil index
        data['phi'] = np.array(f['/data/segmentation_phase'])
        data['x'] = np.array(f['/data/segmentation_x'])
        data['y'] = np.array(f['/data/segmentation_y'])
        data['filename'] = filename
        data['dirname'] = dirname

    return data



# RETRIEVE_PHASE Retrieves and unwraps the phase in each pupil
# input: 
#   data.im(x,y,phase,pupil)
#   param.phase.ref phase invariant reference point (2 coordinates)
# output: 
#   out.phase_unwrapped: spatially unwrapped phase in units of pi
#   out.phase_raw: phase in units of pi
#   out.a,out.b as defined in I = a + b * cos(phi+d_phi)

def retrieve_phase(data,param,out):
    print('### Retrieving phase from images ###')
    x_0=param['phase']['ref'][0]
    y_0=param['phase']['ref'][1]
    im=data['im']
    x=data['x']
    y=data['y']
    phi=data['phi']
    N_phi=len(phi)
    phase_raw = np.zeros((im.shape[3], im.shape[1], im.shape[0]))
    phase_unwrapped = np.zeros((im.shape[3], im.shape[1], im.shape[0]))
    a = np.zeros((im.shape[3], im.shape[1], im.shape[0]))
    b= np.zeros((im.shape[3], im.shape[1], im.shape[0]))


    for i in range(im.shape[3]):
        # Direct phase for arbitrary phases
        y_t = im[:, :, 0, i] * 0
        x_t = im[:, :, 0, i] * 0
       

        for k in range(N_phi):
            y_t = y_t + im[:, :, k, i] * np.sin(-phi[k]*np.pi)
            x_t = x_t + im[:, :, k, i] * np.cos(-phi[k]*np.pi)
        
        val=np.arctan2(y_t, x_t)
        
        phase_raw[i,:,:] = val

        # Unwrapping
        val1=phase_unwrap_regular(val, [x_0, y_0, i])
        phase_unwrapped[i,:,:]= val1

        # Calculate and b defined as I=a+b*cos(phi=d_phi), where:
        # a=E_ref^2 + E_pupil^2
        # b=2*E_ref*E_pupil
        aval = (im[:, :, 0, i] + im[:, :, 2, i] + im[:, :, 1, i] + im[:, :, 3, i]) / 4
        bval = ((im[:, :, 0, i] - im[:, :, 2, i]) / np.cos(phase_unwrapped[i,:,:]) + (im[:, :, 3, i] - im[:, :, 1, i]) / np.sin(phase_unwrapped[i,:,:]))/4
        a[i,:,:] = aval
        b[i,:,:] = bval
      
    #original matlab implementation needed rework to deal with complex values
    #chi = np.real(a/b - np.sqrt((a/b)**2 - 1))
    c=(a/b)**2-1
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                        d=cmath.sqrt(c[i,j,k])
                        c[i,j,k]=np.real(a[i,j,k]/b[i,j,k]-d)
    chi=c

    # Put everything in units of (pi)
    out['phase_raw'] = phase_raw/np.pi
    out['phase_unwrapped'] = phase_unwrapped/np.pi
    out['a'] = a
    out['b'] = b

    #___________________________________________________________________________________
    if param['phase']['int']:
        # average and fit
        seg_rng = param['seg_rng']
        int_raw = np.mean(np.mean(a, axis=0), axis=0)
        int_fit = np.polyfit([x[seg_rng], y[seg_rng]], int_raw[seg_rng], deg=2)
        int_res = np.polyval(int_fit, (x[seg_rng], y[seg_rng]))
        
        # plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].scatter(x[seg_rng], y[seg_rng], c=int_raw[seg_rng], s=200, cmap='viridis', alpha=0.7, edgecolors='none')
        axs[0].set_aspect('equal tight')
        axs[0].set_title('Sum of intensities')
        axs[1].scatter(x[seg_rng], y[seg_rng], c=int_res[seg_rng], s=200, cmap='viridis', alpha=0.7, edgecolors='none')
        axs[1].set_aspect('equal tight')
        axs[1].set_title('Sum of intensities, quad fit')
        plt.show()
    #___________________________________________________________________________________
    if param['debug']:
        x = data['x']
        y = data['y']
        seg_rng = param['seg_rng']
        im_size = param['im_size']

        x_mesh, y_mesh = np.meshgrid(np.arange(phase_unwrapped.shape[2]), np.arange(phase_unwrapped.shape[1]))
        x_mesh = x_mesh.T
        y_mesh = y_mesh.T
        x_mesh2 = im_size*np.linspace(-1, 1, x_mesh.shape[1])
        y_mesh2 = im_size*np.linspace(-1, 1, y_mesh.shape[0])
        #___________________________________________________________________________________
        fig=plt.figure(1,figsize=(param['figure_position'][2]/100,param['figure_position'][3]/100),dpi=100)
        fig.suptitle('Raw data')
        fig.clf()
    
        for j in range(im.shape[2]):
            
            plt.subplot(2, 4, j+1)
            for i in seg_rng:
                plt.imshow(np.fliplr(im[:,:,j,i]).T, extent=[x[i]-im_size, x[i]+im_size, y[i]-im_size, y[i]+im_size],origin='lower', vmin=0.0, vmax=200)
                plt.autoscale(False)
            plt.title('%.1f\u03C0' % phi[j])
            plt.xlabel(r'')
            plt.ylabel(r'')
            plt.colorbar()
            plt.autoscale(True)

        #___________________________________________________________________________________
        plt.subplot(2, 4, 5)
        for i in range(im.shape[3]):
            plt.imshow(np.fliplr(phase_raw[i, :, :]).T, extent=[x[i]+x_mesh2.min(), x[i]+x_mesh2.max(), y[i]+y_mesh2.min(), y[i]+y_mesh2.max()],origin='lower', vmin=-3, vmax=3)
            plt.autoscale(False)

        plt.autoscale(True)
        plt.colorbar()
        plt.title('I = a + b * cos(\u03C0 + \u0394)')
        plt.xlabel(r'')
        plt.ylabel(r'')
       
        #___________________________________________________________________________________
        plt.subplot(2, 4, 6)
        for i in range(im.shape[3]):
            plt.imshow(np.fliplr(a[i, :, :]).T, extent=[x[i]+x_mesh2.min(), x[i]+x_mesh2.max(), y[i]+y_mesh2.min(), y[i]+y_mesh2.max()],origin='lower', vmin=0.0, vmax=200)
            plt.autoscale(False)
        plt.colorbar()

        plt.title('a')
        plt.xlabel(r'')
        plt.ylabel(r'')
        plt.autoscale(True)
        #___________________________________________________________________________________
        plt.subplot(2, 4, 7)
        for i in range(im.shape[3]):
            plt.imshow(np.fliplr(b[i, :, :]).T, extent=[x[i]+x_mesh2.min(), x[i]+x_mesh2.max(), y[i]+y_mesh2.min(), y[i]+y_mesh2.max()],origin='lower', vmin=0.0, vmax=200)
            plt.autoscale(False)
        plt.colorbar()
        plt.autoscale(True)
        plt.title('b')
        plt.xlabel(r'')
        plt.ylabel(r'')
        #___________________________________________________________________________________
        plt.subplot(2, 4, 8)
        for i in range(im.shape[3]):
            plt.imshow(np.fliplr(chi[i, :, :]).T, extent=[x[i]+x_mesh2.min(), x[i]+x_mesh2.max(), y[i]+y_mesh2.min(), y[i]+y_mesh2.max()],origin='lower', vmin=0.0, vmax=1)
            plt.autoscale(False)
        plt.colorbar()
        
        plt.title(r'$\chi$')
        plt.xlabel(r'')
        plt.ylabel(r'')
        plt.autoscale(True)
        #___________________________________________________________________________________

    return out 


def save_figures(out):
    print('### Saving figures ###')
    figs = [f for f in plt.get_fignums()]
    os.makedirs(out['filename_output'][:-3], exist_ok=True)
    for i, fig in enumerate(figs):
        fig_name = plt.figure(fig).get_label().lower().replace(' ', '_')
        fig_path = os.path.join(out['filename_output'][:-3], f"fig_{fig}_{fig_name}.png")
        plt.savefig(fig_path)
        print(f"Figure {fig} {plt.figure(fig).get_label()} saved at {fig_path}")



# SAVE_PHASE_SLM save data into an HDF5 file
# So far, the phase was treated in units of PI. Meaning a value of 1
# means PI. The data saved into the file uses the phase in radians.

def save_phase_slm(data, param, out):
   
    phase_slm = out["phase_slm"]
    filenamePrefix = param['save']['filenamePrefix']
    filename_output = param['save']['dirname'] + filenamePrefix + " src_" + data['filename']
    out['filename_output'] = filename_output
    
    if param['save']['on']:
        if os.path.isfile(filename_output):
            os.remove(filename_output)

        # save segmentation result for slm
        print("### Saving phase for SLM ###")
        with h5py.File(filename_output, "w") as f:
            f.create_dataset("/phase_slm", data=phase_slm * np.pi)

            # save evaluation parameters
            param_names = param.keys()
            dset = f.create_dataset("/parameters", (1, 1))

            print("### Saving parameters...###")
            for param_name in param_names:
                print(param_name)
                subparam = param[param_name]

                try:
                    subparam_names = subparam.keys()

                    for subparam_name in subparam_names:
                        sub2param = subparam[subparam_name]

                        try:
                            sub2param_names = sub2param.keys()

                            for sub2param_name in sub2param_names:
                                sub3param = sub2param[sub2param_name]
                                print("!!! Struct above level 3 not saved !!!")
                        except AttributeError:
                            f["/parameters"].attrs.create(
                                param_name + "." + subparam_name, sub2param
                            )

                except AttributeError:
                    f["/parameters"].attrs.create(param_name, subparam)

        print("### Saving done ###")




# SLM_INTERPOLATION uses the phase in each pupil to interpolate the phase value in each SLM pixel
# input:
#   data['p']: containing SLM parameters (e.g. pixels number)
#   data['x']/data['y']: center coordinate of each pupil on the SLM
#   param['interpolation']
#           .method 'zernike','poly','scattered'
#           .zernike.FitOrder: polynomial order for zernike fit
#           .poly.FitOrder: string containing fit method (e.g.'poly55')
#           .cut.on radial constant value extrapolation of phase values
#                   outside the region defined by 
#           .cut.Factor  
#           .orientation['flipxy']: %%flip or transpose coordinates for SLM interpolation
#           .orientation['transpose']
#           .orientation['flipx']
#           .orientation['flipy']
#   out.phase_zero_tilt: unwrapped and tilt-subtracted phase values of pupils
# output: 
#   out['phase_slm']: 2D array containg the phase values (unit pi) to upload to SLM

def slm_interpolation(data,param,out):
    x=data['x']
    y=data['y']

    match param['interpolation']['phase_choice']:
        case 'direct':
            phase_choice=out['phase_average']
        case 'zero_tilt':
            phase_choice=out['phase_zero_tilt']
        case 'zero_defocus':
            phase_choice=out['phase_zero_defocus']
    
    seg_rng = param['seg_rng']
    p = data['p']


    #problem accesising it, so in the else beam_index instead of 1 a 5 is used as that would be the value obtained from the if
   # if 'Beam_to_use' in p['slm_pupil_segmentation']:
    #    beam_index = p['slm_pupil_segmentation']['Beam_to_use'] + 1
    #else:
    beam_index = 5
    
    outer_diameter_length = 24 #param['interpolation']['extend']['OuterDiameterLength']
    outer_diameter_extension = 1.3 #param['interpolation']['extend']['OuterDiameterExtension']
    fitOrder = 3 #param['interpolation']['FitOrder']
    cutFactor = 0.01 #param['interpolation']['cut']['Factor']
    
    slm_size = [480,576] #np.array(p['slm_pupil_segmentation']['size_template'][0:2], dtype=np.float64)
    beam_center = [-9,-38] #p['slm_control']['beams'][beam_index]['Beam_position__pix_']
    
 #___________________________________________________________________________________
 # slm meshgrid
    if param['interpolation']['orientation']['invertcenterx']:
        x_slm = np.arange(1, slm_size[0]+1) - slm_size[0]/2 + beam_center[0]
    else:
        x_slm = np.arange(1, slm_size[0]+1) - slm_size[0]/2 - beam_center[0]

    if param['interpolation']['orientation']['invertcentery']:
        y_slm = np.arange(1, slm_size[1]+1) - slm_size[1]/2 + beam_center[1]
    else:
        y_slm = np.arange(1, slm_size[1]+1) - slm_size[1]/2 - beam_center[1]

 #___________________________________________________________________________________ 
 # flip coordinates
    if param['interpolation']['orientation']['flipxy']:
        X_slm, Y_slm = np.meshgrid(y_slm, x_slm)
    else:
        X_slm, Y_slm = np.meshgrid(x_slm, y_slm)
        X_slm=X_slm.T
        Y_slm=Y_slm.T


    if param['interpolation']['orientation']['transpose']:
        X_slm = X_slm.T
        Y_slm = Y_slm.T

    if param['interpolation']['orientation']['flipx']:
        X_slm = np.fliplr(X_slm)
        Y_slm = np.fliplr(Y_slm)

    if param['interpolation']['orientation']['flipy']:
        X_slm = np.flipud(X_slm)
        Y_slm = np.flipud(Y_slm)

    print('### Extending region for interpolation ###')

 #___________________________________________________________________________________
 # extend circle of phase values with constant extrapolation
    match param['interpolation']['extend']['method']:
        case 'direct':
            x_int = x[seg_rng]
            y_int = y[seg_rng]
            phase_int = phase_choice[seg_rng]
        case 'repeat outer (last)':
            #outer diameter at the end
            x_int = np.concatenate((x[seg_rng], x[seg_rng[-outer_diameter_length:]] * outer_diameter_extension))
            y_int = np.concatenate((y[seg_rng], y[seg_rng[-outer_diameter_length:]] * outer_diameter_extension))
            phase_int = np.concatenate((phase_choice[seg_rng], phase_choice[seg_rng[-outer_diameter_length:]]))
        case 'repeat outer (first)':
            fit_rng = np.concatenate((seg_rng, seg_rng[:outer_diameter_length]))
            # Outer diameter in the beginning
            x_int = np.concatenate((x[seg_rng], x[seg_rng[:outer_diameter_length]] * outer_diameter_extension))
            y_int = np.concatenate((y[seg_rng], y[seg_rng[:outer_diameter_length]] * outer_diameter_extension))
            phase_int = np.concatenate((phase_choice[seg_rng], phase_choice[seg_rng[:outer_diameter_length]]))
    
 #___________________________________________________________________________________
 # Make minimum value zero to avoid phase wrapping
    if param['interpolation']['zerominimum']:
        phase_int -= np.min(phase_int)

 #___________________________________________________________________________________ 
    # interpolation
    print('### Interpolating ###')

    th_int, r_int = np.arctan2(y_int, x_int), np.hypot(x_int, y_int)
    Th_slm, R_slm = np.arctan2(Y_slm, X_slm), np.hypot(X_slm, Y_slm)

    match param['interpolation']['method']:
        case 'zernike':
            r_ref = np.max(r_int)
            # Zernike polynomial fitting
            z = zernikeFittype(fitOrder)
            p0 = np.zeros((fitOrder+1)*(fitOrder+2)//2)
            popt, pcov = curve_fit(z, (r_int/r_ref, th_int), phase_int, p0=p0)
            phase_fit = z((r_int/r_ref, th_int), *popt)
            phase_slm = z((R_slm.ravel()/r_ref, Th_slm.ravel()), *popt).reshape(R_slm.shape)
            out.qof = np.sum((phase_fit-phase_int)**2)/len(phase_int)
        case 'poly':
            fitfunction = poly_fitfunction_from_order(fitOrder)
            # standard polynomial fitting
            phase_fit = fitfunction.fit(np.stack((x_int.ravel(), y_int.ravel()), axis=1), phase_int.ravel())
            phase_slm = phase_fit(np.stack((X_slm.ravel(), Y_slm.ravel()), axis=1)).reshape(X_slm.shape)
        case 'scattered':
            phase_fit = interp2d(x_int, y_int, phase_int, kind='nearest')
            phase_slm = phase_fit(X_slm[0], Y_slm[:,0])
        case 'thin_plate_smoothing_spline':
            phase_spline = SmoothBivariateSpline(x_int.ravel(), y_int.ravel(), phase_int.ravel(), s=param['interpolation']['smoothing'])
            phase_slm = phase_spline.ev(X_slm, Y_slm)

    print('### Extrapolating for whole SLM ###')

 #___________________________________________________________________________________
 # Set phase values outside of circular region of interest to constant value
    if param['interpolation']['cut']['on']:
        filter_ring = (R_slm < (1+cutFactor)*np.max(r_int)) & \
                  (R_slm > (1-cutFactor)*np.max(r_int))
        filter_outer = R_slm >= (1+cutFactor)*np.max(r_int)
    
        if param['interpolation']['cut']['extrapolate']:
            coords = np.column_stack((X_slm[filter_ring], Y_slm[filter_ring]))
            values = phase_slm[filter_ring]
            phase_extrapolation_constructor = \
                NearestNDInterpolator(coords, values)
            phase_extrapolation = \
                phase_extrapolation_constructor(np.column_stack((X_slm, Y_slm)))
            phase_slm[filter_outer] = phase_extrapolation[filter_outer]
        else:
            phase_slm[filter_outer] = 0

    out['phase_slm'] = phase_slm

    print('### Plotting result ###')

 #___________________________________________________________________________________
 # plot final results
    x_mesh_surf = np.linspace(np.min(x_int), np.max(x_int), 100)
    y_mesh_surf = np.linspace(np.min(y_int), np.max(y_int), 100)
    X_mesh_surf, Y_mesh_surf = np.meshgrid(x_mesh_surf, y_mesh_surf)
    X_mesh_surf= X_mesh_surf.T
    Y_mesh_surf =  Y_mesh_surf.T

 #___________________________________________________________________________________ 
    plt.figure(7)
    fig=plt.figure(7,figsize=(param['figure_position'][2]/100,param['figure_position'][3]/100),dpi=100)
    fig.suptitle('Final phase retrieval result')
    plt.clf()
 #___________________________________________________________________________________
    plt.subplot(2,3,1)
    spot_size = 200
    plt.scatter(x[seg_rng], y[seg_rng], spot_size, out['phase_average'][seg_rng],vmin=out['phase_average'][seg_rng].min(),vmax=out['phase_average'][seg_rng].max())
    plt.colorbar()
    plt.title('Retrieved phase structure')

 #___________________________________________________________________________________ 
    plt.subplot(2,3,2)
    plt.scatter(x[seg_rng], y[seg_rng], spot_size, out['phase_zero_tilt'][seg_rng],vmin=out['phase_zero_tilt'][seg_rng].min(),vmax=out['phase_zero_tilt'][seg_rng].max())
    plt.colorbar()
    plt.title('Unwrapped without tilt')
    
 #___________________________________________________________________________________
    plt.subplot(2,3,3)
    plt.scatter(x[seg_rng], y[seg_rng], spot_size, out['phase_zero_defocus'][seg_rng], vmin=out['phase_zero_defocus'][seg_rng].min(),vmax=out['phase_zero_defocus'][seg_rng].max())
    plt.colorbar()
    plt.title('Unwrapped without defocusing')

 #___________________________________________________________________________________ 
    plt.subplot(2,2,3)
    plt.imshow(phase_slm.T, extent=(x_slm.min(), x_slm.max(), y_slm.min(), y_slm.max()), vmin=phase_int.min(), vmax=phase_int.max())
    plt.colorbar()
    plt.plot(x_int, y_int, '.w')
    plt.plot(beam_center[1], beam_center[0], 'x', markersize=5, linewidth=2)
    plt.title('SLM correction')
    

 #___________________________________________________________________________________
    match param['interpolation']['method']:
        case 'zernike':
            Th_mesh_surf, R_mesh_surf = np.cart2pol(X_mesh_surf, Y_mesh_surf)

            phase_fit_mesh = phase_fit(R_mesh_surf.ravel()/r_ref, Th_mesh_surf.ravel()).reshape(R_mesh_surf.shape)

            phase_fit_at_data_points = np.zeros(len(phase_int))
            for i in range(len(phase_int)):
                phase_fit_at_data_points[i] = phase_fit(r_int[i]/r_ref, th_int[i])

        case 'thin_plate_smoothing_spline':
            phase_fit_mesh = phase_spline(x_mesh_surf, y_mesh_surf)
            phase_fit_at_data_points = []
            for i in range(len(phase_int)):
                phase_fit_at_data_points.append(phase_spline.ev(x_int[i], y_int[i]))
            
        case _:
            phase_fit_mesh = phase_fit(X_mesh_surf, Y_mesh_surf)
            phase_fit_at_data_points = []
            for i in range(len(phase_int)):
                phase_fit_at_data_points.append(phase_fit(x_int[i], y_int[i]))
    
    
    ax=plt.subplot(2,2,4, projection='3d')
    surf=ax.plot_surface(X_mesh_surf, Y_mesh_surf, phase_fit_mesh,cmap='viridis', alpha=0.35)
    ax.view_init(elev=10, azim=225)
    
  
    ax.contour(X_mesh_surf, Y_mesh_surf, phase_fit_mesh, levels=np.arange(-2, 2.1, 0.1), colors='k')
    points=ax.scatter(x_int, y_int,phase_int, s=50)
    points.set_facecolor(plt.cm.viridis(phase_int))

    for i in range(len(phase_int)):
        ax.plot([x_int[i], x_int[i]], [y_int[i], y_int[i]], [phase_fit_at_data_points[i], phase_int[i]], color=[0.7, 0.7, 0.7])

    plt.ylim([np.min(Y_mesh_surf), np.max(Y_mesh_surf)])
    plt.xlim([np.min(X_mesh_surf), np.max(X_mesh_surf)])
    ax.set_zlim([np.min(phase_int), np.max(phase_int)])
    plt.clim([np.min(phase_int), np.max(phase_int)])
    plt.title('Interpolation')
   plt.show()
 #___________________________________________________________________________________
    return out

# UNWRAP_ALL_PUPILS unwraps the phases over all pupils and subtracts a remaining linear term
# input: 
#   data.x,data.y: coordinates of pupils
#   out.phase_average: vector containing resulting phase value in each pupil
#   param.seg_rng: pupils to use
# output: 
#   out.phase_zero_tilt: phases at pupil coordinates x,y

def unwrap_all_pupils(data, param, out):

    print('### Unwrapping and subtracting linear term ###')

    #unwrap recovered phase over pupils and substract linear term
    x = data['x']
    y = data['y']
    phase_average = out['phase_average']
    seg_rng = param['seg_rng']
    unwrapMethod = param['unwrap']['method']
    interp_method = param['unwrap']['interpMethod']

    if unwrapMethod == '1D':
        phase_average = np.unwrap(phase_average*np.pi)/np.pi
    elif unwrapMethod == '2D':
        step = np.floor(np.min([np.min(np.abs(np.diff(x))), np.min(np.abs(np.diff(y.flatten())))]))
        gridx = np.arange(np.floor(np.min(x)), np.ceil(np.max(x))+step, step)
        gridy = np.arange(np.floor(np.min(y)), np.ceil(np.max(y))+step, step)
        phases = np.full((len(gridy), len(gridx)), np.nan)

        for ii in range(len(seg_rng)):
            rowx = np.where(np.abs(gridx-x[seg_rng[ii]])<=step/2)[0][0]
            coly = np.where(np.abs(gridy-y[seg_rng[ii]])<=step/2)[0][0]
            phases[coly, rowx] = phase_average[seg_rng[ii]]
        
        gridX, gridY = np.meshgrid(gridx, gridy)
        mask = ~np.isnan(phases)
        phases_cart = griddata((gridX[mask], gridY[mask]), phases[mask], (gridX, gridY), method=interp_method)
        phases_new = np.unwrap(phases_cart*np.pi, axis=0, period=2*np.pi)/np.pi

        for ii, idx in enumerate(seg_rng):
            phase_average[idx] = phases_new[rowx[ii], coly[ii]]

    elif unwrapMethod == 'none':
        pass
    
    else:
        pass

    # Remove tilt    
    def poly2(x, a, b, c):
      return a + b*x[0] + c*x[1]**2

    def phase_seg_fit2(x, y, z):
        popt, _ = curve_fit(poly2, (x, y), z)
        return poly2((x, y), *popt)

    phase_seg_fit = phase_seg_fit2(x[seg_rng], y[seg_rng], phase_average[seg_rng])
    phase_zero_tilt = phase_average - phase_seg_fit

    # Remove defocus
    X = np.vstack((x[seg_rng], y[seg_rng])).T
    p0 = np.zeros(3)
    (A, B, C), _ = leastsq(lambda p, X, y: y - (p[0] + p[1] * X[:, 0]**2 + p[2] * X[:, 1]**2),
                            p0, args=(X, phase_average[seg_rng]))
    phase_zero_defocus = phase_zero_tilt - (A + B * x**2 + C * y**2)


    
    out['phase_zero_tilt'] = phase_zero_tilt
    out['phase_zero_defocus'] = phase_zero_defocus

    return out