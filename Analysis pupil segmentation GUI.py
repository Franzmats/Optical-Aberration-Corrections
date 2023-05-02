from functions import average_phase, fit_phase, generate_filenamePrefix, phase_unwrap_regular, pi,pupil_segmentation_init, read_pupil_segmentation,retrieve_phase,save_figures,save_phase_slm, slm_interpolation, unwrap_all_pupils
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

run_file = True
os.getcwd()

while run_file:
    # Clear all variables
    try:
        from IPython import get_ipython
        get_ipython().magic('reset -sf')
    except:
        pass
    
    # Read measurement data
    root = Tk()
    root.withdraw()
    # filename = askopenfilename(initialdir='Y:/projects/Calibration/mf3 slm', filetypes=[('HDF5 Files', '*.h5')])
    filename = askopenfilename(initialdir='Users\francisco.matos\Documents\SLM', filetypes=[('HDF5 Files', '*.h5')])
    dirname = os.path.dirname(filename)
    filename=os.path.basename(filename)
    root.destroy()

    data = read_pupil_segmentation(filename, dirname)
 
 
    # Define parameters
    param = dict()
    out = dict()
    param,out=pupil_segmentation_init(data) # set default parameters
    
    param['phase']['ref'][0] += 1
    param['phase']['ref'][1] += 1
    
    param['debug'] = 1
    param['phase']['mad'] = 4
    param['average']['method'] = 'single'
    param['average']['pixel'] = [10, 10]
    param['unwrap']['method'] = 'none'
    
    # Interpolation
    param['interpolation']['phase_choice'] = 'zero_defocus' # 'direct', 'zero_tilt', 'zero_defocus'
    
    param['interpolation']['method'] = 'thin_plate_smoothing_spline' # others: 'poly','scattered', 'zernike', 'thin-plate_smoothing_spline'
    param['interpolation']['FitOrder'] = 3
    param['interpolation']['smoothing'] = 10e-4 
    # param.interpolation.smoothing = 0.5; 
    
    # Extend region to fit
    param['interpolation']['extend.method'] = 'direct' #'direct'; # 'repeat outer (last)','repeat outer (first)'
    # param.interpolation.extend.method = 'repeat outer (first)';
    param['interpolation']['extend.OuterDiameterLength'] = 24
    param['interpolation']['extend.OuterDiameterExtension'] = 1.3
        
    # Avoid phase wrap
    param['interpolation']['zerominimum'] = 1
    
    # Cut beam region
    param['interpolation']['cut']['on'] = 0
    param['interpolation']['cut']['Factor'] = 0.01 # beam cut at (1+cutFactor)*R_max
    param['interpolation']['cut']['extrapolate'] = 1
    
    # Saving
    param['save']['on'] = 1
    
    param = generate_filenamePrefix(param)
    
    # Calculate phase for SLM
    out = retrieve_phase(data, param, out)
    out = fit_phase(data, param, out)
    out = average_phase(param, out)
    out = unwrap_all_pupils(data, param, out)
    out = slm_interpolation(data, param, out)
    out = save_phase_slm(data, param, out)
 
    save_figures(out)
    
    run_file = input('Another file? (y/n) ') == 'y'
