

#importing required packages
from numpy import * 
import numpy as np
from matplotlib.pyplot import *
import random
import scipy.ndimage
import matplotlib.pyplot as plt


#%% user parameters

NN = 16#side length of the SLM in pixels; the number of correctable modes is thus N x N
N = 32 #number of scattered modes = N^2

iter = 10 #no. of iterations, i.e. full mode cycles

eta = 100 #efficiency of 2-photon signal generation; vary e.g. between 1 and 100

sampletype = 'gaussian'  #sample can be a point source: "bead" or a 2D fluorescent plane: "layer" adding 'gaussian' option

trials = 5 #run multiple trials to find standard deviation
all_data = zeros([NN**2*iter, trials])
    
#%% calculating plane-wave modes / definition of required functions

N_modes = NN*NN
M = np.zeros((N, N, N_modes)) #initializing array of modes
k = fft.fftfreq(NN, 1/NN)
Kx, Ky = np.meshgrid(k,k) #pixel coordinates

mm = 0

#Square grid
#for m in range(NN):      
 #   for n in range(NN):            
  #      gx = -pi + (n+1)*2*pi/NN #grating vector in x
   #     gy = -pi + (m+1)*2*pi/NN #grating vector in x
    #    M[:,:,mm] = np.resize((gx*Kx + gy*Ky), (N,N))
     #   mm += 1


radius_values = np.linspace(0, NN/2, NN)  # Equidistant radius values
#polar grid
for m in range(NN):
    for n in range(NN):
        thet = (2 * np.pi * m) / NN  # Angle (thet) in polar coordinates
        r = radius_values[n]  # Radius (r) in polar coordinates (equidistant)
        gx = r * np.cos(thet)  # Grating vector in x
        gy = r * np.sin(thet)  # Grating vector in y

        M[:, :, mm] = np.resize((gx * Kx + gy * Ky), (N, N))
        mm += 1


def TPEF(E_SLM, scat, sample): #two photon fluorescence signal changed to one photon changing **4 to **2
    "calculation of two photon signal based on pupil field"
    I2ph = int(eta*1e3*sum(np.abs(sample * (np.fft.fft2(scat * E_SLM)/N/N))**2))
    return np.random.poisson(I2ph)


def gaussian(N,sigma,center_x, center_y):  # Create gaussian profile
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)    
    gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
   
    return gaussian

#%% EXECUTE OPTIMIZATION
for idx in range(trials):

    scat = np.exp(1j*2*pi*np.random.rand(N,N)) #create a random phase scatterer
    phase_smoothed = scipy.ndimage.uniform_filter(np.angle(scat), size=3)  #apply a low-pass filer to the phase distribution, size is used to determine local neighborhood
    scat = np.abs(scat) * np.exp(1j * phase_smoothed)
   

    theta = array([0, 2*pi/3, 4*pi/3]) #reference phase values
    f = 0.3 #energy fraction going into the testing M
    
    #initializations
    I2ph = zeros((3))
    a = zeros(N_modes, dtype = "complex") #initializing mode correction amplitudes
    C = zeros((N,N), dtype = "complex") #initializing correction mask
    signals = zeros((iter, N_modes))
    
    if sampletype == 'layer':
        sample = np.ones((N,N)) #fluorescent plane
    elif sampletype == 'bead':
        sample = zeros((N,N)); sample[0,0]=1 #single beacon
    elif sampletype == 'gaussian':
        sample = np.zeros((N, N))
        numsample=4
        gaussian_sigma=0.08
        for _ in range(numsample):
            center_x = np.random.randint(0, N)
            center_y = np.random.randint(0, N)
            gaussian_profile = gaussian(N, gaussian_sigma, center_x, center_y)  # Create a single Gaussian profile at random position
            sample += gaussian_profile
 
 
    for i in range(iter):
        
        print(iter-i)
        
        for m in range(N_modes): #mode-stepping loop
             
            for p in range(size(theta)): #phase-stepping loop

                E_SLM = exp(1j * angle(sqrt(f) * exp(1j*M[:,:,m] + 1j*theta[p]) + sqrt(1-f) * exp(1j*angle(C)))) #in DASH, the two beams are created by a single phase-only SLM  
                I2ph[p] = TPEF(E_SLM, scat, sample) #get 2-photon signal
            
            signals[i, m] = np.mean(I2ph) #mean signal over all phase steps
            a[m] = np.sum(sqrt(I2ph) * exp(+1j*theta)) / size(theta) #retrieving a = |a|*exp(1j*phi_m), we multiply with exp(+1j*theta) instead of exp(-1j*theta), because we want to directly calculate the correction phase )
                
            C += a[m] * exp(1j*M[:,:,m]) #for DASH, immediately update the correction mask
            

    all_data[:,idx] = ravel(signals)

#-------------- display results------------------



subplot(2, 2, 1)
imshow(angle(scat), cmap='hsv',origin='lower')
colorbar(label='Phase (radians)')
title('Original phase')


subplot(2, 2, 2)
imshow(angle(E_SLM) ,cmap='hsv',origin='lower')
colorbar(label='Phase (radians)')
title('Corrected wavefront phase')

phase_diff=angle(E_SLM)-angle(scat)

#Root mean square error (RMSE)
rmse = np.sqrt(np.mean(phase_diff**2))
print("RMSE Error:", rmse)

#Needs mean quadratic error or sum of error, evolution of error as function of used modes

subplot(2, 2, 3)
imshow(phase_diff, cmap='hsv',origin='lower')
colorbar(label='Phase (radians)')
title('Corrected - original')

#display singal trend
plt.subplot(2, 2, 4)
plot(mean(all_data, axis = 1))
xlabel('measurement no.')
ylabel('Signal / photons')
grid(1)        
title( "DASH-" + str(N_modes) + " modes")
show()

#display focus
# =============================================================================
# I_scat = np.abs(fft.ifftshift(fft.fft2(scat)))**2
# I_corr = np.abs(fft.ifftshift(fft.fft2(scat * exp(1j*angle(C)))))**2
# 
# figure(2)
# imshow(I_scat, cmap = "jet")
# title("scattered focus irradiance")
# colorbar()
# 
# figure(3)
# imshow(I_corr, cmap = "jet")
# title("corrected focus irradiance")
# colorbar()
# =============================================================================

print("minimum / maximum signals per measurement: " + str(np.int(np.min(signals))) + " / " + str(np.int(np.max(signals))) + " photons")

