#importing required packages
from numpy import * 
import numpy as np
from matplotlib.pyplot import *
import random
import math
import scipy.ndimage
import matplotlib.pyplot as plt


#%% user parameters


#In the scattering regime the complexity of the aberration is beyond the capabilities of the correction device, meaning that the number of scattering modes is larger than the number of correctable ones.
NN = 16#side length of the SLM in pixels; the number of correctable modes is thus N x N
N = 16#number of scattered modes = N^2

iter = 10#no. of iterations, i.e. full mode cycles

eta = 100 #efficiency of signal generation; vary e.g. between 1 and 100

sampletype = 'bead'  #sample can be a point source: "bead" or a 2D fluorescent plane: "layer" adding 'gaussian'
gridtype = 'square'  #grid can be square, polar0 (not evenly spaced polar), polar1 (evenly spaced polar) or hexagonal

def polargrid(N):
    max_radius = N  # Maximum radius for the points
    num_lines = N + 1  # Number of lines

    # Plot the polar grid
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Create empty lists to store the radius and angle of points
    radius_values = []
    angle_values = []

    # Calculate the angle difference between each line
    angle_diff = 180 / num_lines

    # Calculate the total length of all lines
    total_length = 0

    # Calculate line lengths and store start and end points
    start_points = []
    end_points = []

    for i in range(num_lines):
        line_angle = i * angle_diff

        # Calculate the start and end angles of the line
        start_angle = 180 - line_angle
        end_angle = line_angle + 180

        # Calculate the start and end points of the line
        start_radius = max_radius
        end_radius = max_radius

        start_x = start_radius * np.cos(np.deg2rad(start_angle))
        start_y = start_radius * np.sin(np.deg2rad(start_angle))
        end_x = end_radius * np.cos(np.deg2rad(end_angle))
        end_y = end_radius * np.sin(np.deg2rad(end_angle))

        # Store the start and end points
        radius_values.append(max_radius)
        angle_values.append(np.deg2rad(start_angle))

        if i == 0:
            radius_values.append(max_radius)
            angle_values.append(0)

        if i > 0:
            radius_values.append(max_radius)
            angle_values.append(np.deg2rad(end_angle))

        start_points.append((start_x, start_y))
        end_points.append((end_x, end_y))

        # Calculate the line length
        line_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

        # Add the line length to the total length
        total_length += line_length

    # Draw lines and distribute points along each line
    for i in range(num_lines):
        start_x, start_y = start_points[i]
        end_x, end_y = end_points[i]

        # Plot the start and end points
        ax.scatter(np.deg2rad(180 - i * angle_diff), max_radius, color='red', s=2)
        if i == 0:
            ax.scatter(0, max_radius, color='red', s=2)
            
        if i > 0:
            ax.scatter(np.deg2rad(i * angle_diff + 180), max_radius, color='red', s=2)

            # Calculate the line length
            line_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

            # Calculate the number of points based on line length
            num_points = int(line_length / total_length * (N * N - 2 * N - 1))

            # Calculate and plot the weighted distribution of points along the line
            for j in range(1, num_points + 1):
                # Calculate the parameter t for the current point
                t = j / (num_points + 1)

                # Calculate the coordinates of the point
                point_x = start_x + t * (end_x - start_x)
                point_y = start_y + t * (end_y - start_y)

                # Calculate the radius and angle of the point
                point_radius = np.sqrt(point_x ** 2 + point_y ** 2)
                point_angle = np.arctan2(point_y, point_x)

                # Store the radius and angle values
                radius_values.append(point_radius)
                angle_values.append(point_angle)

                # Plot the point
                ax.scatter(point_angle, point_radius, color='red', s=2)

    if len(radius_values) != N * N:
        remaining_points = N * N - len(radius_values)
        if remaining_points == 1:
            radius_values.append(0)
            angle_values.append(0)
            ax.scatter(0, 0, color='red', s=2)
        elif remaining_points == 2:
            radius_values.append(max_radius)
            angle_values.append(np.deg2rad(90))
            radius_values.append(max_radius)
            angle_values.append(np.deg2rad(270))
            ax.scatter(np.deg2rad(90), max_radius, color='red', s=2)
            ax.scatter(np.deg2rad(270), max_radius, color='red', s=2)
        else:
            radius_values.append(max_radius)
            angle_values.append(np.deg2rad(90))
            radius_values.append(max_radius)
            angle_values.append(np.deg2rad(270))
            ax.scatter(np.deg2rad(90), max_radius, color='red', s=2)
            ax.scatter(np.deg2rad(270), max_radius, color='red', s=2)
            start_radius = max_radius
            end_radius = max_radius
            start_angle = 90
            end_angle = 270

            start_x = start_radius * np.cos(np.deg2rad(start_angle))
            start_y = start_radius * np.sin(np.deg2rad(start_angle))
            end_x = end_radius * np.cos(np.deg2rad(end_angle))
            end_y = end_radius * np.sin(np.deg2rad(end_angle))
            for j in range(remaining_points - 2):
                # Calculate the parameter t for the current point
                t = j / (remaining_points - 2)

                # Calculate the coordinates of the point
                point_x = start_x + t * (end_x - start_x)
                point_y = start_y + t * (end_y - start_y)

                # Calculate the radius and angle of the point
                point_radius = np.sqrt(point_x ** 2 + point_y ** 2)
                point_angle = np.arctan2(point_y, point_x)

                # Store the radius and angle values
                radius_values.append(point_radius)
                angle_values.append(point_angle)

                # Plot the point
                ax.scatter(point_angle, point_radius, color='red', s=4)

    # Set the limits and labels
    ax.set_ylim([0, max_radius + 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)

    # Add title and angle values
    ax.set_title('Polar Grid')
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['0', '90', '180', '270'])

    return np.array(radius_values), np.array(angle_values)

trials = 1 #run multiple trials to find standard deviation
all_data = zeros([NN**2*iter, trials])
    
#%% calculating plane-wave modes / definition of required functions

N_modes = NN*NN
mm = 0 
M = np.zeros((N, N, N_modes)) #initializing array of modes

if gridtype=='square':

    k = fft.fftfreq(NN, 1/NN)
    Kx, Ky = meshgrid(k,k) #pixel coordinates
    for m in range(NN):      
       for n in range(NN):            
          gx = -pi + (n+1)*2*pi/NN #grating vector in x
          gy = -pi + (m+1)*2*pi/NN #grating vector in x
          M[:,:,mm] = np.resize((gx*Kx + gy*Ky), (N,N))
          mm += 1
       
elif gridtype=='polar0':

    k = fft.fftfreq(NN, 1 / NN)
    Kx, Ky = np.meshgrid(k, k)  # Pixel coordinates

    # Convert to polar coordinates
    Kr = np.sqrt(Kx ** 2 + Ky ** 2)
    Ktheta = np.arctan2(Ky, Kx)
    radius_values = radius_values = np.linspace(0, NN/2 , NN) 
    for m in range(NN):
        for n in range(NN):
            thet = (2 * pi * m) / NN  # Angle (thet) in polar coordinates
            r = radius_values[n]  # Radius (r) in polar coordinates (equidistant)
            #gx = r * cos(thet)  # Grating vector in x
            #gy = r * sin(thet)  # Grating vector in y
            M[:, :, mm] = resize((r * Kr + thet * Ktheta), (N, N))
            mm += 1

elif gridtype=='polar1':
    k = fft.fftfreq(NN, 1 / NN)
    Kx, Ky = np.meshgrid(k, k)  # Pixel coordinates

    # Convert to polar coordinates
    Kr = np.sqrt(Kx ** 2 + Ky ** 2)
    Ktheta = np.arctan2(Ky, Kx)

    r_values,a_values=polargrid(NN)
    sorted_indices = np.argsort(r_values)
    r_values = r_values[sorted_indices]
    a_values = a_values[sorted_indices]

    for n in range(NN*NN):
        thet = a_values[n]  # Angle (thet) in polar coordinates
        r = r_values[n]  # Radius (r) in polar coordinates (equidistant)
        #gx = r * cos(thet)  # Grating vector in x
        #gy = r * sin(thet)  # Grating vector in y
        M[:, :, mm] = resize((r * Kr + thet * Ktheta), (N, N))
        mm += 1

elif gridtype=='hexagonal':
    k = fft.fftfreq(NN, 1/NN)
    Kx, Ky = meshgrid(k,k) #pixel coordinates
    for m in range(NN):
        for n in range(NN):
            x = n * np.sqrt(3)
            y = m * 1.5
            gx = x * pi / NN
            gy = y * pi / NN
            M[:, :, mm] = gx * Kx + gy * Ky
            mm += 1


def gaussian(N,sigma,center_x, center_y):  # Create gaussian profile
    x = arange(N)
    y = arange(N)
    X, Y = meshgrid(x, y)    
    gaussian_profile = exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
   
    return gaussian_profile



#signal on the pupil field,  applying the sample and scattered phase distributions to the electric field on the SLM
def Intensity(E_SLM, scat, sample): #fluorescence signal changed to one photon  **4 to **2
    "calculation signal based on pupil field"
    I = int(eta*1e3*sum(np.abs(sample * (np.fft.fft2(scat * E_SLM)/N/N))**2))
    return np.random.poisson(I)



#%% EXECUTE OPTIMIZATION
#a random phase mask that emulates scattering medium in the light path, is assumed, which is optically in the same plane ("conjugate") as the correction device (SLM)
scat = np.exp(1j*2*pi*np.random.rand(N,N)) #create a random phase scatterer
phase_smoothed = scipy.ndimage.uniform_filter(angle(scat), size=8)  #apply a low-pass filer to the phase distribution, size is used to determine local neighborhood
scat = abs(scat) * exp(1j * phase_smoothed)

if sampletype == 'layer':
    sample = np.ones((N,N)) #fluorescent plane
elif sampletype == 'bead':
    sample = zeros((N,N)); sample[int(N/2),int(N/2)]=1 #single beacon
elif sampletype == 'gaussian':
    sample = np.zeros((N, N))
    numsample=15
    gaussian_sigma=0.08
    for _ in range(numsample):
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, N)
        gaussian_profile = gaussian(N, gaussian_sigma, center_x, center_y)  # Create a single Gaussian profile at random position
        sample += gaussian_profile


for idx in range(trials):

    P= 5 #decide numnber of phase steps to be considered
    theta=zeros((P))
    for p in range (P):
        theta[p] = p*2*pi/P

    f = 0.35 #energy fraction going into the testing M
    
    #initializations
    I = zeros((P))
    a = zeros(N_modes, dtype = "complex") #initializing mode correction amplitudes
    C = zeros((N,N), dtype = "complex") #initializing correction mask
    signals = zeros((iter, N_modes))
    
    for i in range(iter):
        
        print(iter-i)
        
        for m in range(N_modes): #mode-stepping loop
            best_signal = 0  # Variable to store the best signal value found
            best_amplitude = 0 
            best_angle=0
            for p in range(size(theta)): #phase-stepping loop
                E_SLM = exp(1j * angle(sqrt(f) * exp(1j*M[:,:,m] + 1j*theta[p]) + sqrt(1-f) * exp(1j*angle(C)))) #in DASH, the two beams are created by a single phase-only SLM , this represents the phase mask
                I[p] = Intensity(E_SLM, scat, sample) #get signal

            signals[i, m] = np.mean(I) #mean signal over all phase steps
          
            
            a[m] = np.sum(sqrt(I) * exp(+1j*theta)) / size(theta) #retrieving a = |a|*exp(1j*phi_m), we multiply with exp(+1j*theta) instead of exp(-1j*theta), because we want to directly calculate the correction phase )    
            C += a[m] * exp(1j*M[:,:,m]) #for DASH, immediately update the correction mask

            #new procedure, normalization of corrected beam, which erases the amplitude information from the corrected field C is replaced by a different scalarn ormalization factor, which can lead to slightly better performance.
            norm=np.sqrt((1/N_modes*N_modes)*np.sum(abs(C) ** 2))    #calculate normalization factor
            C=C/norm
            
    all_data[:,idx] = ravel(signals)

#-------------- display results------------------


subplot(2, 2, 1)
imshow(angle(scat), cmap='hsv',origin='lower')
colorbar(label='Phase (radians)')
title('Original phase')

corrected_wavefront =C

subplot(2, 2, 2)
imshow(angle(corrected_wavefront) ,cmap='hsv',origin='lower')
colorbar(label='Phase (radians)')
title('Corrected wavefront')


#displaying the retrieved phase compensation patterns on top of the given scattering mask, we can test and compare the algorithm performance
phase_diff=angle(scat)-angle(corrected_wavefront)

#Root mean square error (RMSE)
rmse = sqrt(mean(phase_diff**2))
print("RMSE Error:", rmse)

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


#num_rows = int(np.ceil(np.sqrt(mm)))
#num_cols = int(np.ceil(mm / num_rows))

# Create subplots for each value of M
#fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# Flatten the axs array if necessary
#if isinstance(axs, np.ndarray):
    #axs = axs.flatten()

# Plot each value of M in a separate subplot
#for m in range(mm):
    #axs[m].imshow(M[:, :, m], cmap='hsv', origin='lower')
    #axs[m].set_title(f"M[:,:,{m}]")
    #axs[m].axis('off')

# Adjust the layout of subplots
#plt.tight_layout()

#figure(3)
#imshow(M[:,:,100], cmap='hsv', origin='lower')
#colorbar()
# Show the plot


plt.show()


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

