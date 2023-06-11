import numpy as np
import xmltodict
from scipy.fft import ifft
from scipy.optimize import curve_fit

def h5_LV_attributes_to_dict(h5_info_attrib, *args):
    # Turns a list of attributes of an HDF element into a data dictionary.
    # Each attribute is supposed to be a LabView control flattened to XML.

    # Obtain strcuture to append, if it is passed.
    if not args:
        attrib_out = {}
    else:
        attrib_out = args[0]

    for i in h5_info_attrib.attrs:
        attrib_out[i]=xmltodict.parse(h5_info_attrib.attrs[i])
    
    return attrib_out


def h5_LV_MF_format_extract_parameters(parameters_info):
    # Load all parameters from file with MF format.
    # Each VI parameters set is loaded into a field of the structure.
    p = dict()
    for group in parameters_info:
        p[group] = h5_LV_attributes_to_dict(parameters_info[group])
    return p


# ZERNFUN Zernike functions of order N and frequency M on the unit circle.
#   Z = ZERNFUN(N,M,R,THETA) returns the Zernike functions of order N
#   and angular frequency M, evaluated at positions (R,THETA) on the
#  unit circle.  N is a vector of positive integers (including 0), and
#   M is a vector with the same number of elements as N.  Each element
#   k of M must be a positive integer, with possible values M(k) = -N(k)
#   to +N(k) in steps of 2.  R is a vector of numbers between 0 and 1,
#   and THETA is a vector of angles.  R and THETA must have the same
#   length.  The output Z is a matrix with one column for every (N,M)
#   pair, and one row for every (R,THETA) pair.

#   Z = ZERNFUN(N,M,R,THETA,'norm') returns the normalized Zernike
#   functions.  The normalization factor sqrt((2-delta(m,0))*(n+1)/pi),
#   with delta(m,0) the Kronecker delta, is chosen so that the integral
#   of (r * [Znm(r,theta)]^2) over the unit circle (from r=0 to r=1,
#   and theta=0 to theta=2*pi) is unity.  For the non-normalized
#   polynomials, max(Znm(r=1,theta))=1 for all [n,m].
#
#   The Zernike functions are an orthogonal basis on the unit circle.
#   They are used in disciplines such as astronomy, optics, and
#   optometry to describe functions on a circular domain.
#
#   The following table lists the first 15 Zernike functions.
#
#       n    m    Zernike function             Normalization
#       ----------------------------------------------------
#       0    0    1                              1/sqrt(pi)
#       1    1    r * cos(theta)                 2/sqrt(pi)
#       1   -1    r * sin(theta)                 2/sqrt(pi)
#       2    2    r^2 * cos(2*theta)             sqrt(6/pi)
#       2    0    (2*r^2 - 1)                    sqrt(3/pi)
#       2   -2    r^2 * sin(2*theta)             sqrt(6/pi)
#       3    3    r^3 * cos(3*theta)             sqrt(8/pi)
#       3    1    (3*r^3 - 2*r) * cos(theta)     sqrt(8/pi)
#       3   -1    (3*r^3 - 2*r) * sin(theta)     sqrt(8/pi)
#       3   -3    r^3 * sin(3*theta)             sqrt(8/pi)
#       4    4    r^4 * cos(4*theta)             sqrt(10/pi)
#       4    2    (4*r^4 - 3*r^2) * cos(2*theta) sqrt(10/pi)
#       4    0    6*r^4 - 6*r^2 + 1              sqrt(5/pi)
#       4   -2    (4*r^4 - 3*r^2) * sin(2*theta) sqrt(10/pi)
#       4   -4    r^4 * sin(4*theta)             sqrt(10/pi)
#       ----------------------------------------------------

def zernfun(n, m, r, theta, nflag=None):
    if (n.ndim != 1) or (m.ndim != 1) or (len(n) != len(m)):
        raise ValueError('N and M must be vectors with the same length.')

    if any((n - m) % 2 != 0):
        raise ValueError('All N and M must differ by multiples of 2 (including 0).')

    if any(m > n):
        raise ValueError('Each M must be less than or equal to its corresponding N.')

    if any((r > 1) | (r < 0)) or (r.ndim != 1) or (theta.ndim != 1) or (len(r) != len(theta)):
        raise ValueError('R and THETA must be vectors with the same length, and R must be between 0 and 1.')

    n = n.reshape(-1, 1)
    m = m.reshape(-1, 1)
    m_abs = np.abs(m)
    rpowers = np.unique(np.concatenate([np.arange(m_abs[j], n[j]+1, 2) for j in range(len(n))]))
    rpowern = np.power(np.tile(r.reshape(-1, 1), len(rpowers)), np.tile(rpowers, (len(r), 1)))
    if rpowers[0] == 0:
        rpowern = np.concatenate([np.ones((len(r), 1)), rpowern[:, 1:]], axis=1)

    z = np.zeros((len(r), len(n)))
    for j in range(len(n)):
        s = np.arange(0, (n[j] - m_abs[j])//2 + 1)
        pows = np.arange(n[j], m_abs[j]-1, -2)
        for k in range(len(s)-1, -1, -1):
            p = (1-2*(s[k] % 2)) * np.prod(np.arange(n[j]-s[k]+1, n[j]+1)) / \
                np.prod(np.arange(1, s[k]+1)) / np.prod(np.arange(1, (n[j]-m_abs[j])//2-s[k]+1)) / \
                np.prod(np.arange(1, (n[j]+m_abs[j])//2-s[k]+1))
            idx = np.where(pows[k] == rpowers)[0][0]
            z[:, j] += p * rpowern[:, idx]

        if nflag == 'norm':
            z[:, j] *= np.sqrt((1 + (m[j] != 0)) * (n[j] + 1) / np.pi)

    idx_pos = m > 0
    idx_neg = m < 0
    if np.any(idx_pos):
        z[:, idx_pos] *= np.cos(np.outer(theta, m_abs[idx_pos]))
    if np.any(idx_neg):
        z[:, idx_neg] *= np.sin(np.outer(theta, m_abs[idx_neg]))

    return z

def zernikeFittype(order):
    if order == 0:
        zernike_fun = lambda w00, r, th: \
            np.sum([w00] * zernfun([0], [0], r, th), axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00',))
    elif order == 1:
        zernike_fun = lambda w00, w11, w1_1, r, th: \
            np.sum([w00, w11, w1_1] * zernfun([0, 1, 1], [0, 1, -1], r, th), axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00', 'w11', 'w1_1'))
    elif order == 2:
        zernike_fun = lambda w00, w11, w1_1, w22, w20, w2_2, r, th: \
            np.sum([w00, w11, w1_1, w22, w20, w2_2] * \
                   zernfun([0, 1, 1, 2, 2, 2], [0, 1, -1, 2, 0, -2], r, th), axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00', 'w11', 'w1_1', 'w22', 'w20', 'w2_2')) 
    elif order == 3:
        zernike_fun = lambda w00, w11, w1_1, w22, w20, w2_2, w33, w31, w3_1, w3_3, r, th: \
            np.sum([w00, w11, w1_1, w22, w20, w2_2, w33, w31, w3_1, w3_3] * \
                   zernfun([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], [0, 1, -1, 2, 0, -2, 3, 1, -1, -3], r, th), axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00', 'w11', 'w1_1', 'w22', 'w20', 'w2_2', 'w33', 'w31', 'w3_1', 'w3_3'))
    elif order == 4:
        zernike_fun = lambda w00, w11, w1_1, w22, w20, w2_2, w33, w31, w3_1, w3_3,w44,w42,w40,w4_2,w4_4, r, th: \
            np.sum([w00, w11, w1_1, w22, w20, w2_2, w33, w31, w3_1, w3_3,w44,w42,w40,w4_2,w4_4] * \
                   zernfun([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4], [0,1,-1,2,0,-2,3,1,-1,-3,4,2,0,-2,-4], r, th), axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00', 'w11', 'w1_1', 'w22', 'w20', 'w2_2', 'w33', 'w31', 'w3_1', 'w3_3','w44','w42','w40','w4_2','w4_4'))
    
    elif order == 5:
         zernike_fun = lambda w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,r,th:\
            np.sum([w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5]*\
                   zernfun([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5],[0,1,-1,2,0,-2,3,1,-1,-3,4,2,0,-2,-4,5,3,1,-1,-3,-5],r,th),axis=1)
         zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00','w11','w1_1','w22','w20','w2_2','w33','w31','w3_1','w3_3','w44','w42','w40','w4_2','w4_4','w_55','w53','w51','w5_1','w5_3','w5_5'))
    
    elif order == 6:
        zernike_fun = lambda w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6,r,th:\
            np.sum([w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6]*\
                   zernfun([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6],[0,1,-1,2,0,-2,3,1,-1,-3,4,2,0,-2,-4,5,3,1,-1,-3,-5,6,4,2,0,-2,-4,-6],r,th),axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00','w11','w1_1','w22','w20','w2_2','w33','w31','w3_1','w3_3','w44','w42','w40','w4_2','w4_4','w_55','w53','w51','w5_1','w5_3','w5_5','w66',
                          'w64','w62','w60','w6_2','w6_4','w6_6'))


    elif order == 7:
        zernike_fun = lambda w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6,w77,w75,w73,w71,w7_1,w7_3,w7_5,w7_7,r,th:\
            np.sum([w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6,
                    w77,w75,w73,w71,w7_1,w7_3,w7_5,w7_7]*\
                   zernfun([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7],[0,1,-1,2,0,-2,3,1,-1,-3,4,2,0,-2,-4,5,3,1,-1,-3,-5,6,4,2,0,-2,-4,-6,7,5,3,1,-1,-3,-5,-7],r,th),axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00','w11','w1_1','w22','w20','w2_2','w33','w31','w3_1','w3_3','w44','w42','w40','w4_2','w4_4','w_55','w53','w51','w5_1','w5_3','w5_5','w66',
                          'w64','w62','w60','w6_2','w6_4','w6_6','w77','w75','w73','w71','w7_1','w7_3','w7_5','w7_7'))

    elif order == 8:
        zernike_fun = lambda w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6, w77,w75,w73,w71,w7_1,w7_3,w7_5,w7_7,w88,w86,w84,w82,w80,w8_2,w8_4,w8_6,w8_8,r,th:\
            np.sum([w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6,
                    w77,w75,w73,w71,w7_1,w7_3,w7_5,w7_7,w88,w86,w84,w82,w80,w8_2,w8_4,w8_6,w8_8]*\
                   zernfun([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8],
                           [0,1,-1,2,0,-2,3,1,-1,-3,4,2,0,-2,-4,5,3,1,-1,-3,-5,6,4,2,0,-2,-4,-6,7,5,3,1,-1,-3,-5,-7,8,6,4,2,0,-2,-4,-6,-8],r,th),axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00','w11','w1_1','w22','w20','w2_2','w33','w31','w3_1','w3_3','w44','w42','w40','w4_2','w4_4','w_55','w53','w51','w5_1','w5_3','w5_5','w66',
                          'w64','w62','w60','w6_2','w6_4','w6_6','w77','w75','w73','w71','w7_1','w7_3','w7_5','w7_7','w88','w86','w84','w82','w80','w8_2','w8_4','w8_6','w8_8'))
   
    elif order == 9:
        zernike_fun = lambda w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6,w77,w75,w73,w71,w7_1,w7_3,w7_5,w7_7,w88,w86,w84,w82,w80,w8_2,w8_4,w8_6,w8_8,w99,w97,w95,w93,w91,w9_1,w9_3,w9_5,w9_7,w9_9,r,th:\
            np.sum([w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6,
                    w77,w75,w73,w71,w7_1,w7_3,w7_5,w7_7,w88,w86,w84,w82,w80,w8_2,w8_4,w8_6,w8_8,w99,w97,w95,w93,w91,w9_1,w9_3,w9_5,w9_7,w9_9]*\
                   zernfun([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9],
                           [0,1,-1,2,0,-2,3,1,-1,-3,4,2,0,-2,-4,5,3,1,-1,-3,-5,6,4,2,0,-2,-4,-6,7,5,3,1,-1,-3,-5,-7,8,6,4,2,0,-2,-4,-6,-8,9,7,5,3,1,-1,-3,-5,-7,-9],r,th),axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00','w11','w1_1','w22','w20','w2_2','w33','w31','w3_1','w3_3','w44','w42','w40','w4_2','w4_4','w_55','w53','w51','w5_1','w5_3','w5_5','w66',
                          'w64','w62','w60','w6_2','w6_4','w6_6','w77','w75','w73','w71','w7_1','w7_3','w7_5','w7_7','w88','w86','w84','w82','w80','w8_2','w8_4','w8_6',
                          'w8_8','w99','w97','w95','w93','w91','w9_1','w9_3','w9_5','w9_7','w9_9'))
    
    elif order == 10:
        zernike_fun = lambda w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6,w77,w75,w73,w71,w7_1,w7_3,w7_5,w7_7,w88,w86,w84,w82,w80,w8_2,w8_4,w8_6,w8_8,w99,w97,w95,w93,w91,w9_1,w9_3,w9_5,w9_7,w9_9,w1010,w108,w106,w104,w102,w100,w10_2,w10_4,w10_6,w10_8,w10_10,r,th: \
            np.sum([w00,w11,w1_1,w22,w20,w2_2,w33,w31,w3_1,w3_3,w44,w42,w40,w4_2,w4_4,w_55,w53,w51,w5_1,w5_3,w5_5,w66,w64,w62,w60,w6_2,w6_4,w6_6,
                    w77,w75,w73,w71,w7_1,w7_3,w7_5,w7_7,w88,w86,w84,w82,w80,w8_2,w8_4,w8_6,w8_8,w99,w97,w95,w93,w91,w9_1,w9_3,w9_5,w9_7,w9_9,w1010,w108,w106,w104,w102,w100,w10_2,w10_4,w10_6,w10_8,w10_10]*\
                   zernfun([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10],
                           [0,1,-1,2,0,-2,3,1,-1,-3,4,2,0,-2,-4,5,3,1,-1,-3,-5,6,4,2,0,-2,-4,-6,7,5,3,1,-1,-3,-5,-7,8,6,4,2,0,-2,-4,-6,-8,9,7,5,3,1,-1,-3,-5,-7,-9,10,8,6,4,2,0,-2,-4,-6,-8,-10],r,th),axis=1)
        zernike_fittype = curve_fit(zernike_fun,
            dependent='z',
            independent=('r', 'th'),
            coefficients=('w00','w11','w1_1','w22','w20','w2_2','w33','w31','w3_1','w3_3','w44','w42','w40','w4_2','w4_4','w_55','w53','w51','w5_1','w5_3','w5_5','w66',
                          'w64','w62','w60','w6_2','w6_4','w6_6','w77','w75','w73','w71','w7_1','w7_3','w7_5','w7_7','w88','w86','w84','w82','w80','w8_2','w8_4','w8_6',
                          'w8_8','w99','w97','w95','w93','w91','w9_1','w9_3','w9_5','w9_7','w9_9','w1010','w108','w106','w104','w102','w100','w10_2','w10_4','w10_6','w10_8','w10_10'))
    
    else:
        print("An implementation for the Zernike fit function of this order does not exist yet!")
        return None
     
##################################################################################################################        
#All the code below corresponds to the phase_unwrap functions that are implemented on MATLAB
##################################################################################################################

def phase_unwrap(psi, weight=None):
    if weight is None:  # unweighted phase unwrap
        # get the wrapped differences of the wrapped values
        dx = np.concatenate((np.zeros((psi.shape[0], 1)), np.unwrap(np.diff(psi, axis=1)), np.zeros((psi.shape[0], 1))), axis=1)
        dy = np.concatenate((np.zeros((1, psi.shape[1])), np.unwrap(np.diff(psi, axis=0)), np.zeros((1, psi.shape[1]))), axis=0)
        rho = np.diff(dx, axis=1) + np.diff(dy, axis=0)

        # get the result by solving the poisson equation
        phi = solvePoisson(rho)

    else:  # weighted phase unwrap
        # check if the weight has the same size as psi
        if weight.shape != psi.shape:
            raise ValueError("Size of the weight must be the same as the size of the wrapped phase.")

        # vector b in the paper (eq 15) is dx and dy
        dx = np.concatenate((np.unwrap(np.diff(psi, axis=1)), np.zeros((psi.shape[0], 1))), axis=1)
        dy = np.concatenate((np.unwrap(np.diff(psi, axis=0)), np.zeros((1, psi.shape[1]))), axis=0)

        # multiply the vector b by weight square (W^T * W)
        WW = weight * weight
        WWdx = WW * dx
        WWdy = WW * dy

        # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
        WWdx2 = np.concatenate((np.zeros((psi.shape[0], 1)), WWdx), axis=1)
        WWdy2 = np.concatenate((np.zeros((1, psi.shape[1])), WWdy), axis=0)
        rk = np.diff(WWdx2, axis=1) + np.diff(WWdy2, axis=0)
        normR0 = np.linalg.norm(rk)

        # start the iteration
        eps = 1e-8
        k = 0
        phi = np.zeros(psi.shape)
        while not np.all(rk == 0):
            zk = solvePoisson(rk)
            k += 1

            if k == 1:
                pk = zk
            else:
                betak = np.sum(rk * zk) / np.sum(rkprev * zkprev)
                pk = zk + betak * pk

            # save the current value as the previous values
            rkprev = rk
            zkprev = zk

            # perform one scalar and two vectors update
            Qpk = applyQ(pk, WW)
            alphak = np.sum(rk * zk) / np.sum(pk * Qpk)
            phi += alphak * pk
            rk -= alphak * Qpk

            # check the stopping conditions
            if k >= np.prod(psi.shape) or np.linalg.norm(rk) < eps * normR0:
                break

    return phi


def solvePoisson(rho):
    # solve the Poisson equation using dct
    dctRho = dct2(rho)
    N, M = rho.shape
    I, J = np.meshgrid(np.arange(M), np.arange(N))
    dctPhi = dctRho / (2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2))
    dctPhi[0, 0] = 0  # handling the inf/nan value

    # now invert to get the result
    phi = idct2(dctPhi)

    return phi

def applyQ(p, WW):
    # apply (A)
    dx = np.diff(p, axis=1, append=0)
    dy = np.diff(p, axis=0, append=0)
    
    # apply (W^T)(W)
    WWdx = WW * dx
    WWdy = WW * dy
    
    # apply (A^T)
    WWdx2 = np.concatenate((np.zeros((p.shape[0], 1)), WWdx), axis=1)
    WWdy2 = np.concatenate((np.zeros((1, p.shape[1])), WWdy), axis=0)
    Qp = np.diff(WWdx2, axis=1) + np.diff(WWdy2, axis=0)
    
    return Qp

def wrap_to_pi(a_in):
    a_out = a_in - 2 * np.pi * np.floor((a_in + np.pi) / (2 * np.pi))
    return a_out

def dct(a, n=None):
    """
    DCT Discrete cosine transform.

    Y = dct(X) returns the discrete cosine transform of X.
    The vector Y is the same size as X and contains the
    discrete cosine transform coefficients.

    Y = dct(X, N) pads or truncates the vector X to length N
    before transforming.

    If X is a matrix, the DCT operation is applied to each
    column. This transform can be inverted using IDCT.

    Parameters:
    - a: input array or matrix
    - n: length to pad or truncate the input (optional)

    Returns:
    - b: discrete cosine transform coefficients

    Reference:
    - A. K. Jain, "Fundamentals of Digital Image Processing", pp. 150-153.
    - Wallace, "The JPEG Still Picture Compression Standard",
      Communications of the ACM, April 1991.
    """
    if len(a) == 0:
        return np.array([])
    
    # If input is a vector, make it a column:
    do_trans = (a.shape[0] == 1)
    if do_trans:
        a = a.reshape(-1, 1)
    
    if n is None:
        n = a.shape[0]
    m = a.shape[1]
    
    # Pad or truncate input if necessary
    if a.shape[0] < n:
        aa = np.zeros((n, m))
        aa[:a.shape[0], :] = a
    else:
        aa = a[:n, :]
    
    # Compute weights to multiply DFT coefficients
    ww = (np.exp(-1j * np.arange(n) * np.pi / (2 * n)) / np.sqrt(2 * n)).reshape(-1, 1)
    ww[0] = ww[0] / np.sqrt(2)
    
    if n % 2 == 1 or not np.isrealobj(a):  # odd case
        # Form intermediate even-symmetric matrix
        y = np.zeros((2 * n, m))
        y[:n, :] = aa
        y[n:2 * n, :] = np.flipud(aa)
        
        # Compute the FFT and keep the appropriate portion
        #yy = np.fft.fft(y, axis=0)[:n, :] #this one doesnt seem to work for all scenarios
        yy = np.fft.fft(y, axis=0)
        yy = yy[:n, :]
    
    else:  # even case
        # Re-order the elements of the columns of x
        y = np.vstack((aa[0::2, :], aa[n - 1::-2, :]))
        yy = np.fft.fft(y, axis=0)
        ww = 2 * ww  # Double the weights for even-length case
    
    # Multiply FFT by weights
    b = ww * yy
    
    if np.isrealobj(a):
        b = b.real
    
    if do_trans:
        b = b.T
    
    return b



def dct2(arg1, mrows=None, ncols=None):
    """
    Compute 2-D discrete cosine transform.

    Parameters:
    - arg1: Input matrix (numeric or logical).
    - mrows: Number of rows for padding (optional).
    - ncols: Number of columns for padding (optional).

    Returns:
    - b: Discrete cosine transform coefficients.
    """
    m, n = arg1.shape

    # Basic algorithm.
    if mrows is None and ncols is None:
        if m > 1 and n > 1:
            b = dct(dct(arg1).T).T
            return b
        else:
            mrows = m
            ncols = n

    # Padding for vector input.
    a = arg1.copy()
    if mrows is not None and ncols is None:
        ncols = mrows[1]
        mrows = mrows[0]

    mpad = mrows
    npad = ncols

    if m == 1 and mpad > m:
        a = np.pad(a, ((0, 1), (0, 0)), mode='constant')
        m = 2

    if n == 1 and npad > n:
        a = np.pad(a, ((0, 0), (0, 1)), mode='constant')
        n = 2

    if m == 1:
        mpad = npad
        npad = 1  # For row vector.

    # Transform.
    b = dct(a, mpad)
    if m > 1 and n > 1:
        b = dct(b.T, npad).T

    return b



def idct(b, n=None):
    """
    Inverse discrete cosine transform.

    Parameters:
    - b: Input vector or matrix.
    - n: Length to pad or truncate the vector (optional).

    Returns:
    - a: Inverse discrete cosine transform result.
    """
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)

    if np.min(b.shape) == 1:
        if b.shape[1] > 1:
            do_trans = True
        else:
            do_trans = False
        b = b.flatten()
    else:
        do_trans = False

    if n is None:
        n = b.shape[0]
    
    m = b.shape[1]

    # Pad or truncate b if necessary
    if b.shape[0] < n:
        bb = np.zeros((n, m))
        bb[:b.shape[0], :] = b
    else:
        bb = b[:n, :]

    if n % 2 == 1 or not np.isreal(b).all():  # odd case
        # Form intermediate even-symmetric matrix.
        ww = np.sqrt(2 * n) * np.exp(1j * np.arange(n) * np.pi / (2 * n))
        ww[0] = ww[0] * np.sqrt(2)
        W = ww[:, np.newaxis] * np.ones((1, m))
        
        yy = np.zeros((2 * n, m), dtype=np.complex128)
        yy[:n, :] = W * bb
        yy[n+1:n+n+1, :] = -1j * W[1:n, :] * np.flipud(bb[1:n, :])

        y = ifft(yy, axis=0)

        # Extract inverse DCT
        a = y[:n, :]

    else:  # even case
        # Compute precorrection factor
        ww = np.sqrt(2 * n) * np.exp(1j * np.pi * np.arange(n) / (2 * n))
        ww[0] = ww[0] / np.sqrt(2)
        W = ww[:, np.newaxis]

        # Compute x tilde using equation (5.93) in Jain
        y = np.fft.ifft(W * bb, axis=0)

        # Re-order elements of each column according to equations (5.93) and (5.94) in Jain
        a = np.zeros((n, m))
        a[0:2:n, :] = y[:n//2, :]
        a[1:2:n, :] = y[n-1::-n//2, :]

    if np.isreal(b).all():
        a = np.real(a)
    
    if do_trans:
        a = a.T

    return a



def idct2(arg1, mrows=None, ncols=None):
    """
    Compute 2-D inverse discrete cosine transform.

    Parameters:
    - arg1: Input matrix (numeric or logical).
    - mrows: Number of rows for padding (optional).
    - ncols: Number of columns for padding (optional).

    Returns:
    - a: Inverse discrete cosine transform result.
    """
    m, n = arg1.shape

    # Basic algorithm.
    if mrows is None and ncols is None:
        if m > 1 and n > 1:
            a = idct(idct(arg1).T).T
            return a
        else:
            mrows = m
            ncols = n

    # Padding for vector input.
    b = arg1.copy()
    if mrows is not None and ncols is None:
        ncols = mrows[1]
        mrows = mrows[0]

    mpad = mrows
    npad = ncols

    if m == 1 and mpad > m:
        b = np.pad(b, ((0, 1), (0, 0)), mode='constant')
        m = 2

    if n == 1 and npad > n:
        b = np.pad(b, ((0, 0), (0, 1)), mode='constant')
        n = 2

    if m == 1:
        mpad = npad
        npad = 1  # For row vector.

    # Transform.
    a = idct(b, mpad)
    if m > 1 and n > 1:
        a = idct(a.T, npad).T

    return a