import numpy as np
from scipy.optimize import curve_fit

def h5_LV_attributes_to_struct(h5_info_attrib, attrib_out=None):
    """
    Turns a list of attributes of an HDF element into a data structure.
    Each attribute is supposed to be a LabView control flattened to XML.
    """
    # Obtain structure to append, if it is passed.
    if attrib_out is None:
        attrib_out = dict()

    # Attributes that will be ignored
    ignored_attribute_names = []

    for attrib in h5_info_attrib:
        name = attrib.name
        if name not in ignored_attribute_names:
            try:
                attrib_out = parseLabviewXML_str(attrib.value[0], attrib_out)
            except TypeError:
                attrib_out = parseLabviewXML_str(attrib.value, attrib_out)

    return attrib_out


# Read pupil segmentation using the following 6 functions

def createLegalStructFieldName(name):
    name = name.replace(' ', '_')
    name = name.replace('[', '_')
    name = name.replace(']', '_')
    name = name.replace('@', '_at_')
    name = name.replace('-', '_dash_')
    name = name.replace(',', '_')
    name = name.replace('(', '')
    name = name.replace(')', '')
    name = name.replace('{', '')
    name = name.replace('}', '')
    name = name.replace('.', '_dot_')
    name = name.replace('>', '_greater_')
    name = name.replace('<', '_less_')
    name = name.replace('=', '_equal_')
    name = name.replace('#', 'numer_of')
    name = name.replace('?', '_questionmark_')
    name = name.replace('+', '_plus_')
    name = name.replace('/', '')
    name = name.replace('\\', '')
    name = name.replace('%', '')
    name = name.replace('\n', '')
    # determine if string starts with number
    if name[0].isdigit():
        name = 'NUM_' + name

    return name

def shorten_group_name(str):
    # find slashes
    slash_idx = [i for i, char in enumerate(str) if char == '/']
    # keep everything after the last slash
    str_out = str[slash_idx[-1]+1:]
    return str_out

def h5_LV_MF_format_extract_parameters(parameters_info):
    # Load all parameters from file with MF format.
    # Each VI parameters set is loaded into a field of the structure.
    p = dict()


    for group in parameters_info['slm_control']:
        group_name = shorten_group_name(group['Name'])
        p[group_name] = h5_LV_attributes_to_struct(group['Attributes'])

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
     
         
   
