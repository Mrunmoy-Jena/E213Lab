#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: smilex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#efficiency matrix calculations
#--------------------
#DATA
#total ee, mm, tt, qq events
tot = np.array([56720, 89887, 79214, 98563])

#ee, mm, tt, qq cuts on monte carlo events
ee = np.array([18835*1.5829, 0, 26, 0]) #ee cut applied to ee, mm, tt, qq MC
mm = np.array([0, 76209, 35, 0]) #mm cut applied to ee, mm, tt, qq MC
tt = np.array([378, 4105, 71131, 173]) #tt cut applied to ee, mm, tt, qq MC
qq = np.array([0, 0, 308, 92898]) #qq cut applied to ee, mm, tt, qq MC

#efficiency matrix
eff = np.zeros([4,4])

eff[0] = ee/tot
eff[1] = mm/tot
eff[2] = tt/tot
eff[3] = qq/tot

#efficiency inverse
effinv = np.linalg.inv(eff)

#error in efficiency matrix
deltaeff = np.zeros([4,4])

deltaeff[0] = np.sqrt(eff[0, :]*(1-eff[0, :])/tot[0])
deltaeff[1] = np.sqrt(eff[1, :]*(1-eff[1, :])/tot[1])
deltaeff[2] = np.sqrt(eff[2, :]*(1-eff[2, :])/tot[2])
deltaeff[3] = np.sqrt(eff[3, :]*(1-eff[3, :])/tot[3])

effinvsq = np.square(effinv)
deltaeffsq = np.square(deltaeff)

#error in efficiency inverse matrix
deltaeffinv = np.sqrt(np.matmul(effinvsq, (np.matmul(deltaeffsq, effinvsq))))

#efficiency matrix
print("Efficiency matrix:")
print(np.round(eff, 2))

#delta efficiency
print("+-")
print(np.round(deltaeff, 3))

#inverse efficiency matrix
print("Inverse efficiency matrix:")
print(np.round(effinv, 4))

#delta inverse efficiency
print("+-")
print(np.round(deltaeffinv, 5))
#--------------------

#cross section calculations
#--------------------
#root-s values labels
#0: 88.47 GeV
#1: 89.46 GeV
#2: 90.22 GeV
#3: 91.22 GeV
#4: 91.97 GeV
#5: 92.96 GeV
#6: 93.71 GeV

#particle counts measured
partcount = np.zeros([7, 4])
#integrated luminosity values for data6.root
intlum = np.array([675.9, 800.8, 873.7, 7893.5, 825.3, 624.6, 942.2])
deltaintlum = np.array([5.7, 6.6, 7.1, 54.3, 6.9, 5.5, 7.7])

#radiative corrections
leptoncorrection = np.array([0.09, 0.20, 0.36, 0.52, 0.22, -0.01, -0.08])
hadroncorrection = np.array([2.0, 4.3, 7.7, 10.8, 4.7, -0.2, -1.6])

#DATA
partcount[0] = [106, 120, 251, 3281]
partcount[1] = [261, 304, 425, 7413]
partcount[2] = [415, 591, 693,14709]
partcount[3] = [4839, 8681, 10178, 221068]
partcount[4] = [393, 767, 863, 18727]
partcount[5] = [150, 296, 427, 8100]
partcount[6] = [178, 339, 459, 8635]

#actual particle counts
partcountact = np.zeros([7, 4])
for i in range(7):
    partcountact[i] = np.matmul(effinv, partcount[i])
print("Actual particle counts:")
print(partcountact)

#delta actual particle counts
deltapartcountact = np.zeros([7, 4])

deltaeffinvsq = np.square(deltaeffinv)

for i in range(7):
    deltapartcountact[i] = np.matmul(deltaeffinvsq, np.square(partcount[i])) + \
        np.matmul(effinvsq, partcount[i])
deltapartcountact = np.sqrt(deltapartcountact)

print("+-")
print(deltapartcountact)

#cross sections
sigma = np.zeros([7, 4])
for i in range(7):
    sigma[i] = partcountact[i]/intlum[i]
for i in range(7):
    sigma[i, :3] = sigma[i, :3] + leptoncorrection[i]
    sigma[i, 3] = sigma[i, 3] + hadroncorrection[i]
print("Cross sections:")
print(sigma)

#delta cross sections
deltasigma = np.zeros([7, 4])

for i in range(7):
    deltasigma[i] = np.square(deltapartcountact[i])/np.square(intlum[i]) + \
        np.square(partcountact[i])*np.square(deltaintlum[i])/np.power(intlum[i], 4)
deltasigma = np.sqrt(deltasigma)

print("+-")
print(deltasigma)
#--------------------

#forward backward asymmetry calculations
#--------------------
#DATA
nminus = np.array([69, 156, 319, 4384, 380, 123, 150])
nplus = np.array([51, 148, 272, 4297, 387, 173, 189])

#afb and weinbergtheta
afbcorrection = np.array([0.021512, 0.019262, 0.016713, 0.018293, 0.030286, 0.062196, 0.093850])

afb = (nplus - nminus)/(nplus + nminus) + afbcorrection
weinbergtheta = (1/4)*(1 - np.sqrt(afb[3]/3))

#delta afb
deltaafb = np.sqrt((nminus*np.square(2*nplus/np.square(nplus+nminus))) + \
                   (nplus*np.square(2*nminus/np.square(nplus+nminus))))
deltaweinbergtheta = (1/2)*(deltaafb[3]/afb[3])*weinbergtheta

print("AFB values:")
print(afb)
print("+-")
print(deltaafb)

print("Weinberg theta: ", weinbergtheta, "+-", deltaweinbergtheta)
#--------------------

#breit wigner curve fit
#--------------------
