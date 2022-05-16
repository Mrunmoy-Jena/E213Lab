#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: smilex
"""

#modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#efficiency matrix calculations
#--------------------

#DATA! data from the experiment between asteriks

#*****
#total ee, mm, tt, qq events
tot = np.array([56720, 89887, 79214, 98563])

#ee, mm, tt, qq cuts on monte carlo events
ee = np.array([18835*1.5829, 0, 26, 0]) #ee cut applied to ee, mm, tt, qq MC
mm = np.array([0, 76209, 35, 0]) #mm cut applied to ee, mm, tt, qq MC
tt = np.array([378, 4105, 71131, 173]) #tt cut applied to ee, mm, tt, qq MC
qq = np.array([0, 0, 308, 92898]) #qq cut applied to ee, mm, tt, qq MC
#*****

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
print("*****")
print("Efficiency matrix:")
print(np.round(eff, 2))

#delta efficiency
print("+-")
print(np.round(deltaeff, 3))
print("*****")

#inverse efficiency matrix
print("Inverse efficiency matrix:")
print(np.round(effinv, 4))

#delta inverse efficiency
print("+-")
print(np.round(deltaeffinv, 5))
print("*****")
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

#radiative corrections - values taken from the manual
leptoncorrection = np.array([0.09, 0.20, 0.36, 0.52, 0.22, -0.01, -0.08])
hadroncorrection = np.array([2.0, 4.3, 7.7, 10.8, 4.7, -0.2, -1.6])

#DATA! data from the experiment between asteriks

#*****
partcount[0] = [106, 120, 251, 3281] #ee, mm,tt, qq counts for sqrt label 0
partcount[1] = [261, 304, 425, 7413] #ee, mm,tt, qq counts for sqrt label 1
partcount[2] = [415, 591, 693,14709] #ee, mm,tt, qq counts for sqrt label 2
partcount[3] = [4839, 8681, 10178, 221068] #ee, mm,tt, qq counts for sqrt label 3
partcount[4] = [393, 767, 863, 18727] #ee, mm,tt, qq counts for sqrt label 4
partcount[5] = [150, 296, 427, 8100] #ee, mm,tt, qq counts for sqrt label 5
partcount[6] = [178, 339, 459, 8635] #ee, mm,tt, qq counts for sqrt label 6
#*****

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
print("*****")

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
print("*****")
#--------------------

#forward backward asymmetry calculations
#--------------------

#DATA! data from the experiment between the asteriks

#*****
nminus = np.array([69, 156, 319, 4384, 380, 123, 150]) #backward counts for mm
nplus = np.array([51, 148, 272, 4297, 387, 173, 189]) #forward counts for mm
#*****

#afb correction values - taken from the manual
afbcorrection = np.array([0.021512, 0.019262, 0.016713, 0.018293, 0.030286, 0.062196, 0.093850])

#afb and weinbergtheta
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
print("*****")

print("Weinberg theta: ", weinbergtheta, "+-", deltaweinbergtheta)
print("*****")
#--------------------

#breit wigner curve fit
#--------------------
#a: Z mass
#b: Z decay width
#gammach: product of electron decay width and the final state decay width
def fitfunc(x, a, b, gammach):
    sigmaofs = (12 * np.pi * 3.89E5 * gammach * x) / \
        (a**2 * ((x - a**2)**2 + (x * b/a)**2))
    return sigmaofs

#decay width values - taken from the manual
gammalep = 83.8E-3
gammaneu = 167.6E-3
gammauc = 299E-3
gammadsb = 378E-3

#s values - taken from the manual (notice the square!)
x_data = np.square(np.array([88.47, 89.46, 90.22, 91.22, 91.97, 92.96, 93.71]))

#slice ee sigma values
ee_data = sigma[:, 0]
ee_data_err = deltasigma[:, 0]

#slice mm sigma values
mm_data = sigma[:, 1]
mm_data_err = deltasigma[:, 1]

#sllice tt sigma values
tt_data = sigma[:, 2]
tt_data_err = deltasigma[:, 2]

#slice qq sigma values
qq_data = sigma[:, 3]
qq_data_err = deltasigma[:, 3]


#fit for leptons
#fit guess taken as 91, 2.5 and 6.5 - Z mass and decay width and channel decay width
popt_ee, pcov_ee = curve_fit(fitfunc, x_data, ee_data, p0 = [91, 2.5, 6.5])
popt_mm, pcov_mm = curve_fit(fitfunc, x_data, mm_data, p0 = [91, 2.5, 6.5])
popt_tt, pcov_tt = curve_fit(fitfunc, x_data, tt_data, p0 = [91, 2.5, 6.5])

#fit for hadrons
#fit guess taken as 91, 2.5 and 150 - Z mass and decay width and channel decay width
popt_qq, pcov_qq = curve_fit(fitfunc, x_data, qq_data, p0 = [91, 2.5, 150])

x_model = np.linspace(min(x_data), max(x_data), 100)

#ee fit
ee_model = fitfunc(x_model, popt_ee[0], popt_ee[1], popt_ee[2])
plt.scatter(x_data, ee_data)
plt.errorbar(x_data, ee_data, yerr = ee_data_err, fmt="o")
plt.plot(x_model, ee_model)
plt.show()

#mm fit
mm_model = fitfunc(x_model, popt_mm[0], popt_mm[1], popt_mm[2])
plt.scatter(x_data, mm_data)
plt.errorbar(x_data, mm_data, yerr = mm_data_err, fmt="o")
plt.plot(x_model, mm_model)
plt.show()

#tt fit
tt_model = fitfunc(x_model, popt_tt[0], popt_tt[1], popt_tt[2])
plt.scatter(x_data, tt_data)
plt.errorbar(x_data, tt_data, yerr = tt_data_err, fmt="o")
plt.plot(x_model, tt_model)
plt.show()

#qq fit
qq_model = fitfunc(x_model, popt_qq[0], popt_qq[1], popt_qq[2])
plt.scatter(x_data, qq_data)
plt.errorbar(x_data, qq_data, yerr = qq_data_err, fmt="o")
plt.plot(x_model, qq_model)
plt.show()

#print a, b and gammach for all branches
print("Fit parameters")
print("ee: ", popt_ee)
print("mm: ", popt_mm)
print("tt: ", popt_tt)
print("qq: ", popt_qq)
print("*****")

print("Goodness of fit parameters (lower the value, the better)")
print("ee: ", np.sqrt(np.diag(pcov_ee)))
print("mm: ", np.sqrt(np.diag(pcov_mm)))
print("tt: ", np.sqrt(np.diag(pcov_tt)))
print("qq: ", np.sqrt(np.diag(pcov_qq)))
print("*****")