# -*- coding: utf-8 -*-
"""
Code to read in RR pair counts, convert to Q_q(s), then renormalise such
that the first non-zero Q_0(s) is 1
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Config space window function from Cobaya tree directory
'''
Q_s = np.loadtxt("window_BOSS_CMASS_NGC_z057_cobaya.txt", usecols=0)
Q_0 = np.loadtxt("window_BOSS_CMASS_NGC_z057_cobaya.txt", usecols=1)
Q_2 = np.loadtxt("window_BOSS_CMASS_NGC_z057_cobaya.txt", usecols=2)
Q_4 = np.loadtxt("window_BOSS_CMASS_NGC_z057_cobaya.txt", usecols=3)

fig = plt.figure(1, (7,7))
ax = fig.add_subplot(1,1,1)
figure = plt.gcf()
figure.set_size_inches(10, 8)

matplotlib.rcParams.update({'font.size': 20})
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
#plt.rc('axes', labelsize=20)
#plt.rc('axes', titlesize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.semilogx(Q_s, Q_0, '-', label=r'$Q_{0}$')
ax.semilogx(Q_s, Q_2, '-', label=r'$Q_{2}$')
ax.semilogx(Q_s, Q_4, '-', label=r'$Q_{4}$')
ax.legend(prop={'size':20})
plt.ylim(-0.8, 1.2)
plt.xlim(0.1, 4000)
plt.savefig("DirectQ_0Q_2fromCobaya.png", bbox_inches='tight', dpi=300)
plt.show()
'''

s = np.loadtxt("Beutleretal_window_z1_NGC.dat", skiprows=1, usecols=0) #Extract real space s from Beutler rr pair counts

RR0_z1_NGC = np.loadtxt("Beutleretal_window_z1_NGC.dat", skiprows=0, usecols=2)
RR2_z1_NGC = np.loadtxt("Beutleretal_window_z1_NGC.dat", skiprows=0, usecols=3)
RR4_z1_NGC = np.loadtxt("Beutleretal_window_z1_NGC.dat", skiprows=0, usecols=4)

RR0_z2_NGC = np.loadtxt("Beutleretal_window_z2_NGC.dat", skiprows=1, usecols=2)
RR2_z2_NGC = np.loadtxt("Beutleretal_window_z2_NGC.dat", skiprows=1, usecols=3)
RR4_z2_NGC = np.loadtxt("Beutleretal_window_z2_NGC.dat", skiprows=1, usecols=4)

RR0_z3_NGC = np.loadtxt("Beutleretal_window_z3_NGC.dat", skiprows=1, usecols=2)
RR2_z3_NGC = np.loadtxt("Beutleretal_window_z3_NGC.dat", skiprows=1, usecols=3)
RR4_z3_NGC = np.loadtxt("Beutleretal_window_z3_NGC.dat", skiprows=1, usecols=4)

RR0_z1_SGC = np.loadtxt("Beutleretal_window_z1_SGC.dat", skiprows=1, usecols=2)
RR2_z1_SGC = np.loadtxt("Beutleretal_window_z1_SGC.dat", skiprows=1, usecols=3)
RR4_z1_SGC = np.loadtxt("Beutleretal_window_z1_SGC.dat", skiprows=1, usecols=4)

RR0_z2_SGC = np.loadtxt("Beutleretal_window_z2_SGC.dat", skiprows=1, usecols=2)
RR2_z2_SGC = np.loadtxt("Beutleretal_window_z2_SGC.dat", skiprows=1, usecols=3)
RR4_z2_SGC = np.loadtxt("Beutleretal_window_z2_SGC.dat", skiprows=1, usecols=4)

RR0_z3_SGC = np.loadtxt("Beutleretal_window_z3_SGC.dat", skiprows=1, usecols=2) 
RR2_z3_SGC = np.loadtxt("Beutleretal_window_z3_SGC.dat", skiprows=1, usecols=3) 
RR4_z3_SGC = np.loadtxt("Beutleretal_window_z3_SGC.dat", skiprows=1, usecols=4)
#RR4 = np.loadtxt("Beutleretal_window_z1_NGC.dat", skiprows=1, usecols=4) #Extract real space s from Beutler rr pair counts
#RR6 = np.loadtxt("Beutleretal_window_z1_NGC.dat", skiprows=1, usecols=5) #Extract real space s from Beutler rr pair counts
#RR8 = np.loadtxt("Beutleretal_window_z1_NGC.dat", skiprows=1, usecols=6) #Extract real space s from Beutler rr pair counts

length = len(s)

#Includes Qal output from pybird

data = np.load("FirstRoundQalOutput.npy")
print(data[0,2,2400])
Window_0_0 = data[0,0,:] 
Window_2_2 = data[1,1,:] 

#First, we run through and divide by s^3. Also search for first non-zero RR0 term for normalisation

#Test 0- Rebin, then normalise as per 1511.07799 Updated to plot NGC/SGC RR0 RR2 pairs as per Beutler paper

def Test0(RR0, RR2, RR4, binning, OutputFile):

    RR0_binned = np.zeros(binning)
    RR0_Binned_value = 0
    RR2_binned = np.zeros(binning)
    RR2_Binned_value = 0
    RR4_binned = np.zeros(binning)
    RR4_Binned_value = 0
    s_binned = np.zeros(binning)
    step = int((length+1)/binning)
    print("step = %lf" %step)

    for i in range(length):
        RR0_Binned_value = RR0_Binned_value + RR0[i]
        RR2_Binned_value = RR2_Binned_value + RR2[i]
        RR4_Binned_value = RR4_Binned_value + RR4[i]
        if (i % step == 0) & (i !=0):
            print("i binned = %lf" %(i/step))
            RR0_binned[int(i/step)-1] = RR0_Binned_value
            RR2_binned[int(i/step)-1] = RR2_Binned_value
            RR4_binned[int(i/step)-1] = RR4_Binned_value
            RR0_Binned_value = 0
            RR2_Binned_value = 0
            RR4_Binned_value = 0
            s_binned[int(i/step)-1] = s[i-(int(step/2))] #choose the midpoint- evenly spaced in ln-space

    #now we divide by s^3, and normalise such that RR0_binned[0] = 1

    counter = 0
    RR0_max = 0
    sumNorm = 0
    for i in range(binning):
        RR0_binned[i] = RR0_binned[i]/(s_binned[i]**3)
        RR2_binned[i] = RR2_binned[i]/(s_binned[i]**3)
        RR4_binned[i] = RR4_binned[i]/(s_binned[i]**3)
        
        #if (RR0_binned[i] != 0) & (counter == 0): #Use first non-zero RR0
        #if (RR0_binned[i] > RR0_max): #Use largest RR0 term as normalisation
        #    Norm = RR0_binned[i]
        #    RR0_max = RR0_binned[i]
        #    print("s_binned = %lf" %s_binned[i])
        #    print("Norm is %lf" %Norm)
        #    print(RR0_binned[i]/Norm)
        #    counter = counter + 1
        
        '''
        if (RR0_binned[i] != 0) & (counter < 100) & (s_binned[i]>1): #Use first 100 non-zero Q_0 where s > 1,
            sumNorm = sumNorm + RR0_binned[i] 
            print(sumNorm)
            print(counter)
            print(s_binned[i])
            counter = counter + 1
            Norm = sumNorm/counter
            print("Norm = %lf" %Norm)
        '''
        
        if (RR0_binned[i] != 0) & (counter < 20) & (s_binned[i]>10): #Fix terms up to s=10 to 1, use 20 terms after this for normalisation
            sumNorm = sumNorm + RR0_binned[i] 
            print(sumNorm)
            print(counter)
            print(s_binned[i])
            counter = counter + 1
            Norm = sumNorm/counter
            print("Norm = %lf" %Norm)
        
    for i in range(binning):
        if s_binned[i]<10:
            RR0_binned[i] = 1
            RR2_binned[i] = 0
            RR4_binned[i] = 0
        else:
            RR0_binned[i] = RR0_binned[i]/Norm
            RR2_binned[i] = RR2_binned[i]/Norm 
            RR4_binned[i] = RR4_binned[i]/Norm 
    
    output_array = np.column_stack((s_binned, RR0_binned, RR2_binned, RR4_binned))
    np.savetxt(OutputFile, output_array)
    
    return(s_binned, RR0_binned, RR2_binned, RR4_binned)

#Run function for Random-Random pair counts, and print

#s_binned, RR0_z1_NGC, RR2_z1_NGC, RR4_z1_NGC = Test0(RR0_z1_NGC, RR2_z1_NGC, RR4_z1_NGC, 5000, "Ql_z1_NGC_rebinned_5000bins_s1fixed.dat")
s_binned, RR0_z1_NGC, RR2_z1_NGC, RR4_z1_NGC = Test0(RR0_z1_NGC, RR2_z1_NGC, RR4_z1_NGC, 5000, "Ql_z1_NGC_rebinned_5000bins_s10fixed.dat") #Don't want to overwrite the actual results lmao
s_binned, RR0_z1_SGC, RR2_z1_SGC, RR4_z1_SGC = Test0(RR0_z1_SGC, RR2_z1_SGC, RR4_z1_SGC, 5000, "Ql_z1_SGC_rebinned_5000bins_s10fixed.dat") #Don't want to overwrite the actual results lmao

#block of code to bin after normalising (for nice plots):

binning = 100 #Bin all 5000 points into 100 bins equally spaced in ln(k)
RR0_NGC = RR0_z1_NGC
RR2_NGC = RR2_z1_NGC
RR4_NGC = RR4_z1_NGC

RR0_SGC = RR0_z1_SGC
RR2_SGC = RR2_z1_SGC
RR4_SGC = RR4_z1_SGC
    
RR0_NGC_binned = np.zeros(binning)
RR0_NGC_Binned_value = 0
RR2_NGC_binned = np.zeros(binning)
RR2_NGC_Binned_value = 0
RR4_NGC_binned = np.zeros(binning)
RR4_NGC_Binned_value = 0

RR0_SGC_binned = np.zeros(binning)
RR0_SGC_Binned_value = 0
RR2_SGC_binned = np.zeros(binning)
RR2_SGC_Binned_value = 0
RR4_SGC_binned = np.zeros(binning)
RR4_SGC_Binned_value = 0

s_binned = np.zeros(binning)
step = int((length+1)/binning)
print("step = %lf" %step)    
counter_NGC = 0
counter_SGC = 0
for i in range(length):
     RR0_NGC_Binned_value = RR0_NGC_Binned_value + RR0_NGC[i]
     RR2_NGC_Binned_value = RR2_NGC_Binned_value + RR2_NGC[i]
     RR4_NGC_Binned_value = RR4_NGC_Binned_value + RR4_NGC[i]
     
     RR0_SGC_Binned_value = RR0_SGC_Binned_value + RR0_SGC[i]
     RR2_SGC_Binned_value = RR2_SGC_Binned_value + RR2_SGC[i]
     RR4_SGC_Binned_value = RR4_SGC_Binned_value + RR4_SGC[i]
     if RR0_NGC_Binned_value != 0:
         counter_NGC = counter_NGC + 1     
         print("NGC counted!")
     if RR0_SGC_Binned_value != 0:
         counter_SGC = counter_SGC + 1     
         print("SGC counted!")     
    
     if (i % step == 0) & (i !=0):
         print("i binned = %lf" %(i/step))
         RR0_NGC_binned[int(i/step)-1] = RR0_NGC_Binned_value/counter_NGC
         RR2_NGC_binned[int(i/step)-1] = RR2_NGC_Binned_value/counter_NGC
         RR4_NGC_binned[int(i/step)-1] = RR4_NGC_Binned_value/counter_NGC
         RR0_NGC_Binned_value = 0
         RR2_NGC_Binned_value = 0
         RR4_NGC_Binned_value = 0
         
         RR0_SGC_binned[int(i/step)-1] = RR0_SGC_Binned_value/counter_SGC
         RR2_SGC_binned[int(i/step)-1] = RR2_SGC_Binned_value/counter_SGC
         RR4_SGC_binned[int(i/step)-1] = RR4_SGC_Binned_value/counter_SGC
         RR0_SGC_Binned_value = 0
         RR2_SGC_Binned_value = 0
         RR4_SGC_Binned_value = 0
         
         s_binned[int(i/step)-1] = s[i-(int(step/2))] #choose the midpoint- evenly spaced in ln-space
         counter_NGC = 0
         counter_SGC = 0
    
#s_binned, RR0_z1_SGC, RR2_z1_SGC = Test0(RR0_z1_SGC, RR2_z1_SGC)
#s_binned, RR0_z2_NGC, RR2_z2_NGC = Test0(RR0_z2_NGC, RR2_z2_NGC)
#s_binned, RR0_z2_SGC, RR2_z2_SGC = Test0(RR0_z2_SGC, RR2_z2_SGC)
#s_binned, RR0_z3_NGC, RR2_z3_NGC = Test0(RR0_z3_NGC, RR2_z3_NGC)
#s_binned, RR0_z3_SGC, RR2_z3_SGC = Test0(RR0_z3_SGC, RR2_z3_SGC)

#Now, we plot these normalised Q_0 and Q_2 terms:
    
fig = plt.figure(1, (7,7))
ax = fig.add_subplot(1,1,1)
figure = plt.gcf()
figure.set_size_inches(10, 8)

matplotlib.rcParams.update({'font.size': 20})
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
#plt.rc('axes', labelsize=20)
#plt.rc('axes', titlesize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.semilogx(s_binned, RR0_NGC_binned, '-', label=r'$RR_{0} \ NGC z3$')
ax.semilogx(s_binned, RR2_NGC_binned, '-', label=r'$RR_{2} \ NGC z3$')
ax.semilogx(s_binned, RR0_SGC_binned, '-', label=r'$RR_{0} \ SGC z3$')
ax.semilogx(s_binned, RR2_SGC_binned, '-', label=r'$RR_{2} \ SGC z3$')
#ax.semilogx(s_binned, RR0_z1_SGC, '-', label=r'$RR_{0} \ SGC z1$')
#ax.semilogx(s_binned, RR2_z1_SGC, '-', label=r'$RR_{2} \ SGC z1$')
#ax.semilogx(s_binned, RR2_binned, '-', label=r'$Q_{2} \ SGC z3$')
#ax.semilogx(s_binned, RR4_binned, '-', label=r'$Q_{4} \ SGC z3$')
ax.legend(prop={'size':20})
plt.ylim(-0.7, 1.1)
plt.xlim(1, 4000)
plt.savefig("z3_NGC_SGC_ConfigSpace_Comparison.png", dpi=500)
plt.show()

#Test 1: Qq(s) = RR/s^3 w/ normalisation of Q_0(s->0)
'''
counter = 0

for i in range(length):
    RR0[i] = RR0[i]/(s[i]**3)
    RR2[i] = RR2[i]/(s[i]**3)
    RR4[i] = RR4[i]/(s[i]**3)
    RR6[i] = RR6[i]/(s[i]**3)
    RR8[i] = RR8[i]/(s[i]**3)
    if (RR0[i] != 0) & (counter == 0) & (s[i] > 1):
        Norm = RR0[i]
        print("Norm is %lf" %Norm)
        print(RR0[i]/Norm)
        counter = 1

#Now, we normalise each of these terms        

RR0_sum = 0

for i in range(length):
    RR0_sum = RR0_sum + RR0[i]
    RR0[i] = (RR0[i])/Norm
    RR2[i] = RR2[i]/Norm
    RR4[i] = RR4[i]/Norm
    RR6[i] = RR6[i]/Norm
    RR8[i] = RR8[i]/Norm
'''
'''
#Test 2: Plot RR_0 and RR_2 vs W as in Fig 3 of Beutler 2016

fig = plt.figure(1, (7,7))
ax = fig.add_subplot(1,1,1)
figure = plt.gcf()
figure.set_size_inches(10, 8)

matplotlib.rcParams.update({'font.size': 20})
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
#plt.rc('axes', labelsize=20)
#plt.rc('axes', titlesize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.semilogx(s, RR0, '.', label=r'$RR_{0}$')
ax.semilogx(s, RR2, '.', label=r'$RR_{2}$')
#ax.semilogx(s, RR4, '-', label=r'$RR_{4}$')
#ax.semilogx(s, RR6, '-', label=r'$RR_{6}$')
ax.legend(prop={'size':20})
plt.ylim(-0.6, 1.2)
plt.xlim(1, 4000)
plt.show()
'''