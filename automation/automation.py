'''Purpose: The fully automated experiment'''

from command import spinspj
import numpy as np

spinspj.setActiveWs()
spinspj.aij(2)#Inject sample 2
print('Sample 2 has been injected.\n')
spinspj.setsolvent('D2O')#Set the solvent of sample 2
print('Solvent has been set to D2O.\n')
spinspj.stm('H1')#Automatic tuning and matching
print('Smart tuning and matching is over.\n')
spinspj.smartmapshim(2000)#1D gradient shimming
print('Smart gradient shimming is over.\n')
spinspj.alock()#Automatic locking
print('Automatic locking is over.\n')
spinspj.go()#Acquisition
print('Data acquisition is over.\n')
spinspj.wft()#Window + FFT
fid = spinspj.getFid()#Read FID data
spec = spinspj.getSpec()#Read spectrum data
spinspj.updateSpecShow()#Update the display of spectrum data