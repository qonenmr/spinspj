'''Purpose: The fully automated experiment'''

from command import spinspj
import numpy as np

#set active workspace
path=spinspj.getExamplePath()+'automation.nmr'
spinspj.setWs(path)
spinspj.aij(2)#Inject sample 2
spinspj.setsolvent('D2O')#Set the solvent of sample 2
spinspj.stm('H1')#Automatic tuning and matching
spinspj.smartmapshim3d('H1',15,20,200)#3D gradient shimming
spinspj.alock()#Automatic locking
spinspj.go()#Acquisition
spinspj.wft()#Window + FFT
spec = spinspj.getSpec()#Read spectrum data
fid = spinspj.getFid()#Read FID data
x = np.array(spec)#Transform spectrum data as ndarray
y = np.array(fid)#Transform FID data as ndarray
spinspj.updateSpecShow()#Update the display of spectrum data
spinspj.openWs()
