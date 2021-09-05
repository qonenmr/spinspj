'''Purpose: Peak picking for 1D spectrum'''

from command import spinspj
import numpy as np

#set active workspace as workspace of CPython
spinspj.setActiveWs();
spinspj.ppa();
peaks = spinspj.getPeaks();
num=int(np.shape(peaks)[0]);
print("peak no "  + "      " + "ppm          " + "     intensity  ");
for i in range(0, num):
	print(str(i) + "       " + str(peaks[i][2]) + "     " +str(peaks[i][3]));
