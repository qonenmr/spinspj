from command import spinspj
import numpy as np
import scipy
from scipy.optimize import minimize
def evaluate(x):
    spinspj.setshimvalue('z1',round(x[0]))
    spinspj.setshimvalue('z2',round(x[1]))
    spinspj.setshimvalue('z3',round(x[2]))
    spinspj.go()
    fid = spinspj.getFid()
    npfid = np.array(fid)
    fidArea = sum(abs(npfid[0][0][:]))
    print('z1=',round(x[0]),'z2=',round(x[1]),'z3=',round(x[2]),'fidArea:', round(fidArea))
    return 1/fidArea
  
path=spinspj.getExamplePath()+'Proton.nmr'
spinspj.setWs(path)
res = scipy.optimize.minimize(evaluate, [4300, -10000,10000], method="Nelder-Mead",options={'xtol': 2, 'disp': True})
print(res)
