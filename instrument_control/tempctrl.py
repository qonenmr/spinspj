from command import spinspj

#set active workspace as the current workspace of python
spinspj.setActiveWs()
#control the temerature to 30 degree
spinspj.vartemp(30)
#acquisition
spinspj.go()
#update the display of FID
spinspj.updateFidShow()
#stop to control the temperature
spinspj.stopvartemp()
