import numpy

# Format data
def formatData(data):
	data = numpy.array(data)
	data = data.T
	return data

# Read spectral peak interval file
def readPeakAligmentFile(filePath):
	interList = []
	with open(filePath) as fd:
		lines = fd.readlines()
		for line in lines:
			interTmp = line.split()
			interTmp = list(map(int, interTmp))
			interList.append(interTmp)
	return interList

# Read excluded segment integral file
def readBatchIntegrationFile(filePath):
	exclude=[]
	with open(filePath) as fd:
		lines=fd.readlines()
		for line in lines:
			dataTmp=line.split()
			exclude.append([float(dataTmp[0]),float(dataTmp[1])])
	return exclude

# normalization
def normalization(data):
	data = numpy.array(data, "f4")
	# Calculate median
	medianArray = []
	dataTran = data.T
	for i in range(len(dataTran)):
		medianArray.append(numpy.median(dataTran[i]))
	# Median spectrum
	dataMediaArray = []
	for i in range(len(data)):
		dataMediaArray.append(data[i] / medianArray)
	# Calculate median
	medianArray.clear()
	for i in range(len(dataMediaArray)):
		medianArray.append(numpy.median(dataMediaArray[i]))
	# Final results
	dataRet = []
	for i in range(len(data)):
		dataRet.append(data[i] / medianArray[i])
	return dataRet

def getRealImag(data):
	real = data
	# get the real part and imaginary part of the spectrum
	real = numpy.asarray(real,"f4").reshape((1,len(real)))
	imag = numpy.zeros((1,len(real.T)))
	return real.tolist(), imag.tolist()