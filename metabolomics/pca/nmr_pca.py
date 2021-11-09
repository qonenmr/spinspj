from command import spinspj
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy
import os
from pca import pcaUtils

rootPath = os.getcwd()
# Experiment folder
folderPath = os.path.join(rootPath, r"./system/data/pyexample/pca/Proton_1D_RAW")
# Spectral peak alignment interval file
peakAligmentFile = os.path.join(rootPath, r"./system/data/pyexample/pca/configure/peakAligment")
# Exclude integral interval file
integralExcludeFile = os.path.join(rootPath, r"./system/data/pyexample/pca/configure/integralExclude")

# Experiment file name
fileList = os.listdir(folderPath)
# Preliminary treatment
specData = []	# Storing spectral data
# Traverse experimental files
for fileName in fileList:
	# Splicing path
	filePath = folderPath + r"/" + fileName
	print("Processing：" + filePath)
	# Set workspace
	spinspj.setWs(filePath)
	# Experimental treatment
	spinspj.wft()	# Fourier
	spinspj.aph()	# Phase correction
	spinspj.abs()	# Baseline correction
	spinspj.updateSpecShow()	# Update spectrum
    # get spectrum data
	specDataTmp=spinspj.getSpec()
	for i in range(len(specDataTmp)):
		specData.append(specDataTmp[i][0])

# Spectral peak alignment
specData = pcaUtils.formatData(specData).tolist()
peakAligment = pcaUtils.readPeakAligmentFile(peakAligmentFile)
specData = spinspj.peaksAligment([3,0,1],specData,1,peakAligment)
specData = pcaUtils.formatData(specData)

# Set the spectrum and integrate
centerFreq = []	# center frequency
relIntensity = [] # Integral value
for i in range(len(fileList)):
	filePath = folderPath + r"/" + fileList[i]
	print("Processing：" + filePath)
	spinspj.setWs(filePath)
	real, imag = pcaUtils.getRealImag(specData[i,:])
	# Set new spectrum
	spinspj.setSpec(real,imag)
	spinspj.updateSpecShow()
	exclude = pcaUtils.readBatchIntegrationFile(integralExcludeFile)
	spinspj.batchIntegration([8.9604,-0.040229],exclude,2843,200000)
	# Update integral
	spinspj.updateIntegrationShow()
	# get center frequency and integral value
	centerFreq.append(spinspj.getCenterFreqOfIntegrals())
	relIntensity.append(spinspj.getRefValueOfIntegrals())
# normalization
pcaData = pcaUtils.normalization(relIntensity)

#Uncomment following 2 lines to get exact the same result of the website
# filePath = r"./system/data/pyexample/ST000101.txt"
# pcaData = spinspj.loadmetabolicdata(filePath,1,3);

data = numpy.transpose(pcaData)
rownum = data.shape[0]
colnum = data.shape[1]

for k in range(rownum):
    m=numpy.mean(data[k,:])
    stddata = numpy.std(data[k,:])
    data[k,:] = (data[k,:] - m)/stddata

data2 = numpy.transpose(data)    
pca1=PCA(n_components=8)
newData=pca1.fit(data2)
print(pca1.explained_variance_ratio_)
print(pca1.explained_variance_)
eig=pca1.explained_variance_
xcomp = [1, 2, 3, 4, 5, 6, 7, 8]

plt.figure(1)
plt.bar(xcomp, pca1.explained_variance_ratio_[0:8], label='', color='blueviolet')
plt.xlabel('principal component')
plt.ylabel('R2X')
plt.title('principal component analysis')
plt.show()

plt.figure(2)
fitted_data = pca1.fit_transform(data.T)        # numpy.ndarray
plt.scatter(fitted_data[:, 0], -fitted_data[:, 1],marker='x')

for i in range(1,6):
    plt.annotate("Mixture B" + str(i), xy=(fitted_data[:, 0][i-1]+2, -fitted_data[:, 1][i-1]), color='black')
for i in range(1,6):
    plt.annotate("Mixture A" + str(i), xy=(fitted_data[:, 0][i+4]+2, -fitted_data[:, 1][i+4]), color='red')

plt.xlim((-50,80))
plt.ylim((-50,30))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('score scatter plot')
plt.show()
print('over')