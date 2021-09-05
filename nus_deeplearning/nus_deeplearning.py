import os
import sys

pathTmp = os.path.abspath(os.path.join(os.getcwd(), r"./system/data/pyexample/nus"))
sys.path.append(pathTmp)

from command import spinspj
import numpy
import matplotlib.pyplot as plt
from nus import pipe
from nus import ProcessUtil

if __name__ == "__main__":

    path = r"./system/data/pyexample/3DFid/Azurin" 
    fid = spinspj.readNUS3DFid(path)
    fid = numpy.asarray(fid) 
    
    np = 732  # float
    np1 = 120
    np2 = 120

    fid = fid[:, :, 0:732:2] + fid[:, :, 1:732:2] * 1j  # 120*120*366 complex

    si = 366
    si1 = 64
    si2 = 64
    spec = ProcessUtil.fidToSpec(fid, si, si1, si2)

    # Computational projection
    spceReal = numpy.real(spec[0:si2, 0:si1, :])
    specF3F2 = numpy.amax(spceReal, axis=0)
    specF3F1 = numpy.amax(spceReal, axis=1)
    specF2F1 = numpy.amax(spceReal, axis=2)

    # 3D reconstruction
    samplingRate = 20
    fid2 = ProcessUtil.reconProcess(fid, samplingRate)  # Raw FID, sample rate
    spec2 = ProcessUtil.fidToSpec(fid2, si, si1, si2)

    # Computational projection
    spceReal2 = numpy.real(spec2[0:si2, 0:si1, :])
    specF3F22 = numpy.amax(spceReal2, axis=0)
    specF3F12 = numpy.amax(spceReal2, axis=1)
    specF2F12 = numpy.amax(spceReal2, axis=2)

    # threshold
    threshold = 4000

    # select the spectral peak of specf3f2
    peaksArray, peaksIndex = ProcessUtil.getPeaks(specF3F2, threshold)
    peaksArray2 = []
    for x, y in peaksIndex:
        # Directly take the value of the same index position
        peaksArray2.append(specF3F22[x][y])

    print("Number of spectral peaks： ", str(len(peaksArray)))
    # Under the threshold of 4000, there are 295 peaks,
    # In descending order, take the first 200 peaks
    peaksArray = sorted(peaksArray, reverse=True)[0:200]
    peaksArray2 = sorted(peaksArray2, reverse=True)[0:200]
    corr = ProcessUtil.calc_corr1(peaksArray, peaksArray2)
    print("correlation coefficient:", str(corr))

    # Plot the coefficient of determination
    plt.figure(1)
    parameter = numpy.polyfit(peaksArray, peaksArray2, 1)  # Fitting polynomial
    p = numpy.poly1d(parameter)  # Fitting polynomial
    plt.subplot(1, 3, 1)
    # Scatter plot of original data
    plt.scatter(peaksArray, peaksArray2, s=1, c="#F15A25")
    # Draw fitting curve according to fitting polynomial
    plt.plot(peaksArray, p(peaksArray), color="#10ADE6", linewidth=2)
    # Description label (position of X coordinate, position of Y coordinate, label string)
    # R2 (coefficient of determination) = square of corr (correlation coefficient)
    plt.text(
        peaksArray[-1],
        peaksArray2[1],
        "R" + "\u00b2" + " = " + str(round(corr ** 2, 4)),
    )
    plt.title("specF3F2")
    plt.yscale("symlog")  # The X-axis coordinate is expressed in the form of power
    plt.xscale("symlog")  # The Y-axis coordinate is expressed in the form of power
    plt.xlabel("intensity(a.u., fully sampled)")
    plt.ylabel("intensity(a.u.," + str(samplingRate) + "% NUS)")

    # select the spectral peak of specF3F1
    peaksArray, peaksIndex = ProcessUtil.getPeaks(specF3F1, threshold)
    peaksArray2 = []
    for x, y in peaksIndex:
        # Directly take the value of the same index position
        peaksArray2.append(specF3F12[x][y])

    print("Number of spectral peaks： ", str(len(peaksArray)))
    # Under the threshold of 4000, there are 253 peaks,
    # In descending order, take the first 200 peaks
    peaksArray = sorted(peaksArray, reverse=True)[0:200]
    peaksArray2 = sorted(peaksArray2, reverse=True)[0:200]
    corr = ProcessUtil.calc_corr1(peaksArray, peaksArray2)
    print("correlation coefficient:", str(corr))

    # Plot the coefficient of determination
    parameter = numpy.polyfit(peaksArray, peaksArray2, 1)  # Fitting polynomial
    p = numpy.poly1d(parameter)  # Fitting polynomial
    plt.subplot(1, 3, 2)
    # Scatter plot of original data
    plt.scatter(peaksArray, peaksArray2, s=1, c="#F15A25")
    # Draw fitting curve according to fitting polynomial
    plt.plot(peaksArray, p(peaksArray), color="#10ADE6", linewidth=2)
    # Description label (position of X coordinate, position of Y coordinate, label string)
    # R2 (coefficient of determination) = square of corr (correlation coefficient)
    plt.text(
        peaksArray[-1],
        peaksArray2[1],
        "R" + "\u00b2" + " = " + str(round(corr ** 2, 4)),
    )
    plt.title("specF3F1")
    plt.yscale("symlog")  # The X-axis coordinate is expressed in the form of power
    plt.xscale("symlog")  # The Y-axis coordinate is expressed in the form of power
    plt.xlabel("intensity(a.u., fully sampled)")
    plt.ylabel("intensity(a.u.," + str(samplingRate) + "% NUS)")

    # select the spectral peak of specF2F1
    peaksArray, peaksIndex = ProcessUtil.getPeaks(specF2F1, threshold)
    peaksArray2 = []
    for x, y in peaksIndex:
        # Directly take the value of the same index position
        peaksArray2.append(specF2F12[x][y])

    print("Number of spectral peaks： ", str(len(peaksArray)))
    # Under the threshold of 4000, there are 177 peaks,
    # In descending order, take the first 150 peaks
    peaksArray = sorted(peaksArray, reverse=True)[0:150]
    peaksArray2 = sorted(peaksArray2, reverse=True)[0:150]
    corr = ProcessUtil.calc_corr1(peaksArray, peaksArray2)
    print("correlation coefficient:", str(corr))

    # Plot the coefficient of determination
    parameter = numpy.polyfit(peaksArray, peaksArray2, 1)  # Fitting polynomial
    p = numpy.poly1d(parameter)  # Fitting polynomial
    plt.subplot(1, 3, 3)
    # Scatter plot of original data
    plt.scatter(peaksArray, peaksArray2, s=1, c="#F15A25")
    # Draw fitting curve according to fitting polynomial
    plt.plot(peaksArray, p(peaksArray), color="#10ADE6", linewidth=2)
    # Description label (position of X coordinate, position of Y coordinate, label string)
    # R2 (coefficient of determination) = square of corr (correlation coefficient)
    plt.text(
        peaksArray[-1],
        peaksArray2[1],
        "R" + "\u00b2" + " = " + str(round(corr ** 2, 4)),
    )
    plt.title("specF2F1")
    plt.yscale("symlog")  # The X-axis coordinate is expressed in the form of power
    plt.xscale("symlog")  # The Y-axis coordinate is expressed in the form of power
    plt.xlabel("intensity(a.u., fully sampled)")
    plt.ylabel("intensity(a.u.," + str(samplingRate) + "% NUS)")
    # plt.show()

    # Drawing
    plt.figure(2)
    max1 = numpy.amax(specF3F2)
    contourValue1 = ProcessUtil.getContour(max1, 0.1, 15)
    ax1 = plt.subplot(1, 3, 1)
    x = numpy.linspace(11, 6, 366)  #X-axis range (start, end, number)
    y = numpy.linspace(166, 186, 64)  #Y-axis range
    X, Y = numpy.meshgrid(x, y)  #The XY axis generates gridlines
    contour1 = plt.contour(X, Y, specF3F2, contourValue1)  # 64*366
    plt.title("F3F2 Plane")  #Title
    plt.xlabel("H1 ppm X")  #X-axis name
    plt.ylabel("C13 ppm Y")  #Name of Y axis
    plt.gca().invert_xaxis()  #X-axis reverse
    plt.gca().invert_yaxis()  #Y-axis reverse
    ax1.yaxis.set_ticks_position("right")  #Y-axis on the right
    ax1.yaxis.set_label_position("right")  #Y-axis name on the right
    plt.clabel(contour1, inline=True, fontsize=10)

    max2 = numpy.amax(specF3F1)
    contourValue2 = ProcessUtil.getContour(max2, 0.1, 15)
    ax2 = plt.subplot(1, 3, 2)
    x2 = numpy.linspace(11, 6, 366)  #X-axis range (start, end, number)
    y2 = numpy.linspace(100, 140, 64)  #Y-axis range
    X2, Y2 = numpy.meshgrid(x2, y2)  #The XY axis generates gridlines
    contour2 = plt.contour(X2, Y2, specF3F1, contourValue2)
    plt.title("F3F1 Plane")
    plt.xlabel("H1 ppm X")
    plt.ylabel("N15 ppm Y")
    plt.gca().invert_xaxis()  #X-axis reverse
    plt.gca().invert_yaxis()  #Y-axis reverse
    ax2.yaxis.set_ticks_position("right")  #Y-axis on the right
    ax2.yaxis.set_label_position("right")  #Y-axis name on the right
    plt.clabel(contour2, inline=True, fontsize=10)

    max3 = numpy.amax(specF2F1)
    contourValue3 = ProcessUtil.getContour(max3, 0.1, 15)
    ax3 = plt.subplot(1, 3, 3)
    x3 = numpy.linspace(167, 185, 64)  #X-axis range (start, end, number)
    y3 = numpy.linspace(100, 140, 64)  #Y-axis range
    X3, Y3 = numpy.meshgrid(x3, y3)  #The XY axis generates gridlines
    contour3 = plt.contour(X3, Y3, specF2F1, contourValue3)
    plt.title("F2F1 Plane")
    plt.xlabel("C13 ppm X")
    plt.ylabel("N15 ppm Y")
    plt.gca().invert_xaxis()  #X-axis reverse
    plt.gca().invert_yaxis()  #Y-axis reverse
    ax3.yaxis.set_ticks_position("right") #Y-axis on the right
    ax3.yaxis.set_label_position("right")  #Y-axis name on the right
    plt.clabel(contour3, inline=True, fontsize=10)
    plt.show()

    print("end")