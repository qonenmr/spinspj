import os
import numpy
import math
from scipy.signal import hilbert
import tensorflow as tf
import keras.backend as K
from r import DLNMR, reconstruction

# Add cos window for one data
def getCosWdw(at, pointNum):
    """
    :param data：fid
    :return Numpy array
    """
    sb = 1
    sbs = 0

    tUnit = at / pointNum
    factor = math.pi / 2.0 / (sb * at)
    currentT = 0

    value = numpy.empty(pointNum)
    for i in range(pointNum):
        value[i] = math.cos((currentT - sbs) * factor)
        currentT += tUnit

    return value


# Add cos window for one data
def phaseCorr(data, ph0, ph1):
    """
    :param data：data
    :param ph0: 0-order phase
    :param ph1: 1-order phase
    :return Numpy array
    """

    spec = numpy.asarray(data)
    np = spec.size

    a_num = -((numpy.arange(1, np + 1))) / (np)
    aa = numpy.exp((1j * numpy.pi / 180 * (ph0 + ph1 * a_num)))
    spec = numpy.multiply(spec, aa)

    return spec


def getContour(maxValue, minFactor, n):
    minValue = maxValue * minFactor
    contour = numpy.empty(n)
    currValue = minValue
    for i in range(n):
        contour[i] = currValue
        currValue *= 1.16
    return contour


# 3D reconstruction function
def reconProcess(fid, samplingRate):
    """
    :param fid: Original FID
    :param samplingRate: Extraction rate
    :return Reconstructed FID
    """
    # Sampling policy file location
    nuslistPath = "system/data/pyexample/nus/nuslist-" + str(samplingRate) + "%"
    # The location of the model file to be loaded is changed according to the sampling rate
    weight_path = (
        r"system/data/pyexample/nus/Model/Weight_DLNMR/HyperComplex/Weight-SR"
        + str(samplingRate)
        + ".h5"
    )
    fidC = fid
    fidTmp = numpy.zeros((120 * 120, 366))
    for i in range(120):
        for j in range(120):
            fidTmp[i * 120 + j] = numpy.real(fidC[i][j])

    # Get mask through nuslist
    mask = []
    with open(nuslistPath) as fd:
        while True:
            lineTmp = fd.readline().strip("\n")
            if not lineTmp:
                break
            else:
                mask.append(list(map(int, lineTmp.split(" "))))

    # Get sampling strategy through mask
    maskTmp = numpy.zeros((60, 60))
    for i in range(len(mask)):
        maskTmp[mask[i][0], mask[i][1]] = 1

    acquiredPtF1_2, acquiredPtF1_1 = numpy.nonzero(maskTmp)

    mask_2D_ZK = numpy.zeros((maskTmp.shape[0] * 2, maskTmp.shape[1] * 2))
    mask_2D_ZK[acquiredPtF1_2 * 2, acquiredPtF1_1 * 2] = 1

    mask_2D_ZK[acquiredPtF1_2 * 2 + 1, acquiredPtF1_1 * 2] = 1
    mask_2D_ZK[acquiredPtF1_2 * 2, acquiredPtF1_1 * 2 + 1] = 1
    mask_2D_ZK[acquiredPtF1_2 * 2 + 1, acquiredPtF1_1 * 2 + 1] = 1

    acquiredPtF1_2_ZK, acquiredPtF1_1_ZK = numpy.nonzero(mask_2D_ZK)
    acquiredPtF1 = acquiredPtF1_2_ZK * mask_2D_ZK.shape[1] + acquiredPtF1_1_ZK

    # sampling
    data_nus = fidTmp[acquiredPtF1, ...]

    # Loading model
    GPU_index = "0"
    GPU_used = str(GPU_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_used
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sessTmp = tf.Session(config=config)
    K.set_session(sessTmp)
    DL_NMR = DLNMR(model_kind="HyperComplex", weight_path=weight_path, verbosity=2)

    dimension = 3
    targetFidPt = (366, 120, 120)
    fnModeF1 = 4
    number_split = 32
    fnModeF2 = None

    # reconstruction
    DL_NMR_rec = reconstruction(
        DL_NMR,
        data_nus,
        dimension,
        targetFidPt,
        acquiredPtF1,
        fnModeF1,
        fnModeF2,
        number_split,
        verbosity=2,
    )

    DL_NMR_rec_RR = numpy.transpose(DL_NMR_rec[:, :, :, 0], axes=(2, 1, 0))
    DL_NMR_rec_RI = numpy.transpose(DL_NMR_rec[:, :, :, 1], axes=(2, 1, 0))
    DL_NMR_rec_IR = numpy.transpose(DL_NMR_rec[:, :, :, 2], axes=(2, 1, 0))
    DL_NMR_rec_II = numpy.transpose(DL_NMR_rec[:, :, :, 3], axes=(2, 1, 0))

    DL_NMR_rec_ZK = numpy.zeros(targetFidPt[::-1])
    DL_NMR_rec_ZK[0::2, 0::2, :] = DL_NMR_rec_RR
    DL_NMR_rec_ZK[1::2, 0::2, :] = DL_NMR_rec_RI
    DL_NMR_rec_ZK[0::2, 1::2, :] = DL_NMR_rec_IR
    DL_NMR_rec_ZK[1::2, 1::2, :] = DL_NMR_rec_II

    DL_NMR_rec_ZK = DL_NMR_rec_ZK.flatten()
    DL_NMR_rec_ZK = DL_NMR_rec_ZK.astype(">f4")

    fidC = numpy.zeros((120, 120, 366), dtype=complex)
    for i in range(120):
        for j in range(120):
            fidC[i][j] = DL_NMR_rec_ZK[
                i * 120 * 366 + j * 366 : i * 120 * 366 + (j + 1) * 366
            ]

    # Hilbert transform
    for i in range(120):
        for j in range(120):
            fidC[i][j] = hilbert(numpy.real(fidC[i][j]))
    return fidC


# Get the spectrum peak
def getPeaks(spec, threshold):
    peaksArray = []
    peaksIndexArray = []
    # Circular array
    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            value = spec[i][j]
            # A value is greater than the threshold and is not around
            if (
                value > threshold
                and i > 2
                and i < spec.shape[0] - 2
                and j > 2
                and j < spec.shape[1] - 2
            ):
                # A value is greater than the value of the surrounding field
                if (
                    value > spec[i - 1][j]
                    and value > spec[i + 1][j]
                    and value > spec[i][j - 1]
                    and value > spec[i][j + 1]
                    and value > spec[i - 2][j]
                    and value > spec[i + 2][j]
                    and value > spec[i][j - 2]
                    and value > spec[i][j + 2]
                ):
                    peaksArray.append(value)
                    peaksIndexArray.append([i, j])
    return peaksArray, peaksIndexArray


# FID variation spectrum
def fidToSpec(fid, si, si1, si2):
    spec = numpy.zeros((si2 * 2, si1 * 2, si), dtype=complex)  # 128*128*366
    spec[0:120, 0:120, :] = fid

    acqtime = 0.091
    acqtime1 = 0.015
    acqtime2 = 0.019

    # The first dimension is indirect
    wdwValue1 = getCosWdw(acqtime1, si1)
    for i in range(si2 * 2):
        for j in range(si):
            # Data reorganization (obtain two one-dimensional arrays with length of SI1)
            tempReal = spec[i, 0 : si1 * 2 : 2, j].real + 1j * (
                spec[i, 1 : si1 * 2 : 2, j].real
            )
            tempImag = spec[i, 0 : si1 * 2 : 2, j].imag + 1j * (
                spec[i, 1 : si1 * 2 : 2, j].imag
            )
            # Add windows
            tempReal = numpy.multiply(tempReal, wdwValue1)
            tempImag = numpy.multiply(tempImag, wdwValue1)
            # FT
            tempReal = numpy.fft.fft(tempReal)
            tempReal = numpy.fft.fftshift(tempReal)
            tempImag = numpy.fft.fft(tempImag)
            tempImag = numpy.fft.fftshift(tempImag)
            # Add phase
            ph0 = 0
            ph1 = 1
            tempReal = phaseCorr(tempReal, ph0, ph1)
            tempImag = phaseCorr(tempImag, ph0, ph1)
            # Data reorganization
            spec[i, 0:si1, j] = tempReal.real + 1j * (tempImag.real)
            spec[i, si1 : si1 * 2, j] = tempReal.imag + 1j * (tempImag.imag)

    # Second indirect dimension：
    wdwValue2 = getCosWdw(acqtime2, si2)
    for i in range(si1 * 2):
        for j in range(si):
            # States mode:
            # Data reorganization (obtaining two one-dimensional arrays of SI2 length)
            tempReal = spec[0 : si2 * 2 : 2, i, j].real + 1j * (
                spec[1 : si2 * 2 : 2, i, j].real
            )
            tempImag = spec[0 : si2 * 2 : 2, i, j].imag + 1j * (
                spec[1 : si2 * 2 : 2, i, j].imag
            )
            # Add windows
            tempReal = numpy.multiply(tempReal, wdwValue2)
            tempImag = numpy.multiply(tempImag, wdwValue2)
            # FT
            tempReal = numpy.fft.fft(tempReal)
            tempImag = numpy.fft.fft(tempImag)
            tempReal = numpy.fft.fftshift(tempReal)
            tempImag = numpy.fft.fftshift(tempImag)
            # Add phase
            ph0 = 0
            ph1 = 0
            tempReal = phaseCorr(tempReal, ph0, ph1)
            tempImag = phaseCorr(tempImag, ph0, ph1)
            # Data reorganization
            spec[0:si2, i, j] = tempReal.real + 1j * (tempImag.real)
            spec[si2 : si2 * 2, i, j] = tempReal.imag + 1j * (tempImag.imag)
    return spec


# Calculate the correlation coefficient
def calc_corr1(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    # Calculate the molecular covariance
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    # Calculate denominator, variance product
    sq = math.sqrt(
        sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b])
    )
    corr_factor = cov_ab / sq
    return corr_factor
