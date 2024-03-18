import os
import cv2
import numpy as np
import scipy.io as sio
from numpy import shape
from scipy.sparse import linalg
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")


def readImofDir(theRoad):
    X = np.zeros([512, 512, 29])
    for root, dirs, files in os.walk(theRoad):
        for i in range(29):
            if files[0] == 'Thumbs.db':
                j = i + 3
            else:
                j = i + 2
            I = cv2.imread(theRoad+'/'+files[j])
            I = I.astype('float32')
            X[:, :, i] = np.mean(I, 2)
    return X


def np2Tensor(img, data_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(255 / data_range)

    return tensor


def Upsample(MSI, HSI, R, C, sf):
    p = 10
    mu = 1e-5
    HSI3 = reshape(np.moveaxis(HSI, 2, 0), (HSI.shape[2], -1))
    D, _, _ = linalg.svds(HSI3, p)
    D = D[:, 0:p]
    RD = np.dot(R, D)
    L1 = D.shape[1]
    nr = MSI.shape[0]
    nc = MSI.shape[1]
    L = HSI.shape[2]
    HSI_int = np.zeros((nr, nc, L))
    HSI_int[0::sf, 0::sf, :] = HSI
    FBmC = np.conj(C)
    FBs1 = np.tile(C[:, :, np.newaxis], (1, 1, L1))
    FBCs = np.tile(FBmC[:, :, np.newaxis], (1, 1, L))
    FBCs1 = np.tile(FBmC[:, :, np.newaxis], (1, 1, L1))
    HHH = ifft2((fft2(HSI_int) * FBCs))
    HHH1 = hyperConvert2D(HHH)
    MSI3 = reshape(np.moveaxis(MSI, 2, 0), (MSI.shape[2], -1))
    n_dr = nr // sf
    n_dc = nc // sf
    HR_load1 = cv2.resize(HSI, None, fx=sf, fy=sf, interpolation=cv2.INTER_CUBIC)
    V2 = np.dot(D.T, hyperConvert2D(HR_load1))
    CCC = np.dot(RD.T, MSI3) + np.dot(D.T, HHH1)
    C1 = np.dot(RD.T, RD) + mu * np.eye(D.shape[1])
    Lambda, Q = np.linalg.eig(C1)
    Lambda = reshape(Lambda, (1, 1, L1))
    InvLbd = 1 / np.tile(Lambda, (sf * n_dr, sf * n_dc, 1))
    B2Sum = PPlus(np.power(np.abs(FBs1), 2) / (sf * sf), n_dr, n_dc)
    InvDI = 1 / (B2Sum[0:n_dr, 0:n_dc, :] + np.tile(Lambda, (n_dr, n_dc, 1)))
    HR_HSI3 = mu * V2
    C3 = CCC + HR_HSI3
    C30 = fft2(reshape(np.dot(np.linalg.inv(Q), C3).T, (nr, nc, L1))) * InvLbd
    temp = PPlus_s(C30 / (sf * sf) * FBs1, n_dr, n_dc)
    invQUF = C30 - np.tile(temp * InvDI, (sf, sf, 1)) * FBCs1
    VXF = np.dot(Q, reshape(invQUF, (nr * nc, L1)).T)
    A = reshape(np.real(ifft2(reshape(VXF.T, (nr, nc, L1)))), (nr * nc, L1)).T
    A = reshape(A.T, (nr, nc, -1))
    return D, A


def hyperConvert2D(Image3D):
    h = Image3D.shape[0]
    w = Image3D.shape[1]
    numBands = Image3D.shape[2]
    Image2D = reshape(Image3D, (w * h, numBands)).T
    return Image2D


def PPlus(X, n_dr, n_dc):
    nr = X.shape[0]
    nc = X.shape[1]
    nb = X.shape[2]
    Temp = reshape(X, (nr * n_dc, nc // n_dc, nb))
    Temp[:, 0, :] = np.sum(Temp, 1)
    Temp1 = reshape(np.transpose(reshape(Temp[:, 0, :], (nr, n_dc, nb)), (1, 0, 2)), (n_dc * n_dr, nr // n_dr, nb))
    X[0:n_dr, 0:n_dc, :] = np.transpose(reshape(np.sum(Temp1, 1), (n_dc, n_dr, nb)), (1, 0, 2))
    return X


def PPlus_s(X, n_dr, n_dc):
    nr = X.shape[0]
    nc = X.shape[1]
    nb = X.shape[2]
    Temp = reshape(X, (nr * n_dc, nc // n_dc, nb))
    Temp[:, 0, :] = np.sum(Temp, 1)
    Temp1 = reshape(np.transpose(reshape(Temp[:, 0, :], (nr, n_dc, nb)), (1, 0, 2)), (n_dc * n_dr, nr // n_dr, nb))
    Y = np.transpose(reshape(np.sum(Temp1, 1), (n_dc, n_dr, nb)), (1, 0, 2))
    return Y


def fft2(x):
    return np.fft.fft2(x, axes=(0, 1))


def ifft2(x):
    return np.fft.ifft2(x, axes=(0, 1))


def reshape(x, axes):
    return np.reshape(x, axes, order="F")