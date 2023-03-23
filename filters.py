import numpy as np
import os
import torch.nn as nn
import torch
import torch.optim as optim
import torch.fft as torch_fft
import time
from sklearn.preprocessing import StandardScaler


class FiltersSet:
    def __init__(self, savedir=None):
        self.savedir = savedir
        self.filters = {}

    def __call__(self, M, N, J, L, precision="single", extracted=False):
        print('filter demanded', M, N, J, L)
        fname = self.__filter_name(M, N, J, L, precision)
        print('demanded filter name', fname)
        if fname in self.filters:
            print('returning filter from memory')
            return (self.filters[fname], J, L) if not extracted else self.__extract_filters(M, N, J, L, self.filters[fname])

        path = self.savedir / fname if self.savedir is not None else None
        if path is not None and path.exists():
            print('loading filter from', path)
            self.filters[fname] = np.load(path, allow_pickle=True).item()
            return (self.filters[fname], J, L) if not extracted else self.__extract_filters(M, N, J, L, self.filters[fname])

        self.filters[fname] = self.__generate_morlet(M, N, J, L, precision)
        print('generated filter', self.filters[fname])
        if self.savedir is not None:
            np.save(path, self.filters[fname])
        return (self.filters[fname], J, L) if not extracted else self.__extract_filters(M, N, J, L, self.filters[fname])

    def __extract_filters(self, M, N, J, L, fil):
        dtype = fil["psi"][0][0].dtype
        filters_set = torch.zeros((J, L, M, N), dtype=dtype)
        for j in range(J):
            for l in range(L):
                filters_set[j, l] = fil["psi"][j * L + l][0]
        return filters_set

    def __generate_morlet(self, M, N, J, L, precision):
        psi = []
        for j in range(J):
            for theta in range(L):
                wavelet = self.morlet_2d(
                    M=M,
                    N=N,
                    sigma=0.8 * 2 ** j,
                    theta=(int(L - L / 2 - 1) - theta) * np.pi / L,
                    xi=3.0 / 4.0 * np.pi / 2 ** j,
                    slant=4.0 / L,
                )
                wavelet_Fourier = np.fft.fft2(wavelet)
                wavelet_Fourier[0, 0] = 0
                if precision == "double":
                    psi.append([torch.from_numpy(wavelet_Fourier.real)])
                if precision == "single":
                    psi.append(
                        [torch.from_numpy(wavelet_Fourier.real.astype(np.float32))]
                    )

        filters_set_mycode = {"psi": psi}
        return filters_set_mycode

    def __filter_name(self, M, N, J, L, precision):
        return "filters_set_M{}N{}J{}L{}_{}.npy".format(M, N, J, L, precision)

    def morlet_2d(self, M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=False):
        """
        Computes a 2D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.

        Parameters
        ----------
        M, N : int
                                        spatial sizes
        sigma : float
                                        bandwidth parameter
        xi : float
                                        central frequency (in [0, 1])
        theta : float
                                        angle in [0, pi]
        slant : float, optional
                                        parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
                                        offset by which the signal starts
        fft_shift : boolean
                                        if true, shift the signal in a numpy style

        Returns
        -------
        morlet_fft : ndarray
                                        numpy array of size (M, N)
        """
        wv = self.gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
        wv_modulus = self.gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
        K = np.sum(wv) / np.sum(wv_modulus)

        mor = wv - K * wv_modulus
        return mor

    def gabor_2d(self, M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=False):
        """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.

        Parameters
        ----------
        M, N : int
                                        spatial sizes
        sigma : float
                                        bandwidth parameter
        xi : float
                                        central frequency (in [0, 1])
        theta : float
                                        angle in [0, pi]
        slant : float, optional
                                        parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
                                        offset by which the signal starts
        fft_shift : boolean
                                        if true, shift the signal in a numpy style

        Returns
        -------
        morlet_fft : ndarray
                                        numpy array of size (M, N)
        """
        gab = np.zeros((M, N), np.complex128)
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            np.float64,
        )
        R_inv = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
            np.float64,
        )
        D = np.array([[1, 0], [0, slant * slant]])
        curv = np.dot(R, np.dot(D, R_inv)) / (2 * sigma * sigma)

        for ex in [-2, -1, 0, 1, 2]:
            for ey in [-2, -1, 0, 1, 2]:
                [xx, yy] = np.mgrid[
                    offset + ex * M : offset + M + ex * M,
                    offset + ey * N : offset + N + ey * N,
                ]
                arg = -(
                    curv[0, 0] * np.multiply(xx, xx)
                    + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy)
                    + curv[1, 1] * np.multiply(yy, yy)
                ) + 1.0j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
                gab = gab + np.exp(arg)

        norm_factor = 2 * np.pi * sigma * sigma / slant
        gab = gab / norm_factor

        if fft_shift:
            gab = np.fft.fftshift(gab, axes=(0, 1))
        return gab
