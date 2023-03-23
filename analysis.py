import argparse
import sys, os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cycler import cycler
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch
import torch.optim as optim
import torch.fft as torch_fft

import cv2
import math
import slidingwindow as sw
from filters import FiltersSet


def rgb(r, g, b):
    return (float(r) / 256.0, float(g) / 256.0, float(b) / 256.0)


cb2 = [
    rgb(31, 120, 180),
    rgb(255, 127, 0),
    rgb(51, 160, 44),
    rgb(227, 26, 28),
    rgb(166, 206, 227),
    rgb(253, 191, 111),
    rgb(178, 223, 138),
    rgb(251, 154, 153),
]


def cmd_parse():
    description = """
    Small functionality that allows us to save the scattering coefficients in a nice data format, the pandas DataFrame.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", "-c", help="set the configuration file path")
    parser.add_argument(
        "--interpolated", "-i", help="set the path to the interpolated images"
    )
    parser.add_argument(
        "--artist",
        "-a",
        help="""the artist inc. 'cardogan', 'monet', 'vincent', "Canaletto". This argument is
                necessary for Monet and van Gogh as the filenames are very different
                to what we consider the default naming convention as for the
                Cardogan artists, i.e. artwork_name_size-widthcmxheightcm.""",
    )

    # Read arguments from the command line
    return parser.parse_args()


class Reconstructor:
    def __init__(self, J, L, window_size, fset, savedir=None):
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)

        self.J = J
        self.L = L
        self.window_size = window_size
        self.fset = fset
        self.savedir = savedir

    def __call__(self, ipath, j1, j2=None, overwrite=False):
        fwin = (
            self.savedir / ("%s_%d.rec.npy" % (ipath.stem, j1))
            if self.savedir is not None
            else None
        )
        print("Using reconstruction", fwin)
        if not overwrite and fwin is not None and fwin.exists():
            print("Reconstruction %s exists" % str(fwin))
            serialised = np.load(fwin, allow_pickle=True)
            return (
                ipath.stem,
                serialised[0],
                serialised[1],
                serialised[2],
                serialised[3],
            )

        image_torch = self.__prepare_artwork(ipath)
        windows, elements, coefs = self.__get_windows(image_torch, j1, j2)

        if self.savedir is not None:
            serialised = np.array([windows, elements, coefs, image_torch.shape])
            np.save(fwin, serialised)
        return ipath.stem, windows, elements, coefs, image_torch.shape

    # kill high frequency successively for faster calculations
    def __cut_high_k_off(self, data_f, j=None):

        M = data_f.shape[-2]
        N = data_f.shape[-1]
        dx = M // 2 ** j
        dy = N // 2 ** j
        result = torch.cat(
            (
                torch.cat((data_f[..., :dx, :dy], data_f[..., -dx:, :dy]), -2),
                torch.cat((data_f[..., :dx, -dy:], data_f[..., -dx:, -dy:]), -2),
            ),
            -1,
        )
        return result

    def __get_coeff_S1(self, res, j1):
        M = N = self.window_size
        data_f = torch_fft.fftn(res, dim=(-2, -1))

        # cut out high frequency
        filters_set = self.fset(M, N, self.J, self.L, extracted=True)
        if j1 >= 1:
            data_f_small = self.__cut_high_k_off(data_f, j1)
            wavelet_f = self.__cut_high_k_off(filters_set[j1], j1)
        else:
            data_f_small = data_f
            wavelet_f = filters_set[j1]

        data_f_small = self.__cut_high_k_off(data_f, j1)
        wavelet_f = self.__cut_high_k_off(filters_set[j1], j1)

        _, M1, N1 = wavelet_f.shape
        # print('comparison', data_f_small.shape, wavelet_f.shape)

        # convolve with the kernel
        I_1_temp = torch_fft.ifftn(
            data_f_small[None, :, :] * wavelet_f,
            dim=(-2, -1),
        ).abs()

        scat_coeff = I_1_temp.mean((-2, -1)) * M1 * N1 / M / N
        I_1_temp = torch.sum(I_1_temp, axis=0) * M1 * N1 / M / N

        return "%.3f" % torch.log10(torch.sum(scat_coeff)).cpu().numpy(), I_1_temp

    def __get_coeff_S2(self, res, j1, j2):
        if j2 > j1:
            if j1>=1:
                factor = j2-j1+1
            else:
                factor = j2

        M = N = self.window_size
        data_f = torch_fft.fftn(res, dim=(-2, -1))

        filters_set = self.fset(M, N, self.J, self.L, extracted=True)
        if j1 >= 1:
            data_f_small = self.__cut_high_k_off(data_f, j1)
            wavelet_f = self.__cut_high_k_off(filters_set[j1], j1)
        else:
            data_f_small = data_f
            wavelet_f = filters_set[j1]

        _, M1, N1 = wavelet_f.shape
        print('qweqwe', data_f_small.shape, wavelet_f.shape)
        I_1_temp = torch_fft.ifftn(data_f_small[None, :, :] * wavelet_f,dim=(-2, -1)).abs()
        I_1_temp_f = torch_fft.fftn(I_1_temp, dim=(-2, -1))

        I_1_temp_f_small = self.__cut_high_k_off(I_1_temp_f, factor)
        wavelet_f2 = self.__cut_high_k_off(filters_set[j2], j2)

        _, M2, N2 = wavelet_f2.shape
        print('asdasd', I_1_temp_f_small.shape, wavelet_f2.shape)
        I_2_temp =torch_fft.ifftn(I_1_temp_f_small[:, None, :, :]* wavelet_f2[None, :, :, :],dim=(-2, -1)).abs()
        I_2_temp = torch.sum(I_2_temp,axis=(0,1))*M2*N2/M/N
        coeff = I_2_temp.mean((-2,-1))*M2*N2/M/N

        return "%.3f" % torch.log10(torch.sum(coeff)).cpu().numpy(), I_2_temp

    def __prepare_artwork(self, ipath, rgb_channel=0):

        img = np.load(ipath, allow_pickle=True)
        img = (
            StandardScaler()
            .fit_transform(img.reshape(-1, img.shape[-1]))
            .reshape(img.shape)
        )

        image_torch = torch.from_numpy(img[:, :, rgb_channel])
        shape = image_torch.shape

        return image_torch

    def __get_windows(self, image_torch, j1, j2=None):
        windows = sw.generate(
            image_torch,
            sw.DimOrder.HeightWidthChannel,
            self.window_size, #2**J
            0.5,
        )  # < 1024, do cookie cutter,
        print('window count', len(windows))
        scattering_coeff_temp = []

        elements = []
        st_coeff_vals = []
        print('number of windows', len(windows), windows[0].w, windows[0].h, image_torch.shape)
        for i in range(len(windows)):
            res = torch.zeros(self.window_size, self.window_size)
            res[:image_torch.shape[0], :image_torch.shape[1]] = image_torch[windows[i].indices()]
            st_coeff, I_temp = self.__get_coeff_S1(res, j1) if j2 is None else self.__get_coeff_S2(res, j1, j2)
            st_coeff_vals.append(float(st_coeff))
            elements.append(I_temp.detach().numpy())

        return windows, elements, st_coeff_vals


class Stitcher:
    class Visualiser:
        def __init__(
            self, name, windows, elements, coefs, stitched, scat_coef, savedir=None
        ):
            self.name = name
            self.windows = windows
            self.elements = elements
            self.coefs = coefs
            self.stitched = stitched
            self.scat_coef = scat_coef
            self.savedir = savedir

        def plot_windows(self, j1=None, pix_fixed=None):
            n_component_x = len(set(map(lambda p: p.x, self.windows)))
            n_component_y = len(set(map(lambda p: p.y, self.windows)))

            f, axarr = plt.subplots(
                n_component_x,
                n_component_y,
                figsize=(n_component_y * 3, n_component_x * 3),
            )
            k = 0
            for j in range(n_component_y):
                for i in range(n_component_x):
                    flat_idx = i + j * n_component_x

                    axidx = []
                    if n_component_x > 1:
                        axidx.append(i)
                    if n_component_y > 1:
                        axidx.append(j)
                    axidx = tuple(axidx)
                    sc = axarr[axidx].imshow(
                        self.elements[flat_idx], cmap="Greys", vmin=0, vmax=1
                    )
                    axarr[axidx].set_xticklabels([])
                    axarr[axidx].set_yticklabels([])

                    t = axarr[axidx].text(
                        0.25,
                        0.85,
                        self.coefs[flat_idx],
                        color=cb2[3],
                        fontsize=20,
                        transform=axarr[axidx].transAxes,
                    )

                    t.set_bbox(dict(color="white", alpha=1.0))
                    plt.colorbar(sc, ax=axarr[axidx], fraction=0.038, pad=0.04)

            name = "Clustering at a scale " + (
                ("of " + str(2 ** j1 * pix_fixed) + " mm")
                if j1 is not None and pix_fixed is not None
                else ""
            )
            f.suptitle(name, fontsize=32, color="k")
            f.subplots_adjust(hspace=0.2, wspace=0.2)
            plt.show()

        def plot_stitched(self, j1=None, pix_fixed=None):

            f, ax = plt.subplots()
            sc = ax.imshow(self.stitched, cmap="Greys", vmin=0.0, vmax=1.0)
            name = "Clustering at a scale " + (
                ("of " + str(2 ** j1 * pix_fixed) + " mm")
                if j1 is not None and pix_fixed is not None
                else ""
            )
            t = ax.text(
                0.25,
                0.85,
                name + ":" + "\n" + "%.3f" % np.log10(self.scat_coef[j1]),
                color=cb2[3],
                fontsize=10,
                transform=ax.transAxes,
            )

            t.set_bbox(dict(color="white", alpha=1.0))
            plt.colorbar(sc, ax=ax, fraction=0.038, pad=0.04)

            f.suptitle(self.name, fontsize=16, color="k")
            if self.savedir is not None:
                f.savefig(self.savedir / (self.name + str(j1) + "_stitched.png"))

            plt.show()

    def __init__(self, J, scattering_path, savedir=None):
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
        self.J = J
        self.scattering_path = scattering_path
        self.savedir = savedir

    def __call__(self, j1, name, windows, elements, coefs, imgshape):
        stitched = self.__stitch(j1, name, windows, elements, imgshape)
        scats = self.__get_interp_scattering_coef(name)
        return self.Visualiser(
            name, windows, elements, coefs, stitched, scats, self.savedir
        )

    def __get_interp_scattering_coef(self, name, rgb_channel=0):
        path = self.scattering_path / ("interp_" + name + "_coef.npy")
        return np.load(path)[rgb_channel, :]

    def __stitch(self, j1, name, windows, elements, imgshape):
        sx = elements[0].shape[0] / windows[0].w
        sy = elements[0].shape[1] / windows[0].h
        print("sx sy", sx, sy)

        res = np.zeros((int(imgshape[0] * sx), int(imgshape[1] * sy)))
        for (win, cf) in zip(windows, elements):
            s = win.indices()
            ovxs = cf.shape[0] // 4 if s[0].start > 0 else 0
            ovys = cf.shape[1] // 4 if s[1].start > 0 else 0
            ovxf = cf.shape[0] // 4 if s[0].stop < imgshape[0] else 0
            ovyf = cf.shape[1] // 4 if s[1].stop < imgshape[1] else 0
            # print('overlapping pixels', ov)
            fheight = min(int(win.h*sx), cf.shape[0])
            fwidth = min(int(win.w*sy), cf.shape[1])
            res[
                int(s[0].start * sx) + ovxs : int(s[0].start * sx) + fheight - ovxf,
                int(s[1].start * sy) + ovys : int(s[1].start * sy) + fwidth - ovyf,
            ] = cf[ovxs:fheight-ovxf, ovys:fwidth-ovyf]
        return res


def get_artist_cache(config, artist):
    cache = Path(config["cache"])
    for x in cache.iterdir():
        if x.name.lower().startswith(artist.lower()) and x.is_dir():
            config["artist_cache"] = x
            break

    config["reconstructions"] = config["artist_cache"] / "reconstructions"
    type_interp = "fixed_" + (
        "grid"
        if config["do_grid"]
        else ("pix_%smm" % str(config["pix_fixed"]).replace(".", "_"))
    )
    J, L = config["J"], config["L"]
    config["scattering"] = config["artist_cache"] / (
        ("reduced" if config["do_reduced"] else "full")
        + "_l_scattering_"
        + type_interp
        + str(J)
        + "_"
        + str(L)
        + "_"
        + str(config["window_size"]) # BIG CHANGE HERE!
    )
    print("scattering pathlib", config["scattering"])


def reconstruct_interpolated(config, ipath):
    if "ow_rec" not in config:
        config["ow_rec"] = False
    rec = Reconstructor(
        config["J"],
        config["L"],
        config["window_size"],
        FiltersSet(Path(config["filters"])),
        Path(config["reconstructions"]),
    )
    stit = Stitcher(config["J"], config["scattering"], Path(config["reconstructions"]))
    for x in ipath.iterdir():
        j1 = config["j1"]
        j2 = config["j2"] if "j2" in config else None
        name, windows, elements, coefs, imgshape = rec(x, j1, j2, overwrite=config["ow_rec"])
        stit(j1, name, windows, elements, coefs, imgshape).plot_stitched(
            j1,  config["pix_fixed"]
        )  # .plot_windows(1, config['pix_fixed'])
        print("windows for", x, windows)


if __name__ == "__main__":
    args = cmd_parse()
    with open(args.config, "r") as f:
        config = json.load(f)
    get_artist_cache(config, artist)
    reconstruct_interpolated(config, Path(args.interpolated))
