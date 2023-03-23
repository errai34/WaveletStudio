import argparse
import sys, os
from pathlib import Path
import json
import logging as log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from alibi.explainers import KernelShap
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

import torch.nn as nn
import torch
import torch.optim as optim
import torch.fft as torch_fft

import seaborn as sns

import html

from analysis import get_artist_cache

class ArtAI:
    def __init__(self, config, artist):
        self.config = config
        self.artist = artist
        self.__embedding = None
        self.__load_coeff(config["scattering"], artist)

    def __load_coeff(self, path, artist):
        savename = path.parent / (path.name + "_df_rgb0.csv")
        if savename.exists():
            log.info("Loading saved data frame from %s" % savename)
            self.df = pd.read_csv(savename)
            return

        compiled_coeff = []
        img_names = []

        for x in path.iterdir():
            print(x)
            compiled_coeff.append(np.load(x)[0, :])
            img_names.append(x.name[0 : x.name.find("_size")])
        compiled_coeff = np.array(compiled_coeff)
        print(compiled_coeff.shape)

        target_names = [
            artist if artist.lower() in x.name.lower() else "Not_" + artist
            for x in path.iterdir()
        ]
       

        cols = list(map(lambda x: "Coeff_{}".format(x), range(compiled_coeff.shape[-1])))


        df = pd.DataFrame(compiled_coeff, columns=cols)
        df["target_names"] = target_names

        class_labels = np.zeros(len(df['target_names']))
        for i in range(len(df["target_names"])):
            if df["target_names"][i] == artist:
                class_labels[i] = 1
            else:
                class_labels[i] = 0

        df["class_labels"] = class_labels
        df["img_names"] = img_names

        log.info("Saving data frame to %s" % savename)
        df.to_csv(savename, index=False)
        self.df = df


def cmd_parse():
    description = """
    Small functionality that allows us to save the scattering coefficients in a nice data format, the pandas DataFrame.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", "-c", help="set the configuration file path")
    parser.add_argument(
        "--artist",
        "-a",
        help="""the artist inc. 'cardogan', 'monet', 'vincent', "Canaletto". This argument is
                necessary for Monet and van Gogh as the filenames are very different
                to what we consider the default naming convention as for the
                Cardogan artists, i.e. artwork_name_size-widthcmxheightcm.""",
    )
    parser.add_argument("--log", "-l", help="Log level")

    # Read arguments from the command line
    return parser.parse_args()

if __name__ == "__main__":
    args = cmd_parse()
    llevel = getattr(log, args.log.upper(), None)
    if not isinstance(llevel, int):
        raise ValueError("Invalid log level: %s" % args.log)
    log.basicConfig(level=getattr(log, args.log.upper(), None))

    with open(args.config, "r") as f:
        config = json.load(f)
    get_artist_cache(config, args.artist)
    
    global ai
    ai = ArtAI(config, args.artist)
