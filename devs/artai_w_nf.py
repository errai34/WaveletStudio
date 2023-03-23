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

import shap
import seaborn as sns
import umap
import html

from nflib.spline_flows import NSF_CL
from nflib.flows import (
    AffineConstantFlow,
    ActNorm,
    Invertible1x1Conv,
    NormalizingFlow,
    NormalizingFlowModel,
)

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
        class_labels = [0 if artist in x.name else 1 for x in path.iterdir()]
        cols = list(map(lambda x: "Coeff_{}".format(x), range(compiled_coeff.shape[-1])))

        df = pd.DataFrame(compiled_coeff, columns=cols)
        df["target_names"] = target_names
        df["class_labels"] = class_labels
        df["img_names"] = img_names

        log.info("Saving data frame to %s" % savename)
        df.to_csv(savename, index=False)
        self.df = df

    def plot_scattering_estimates(self):

        df_not_artist = self.df[self.df["class_labels"] == 1]

        df_artist = self.df[self.df["class_labels"] == 0]

        feat_data_not_artist = df_not_artist.drop(
            ["class_labels", "target_names", "img_names"], axis=1
        )
        feat_data_artist = df_artist.drop(
            ["class_labels", "target_names", "img_names"], axis=1
        )

        data_not_artist = feat_data_not_artist.to_numpy()
        data_artist = feat_data_artist.to_numpy()

        # -------------------------------------------------------------------------------------------------------
        # restore coefficients
        scattering_coeff_array_not_artist = data_not_artist
        scattering_coeff_array_not_artist = np.log10(scattering_coeff_array_not_artist)
        # plot 68, 95 confidence level

        scattering_coeff_array_artist = data_artist
        scattering_coeff_array_artist = np.log10(scattering_coeff_array_artist)

        fig = plt.figure(figsize=(10, 8))

        plt.fill_between(
            np.arange(scattering_coeff_array_not_artist.shape[1]),
            np.percentile(scattering_coeff_array_not_artist, 2.5, axis=0),
            np.percentile(scattering_coeff_array_not_artist, 97.5, axis=0),
            alpha=0.5,
            label="Not" + str(args.artist),
            color="blue",
        )

        plt.fill_between(
            np.arange(scattering_coeff_array_artist.shape[1]),
            np.percentile(scattering_coeff_array_artist, 2.5, axis=0),
            np.percentile(scattering_coeff_array_artist, 97.5, axis=0),
            alpha=0.5,
            label=str(args.artist),
            color="m",
        )

        # plot delimiter
        plt.plot(
             [self.config["J"] - 1, self.config["J"] - 1], [-3.2, 0.2], ls="--", lw=5, color="black"
         )

  
        plt.text(0, -0.1, "Strokes")
        plt.text(11.0, -0.1, "Clustering of Strokes")
        plt.xlim([-5, len(self.df.iloc[0, :]) + 5])

        plt.ylabel("Estimate")
        plt.xlabel("Statistical Descriptors")
        plt.legend(loc=4)

        plt.show()

    @property
    def embedding(self):
        if self.__embedding is not None:
            return self.__embedding

        savename = self.config["artist_cache"] / ("_umap_embedding.csv")
        if savename.exists():
            self.__embedding = pd.read_csv(savename)
            return self.__embedding

        # HOW TO PICK ONLY INTERPOLATED IMAGES
        df_features = self.df.copy()
        df_features = df_features.drop(
            ["class_labels", "target_names", "img_names"], axis=1
        )
        df_features.describe()

        stcoeffs = df_features.to_numpy()
        print(stcoeffs.shape)

        stcoeffs = StandardScaler().fit_transform(stcoeffs)

        reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, n_components=2)
        embd = reducer.fit_transform(stcoeffs)

        self.__embedding = pd.DataFrame()
        for i in range(embd.shape[1]):
            self.__embedding["Dim%d" % i] = embd[:, i]
        self.__embedding["img_names"] = self.df["img_names"].values
        
        self.__embedding.to_csv(savename)
        return self.__embedding


    def plot_umap(self):

        # only do it for the interpolated images....

        interp_filter = self.df["img_names"].str.startswith("interp")
        df_interp = self.df[interp_filter]
        labels = df_interp["class_labels"].values
        names = df_interp["img_names"].values

        clean_names = [x.replace("interp_", "") for x in names]
        fig, ax = plt.subplots(figsize=(8, 8))
        fembd = self.embedding[interp_filter].reset_index()

        ax.scatter(
            fembd["Dim0"],
            fembd["Dim1"],
            c=[sns.color_palette()[x] for x in labels],
            s=100,
        )

        for i, txt in enumerate(clean_names):
            ax.annotate(txt, (fembd["Dim0"][i], fembd["Dim1"][i]), fontsize=16)

        ax.set_xlabel("UMAP dimension 1", fontsize=18)
        ax.set_ylabel("UMAP dimension 2", fontsize=18)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])

        plt.show()

    # would be nice for this to be an abstract class, provide helper methods for data preparations
    # its inheritors would implement the train and predict_proba interface methods
    # or smth like that ¯\_(ツ)_/¯
    class GeneralClassifier:
        def __init__(self, artai):
            self.artai = artai
            self.__get_train_test_data()
            self.model = None
            self.clf_explainer = self.clf_explanation = None

        def __get_train_test_data(self):
            df = self.artai.df
            target = df.class_labels.to_numpy()  # class labels

            target_names = df.target_names.to_list()  # name
            feat_data = df.drop(["class_labels", "target_names", "img_names"], axis=1)
            feature_names = feat_data.columns.to_list()

            data = feat_data.to_numpy()

            X_train, X_test, self.y_train, self.y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0,
            )
            log.info("Training records: {}".format(X_train.shape[0]))
            log.info("Testing records: {}".format(X_test.shape[0]))

            self.scaler = StandardScaler().fit(X_train)
            self.X_train_norm = self.scaler.transform(X_train)
            self.X_test_norm = self.scaler.transform(X_test)

        def train(self):
            if self.model is not None:
                return self.model
            self.model = GaussianProcessClassifier(1 * Matern(2))  # can decide if we want to pick from a suite of them...

            self.model.fit(self.X_train_norm, self.y_train)

            np.random.seed(0)

            y_pred = self.model.predict(self.X_test_norm)
            cm = confusion_matrix(self.y_test, y_pred)
            title = "Confusion matrix for GP Classifier"
            disp = plot_confusion_matrix(
                self.model,
                self.X_test_norm,
                self.y_test,
                # display_labels=target_names,
                cmap=plt.cm.Blues,
                normalize=None,
            )
            disp.ax_.set_title(title)
            plt.show()

            return self.model

        def get_shap_explainer(self):
            if self.clf_explainer is not None and self.clf_explanation is not None:
                return self.clf_explainer, self.clf_explanation

            model = self.train()
            pred_fcn = model.predict_proba
            self.clf_explainer = KernelShap(pred_fcn)
            self.clf_explainer.fit(self.X_train_norm)

            self.clf_explanation = self.clf_explainer.explain(self.X_test_norm, l1_reg=False)
            return self.clf_explainer, self.clf_explanation

        def plot_shap_values(self, idx, class_idx):
            clf_explainer, clf_explanation = self.get_shap_explainer()
            instance = self.X_test_norm[idx][None, :]

            feat_data = self.artai.df.drop(['class_labels', 'target_names', 'img_names'], axis=1)
            feature_names  = feat_data.columns.to_list()

            force_plot = shap.force_plot(
                clf_explainer.expected_value[class_idx],
                clf_explanation.shap_values[class_idx][idx, :],
                instance,
                feature_names, show=False, matplotlib=True
             ).savefig('test_shap.png')


            # shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

            # return html.Iframe(srcDoc=shap_html,
            #            style={"width": "100%", "height": "200px", 
            #            "border": 0})

    def get_classifier(self):
        return self.GeneralClassifier(self)

    class NormalisingFlowGenerator:
        def __init__(self, artai):
            self.artai = artai
            self.__prep_data()

        def __prep_data(self):
            df_follower = self.artai.df[self.artai.df['class_labels'] == 0]
            df_artist = self.artai.df[self.artai.df['class_labels'] == 1]


            feat_data_follower = df_follower.drop(['class_labels', 'target_names', 'img_names'], axis=1)
            feat_data_canaletto = df_canaletto.drop(['class_labels', 'target_names', 'img_names'], axis=1)

            self.data_follower = feat_data_follower.to_numpy()
            self.data_artist = feat_data_canaletto.to_numpy()

        def __get_train_test_artist(self):
            target = self.data_artist.class_labels.to_numpy()  # class labels

            target_names = self.data_artist.target_names.to_list()  # name
            feat_data = self.data_artist.drop(["class_labels", "target_names", "img_names"], axis=1)
            feature_names = feat_data.columns.to_list()

            data = feat_data.to_numpy()

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0,
            )
            log.info("Training records: {}".format(self.X_train.shape[0]))
            log.info("Testing records: {}".format(self.X_test.shape[0]))

        def __configure(self):
            X = np.log10(self.X_train[:, :-1])

            X = torch.Tensor(X)
            print(X.shape)

            dim = X.shape[-1]
            print('New dim after adding the labels is', dim)

            # choose prior here
            base_mu, base_cov = torch.zeros(dim), torch.eye(dim)
            prior = MultivariateNormal(base_mu, base_cov)

            # configure the normalising flow
            nfs_flow = NSF_CL
            nflows = 1
            hidden_dim = 16 #50 dims here... probs neeed to increase

            device='cpu'
            flows = [nfs_flow(dim=dim, device=device, K=8, B=3, hidden_dim=hidden_dim) for _ in range(nflows)] #things to change> maybe more is needed??!
            convs = [Invertible1x1Conv(dim=dim, device=device) for _ in flows]
            norms = [ActNorm(dim=dim, device=device) for _ in flows]
            flows = list(itertools.chain(*zip(norms, convs, flows)))

            # initialise the model
            model = NormalizingFlowModel(prior, flows, device=device)

            #if torch.cuda.device_count() >1:
            #    print("Let's use", torch.cuda.device_count(), "GPUs!")
            #    model = nn.DataParallel(model)

            #assume this is de facto
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)  # todo tune WD



            return X_train, y_train, X_test, y_train

        # Training run
        #-------------------------------------------------------------------------------------

        def run_model(batch_size):
            # train_loader
            train_loader = torch.utils.data.DataLoader(
                X, batch_size=batch_size, shuffle=True, pin_memory=False)

            t0 = time()

            model.train()
            print("Started training")
            n_epochs = 100
            loss_history=[]

            for k in range(n_epochs):
                for batch_idx, data_batch in enumerate(train_loader):
                    
                    x = data_batch.to(device)
                    zs, prior_logprob, log_det = model(x)

                    logprob = prior_logprob + log_det
                    loss = -torch.mean(logprob)  

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_history.append(float(loss))

                if k % 10 == 0:
                    print("Loss at step k =", str(k) + ":", loss.item())
              

            t1 = time()
            print(f'Elapsed time: {t1-t0:.1f} s') 
            
            return model, loss_history

        def __call__(self):
            batch_size=20

            oom = False
            try:
                model, loss_history = run_model(batch_size)

            except RuntimeError: # Out of memory
                oom = True
                raise ValueError('Error!')

            if oom:
                for _ in range(batch_size):
                    model, loss_history = run_model(1)

        def plot_samples(self):
            with torch.no_grad():
                zs = model.sample(5000)
                z = zs[-1]
                z = z.detach().numpy()


            fig = plt.figure(figsize=(12, 10))

            plt.fill_between(np.arange(X.shape[1]),\
                             np.percentile(X,2.5,axis=0),np.percentile(X,97.5,axis=0),\
                             alpha=0.3, label="Canaletto",color='red', )

            plt.fill_between(np.arange(z.shape[1]),\
                             np.percentile(z,2.5,axis=0),np.percentile(z,97.5,axis=0),\
                             alpha=0.3, label="Sampled",color='k',)
            plt.legend()

        def plot_log_probs(self):
            zs, prior_logprob, log_det = model(X)

            logprob = prior_logprob + log_det

                        # define network
            X_test = data_follower
            print(X_test.shape)

            X_test = np.log10(X_test[:, :-1])

            X_test = torch.Tensor(X_test)
            print(X_test.shape)

            zs, prior_logprob, log_det = model(X_test)

            logprob1= prior_logprob + log_det

            
            X_test_can = np.log10(X_test_canaletto[:, :-1])

            X_test_can = torch.Tensor(X_test_can)
            X_test_can.size()

            zs, prior_logprob, log_det = model(X_test_can)

            logprob2= prior_logprob + log_det


            fig = plt.figure(figsize=(12, 8))

            plt.hist(logprob.detach().numpy(), color='r', label='Train Canaletto', histtype='step', density=True, linewidth=2, hatch='/')
            plt.hist(logprob1.detach().numpy(), color='b', label='Follower', histtype='stepfilled', density=True,  hatch='/')
            plt.hist(logprob2.detach().numpy(), color='m', label='Test Canaletto', histtype='step', density=True, linewidth=1, hatch='/')


            plt.legend(fontsize=16)

    def get_generator(self):
        return self.NormalisingFlowGenerator(self)

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
