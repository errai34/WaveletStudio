import argparse
import sys, os
import json
from pathlib import Path

from curation import ArtWork, Interpolate
from filters import FiltersSet
from scattering import ImageScatterer
from albumify import Albumify

sys.path.append("./")

description = """
This is a program that takes in a Path variable where our
artwork (wish it was our own haha!) and processes it (i.e. gets its width,
height, unit) and then interpolates it as we wish. This is built as part of
HephaestusAI, a framework which will be used by Haephaestus Analytical to detect
forgery.
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--path", "-p", help="set the path where the images are")
parser.add_argument(
    "--artist",
    "-a",
    help="""the artist inc. 'cardogan', 'monet', 'vincent', "Caneletto". This argument is
			necessary for Monet and van Gogh as the filenames are very different
			to what we consider the default naming convention as for the
			Cardogan artists, i.e. artwork_name_size-widthcmxheightcm.""",
)
parser.add_argument("--config", "-c", help="Parameters configuration file")

# Read arguments from the command line
args = parser.parse_args()
# Check for --path


if args.artist:
    print("Set artist variable to %s" % args.artist)

if args.path:
    print("Set path variable to %s" % args.path)


def process(path):

    artist = args.artist
    print("artist", artist)

    with open(args.config, "r") as f:
        config = json.load(f)
    print("Using config", config)

    grid_size = config["grid_size"]
    pix_fixed = config["pix_fixed"]
    do_grid = config["do_grid"]
    transforms = config["transforms"]
    J = config["J"]
    L = config["L"]
    do_reduced = config["do_reduced"]

    type_interp = "fixed_" + (
        "grid" if do_grid else ("pix_%smm" % str(pix_fixed).replace(".", "_"))
    )
    print(type_interp)

    srcdir = Path(os.path.abspath(path))
    cachedir = Path(os.path.abspath(path)).parent.parent / "cache"
    filcache = cachedir / "filters"
    os.makedirs(filcache, exist_ok=True)
    filset = FiltersSet(filcache)

    window_size = config["window_size"]
    print("size of window size", window_size)

    def process_rec(srcpath):
        for path in srcpath.iterdir():
            if path.is_file():
                if path.suffix == ".jpg" or path.suffix == ".png":
                    break
            else:
                process_rec(path)

        dirname = srcpath.relative_to(srcdir)
        relpath = cachedir / dirname

        print(relpath)

        if not str(dirname).lower().startswith(artist):
            print("artist not in relpath", relpath)
            return

        savedir = relpath / ("interp_" + type_interp)
        sccache = relpath / (
            ("reduced" if do_reduced else "full")
            + "_l_scattering_"
            + type_interp
            + str(J)
            + "_"
            + str(L)
            + "_"
            + str(window_size)
        )
        albumdir = relpath / ("albums" 
            + type_interp
            + str(J)
            + "_"
            + str(L)
            + "_"
            + str(window_size))

        os.makedirs(savedir, exist_ok=True)
        os.makedirs(sccache, exist_ok=True)
        os.makedirs(albumdir, exist_ok=True)

        artworks = list(
            map(
                lambda x: ArtWork(os.path.join(str(srcpath), x), artist),
                os.listdir(str(srcpath)),
            )
        )
        artwork = artworks[0]

        interp = Interpolate(grid_size, pix_fixed, do_grid, savedir)
        scatterer = ImageScatterer(window_size, do_reduced, sccache)
        albumify = Albumify(albumdir, transforms)

        for artwork in artworks:
            print(artwork.name)
            interpolated = interp(
                artwork
            )  # maybe here use the albumentation library...
            fil = filset(window_size, window_size, J, L)
            album = albumify(artwork.name, interpolated)
            # load the albumentated images, return name and image
            scoeff = {}
            for key, image in album.items():
                scoeff[key] = scatterer(key + "_" + artwork.name, image, fil)
            # sccoef = scatterer(artwork.name, interpolated, fil) # inside scatterer using the window thing #probably needs to put the
            # actual interpolated image in it
            print("scattering coeff", scoeff)

    process_rec(srcdir)


if __name__ == "__main__":
    path = args.path
    #    path = "/Users/ioanaciuca/Desktop/hephaestusai/data/0_raw/Cadogan_Contemporary/Elise_Low_Res    path = args.path
    process(path)
