import os
from pathlib import Path
import re
import matplotlib.image as mpimg
import cv2
import numpy as np
from scipy.interpolate import griddata


class ArtWork:
	"""
	Class that provides functionality to read in the image data, process it,
	and extract its width & heigh and the size unit.

	Parameters
	----------
	path: str
			absolute or local path where the images are

	Returns
	-------
	artwork: object
			An artwork object with width, height and unit attributes.
	"""

	def __init__(self, path, artist=None):
		print("painting", path)
		self.path = os.path.abspath(path)
		self.artist = artist
		self.__get_properties()
		self.__img = None

	def __get_properties(self):

		base = os.path.basename(self.path)
		self.name = os.path.splitext(base)[0]

		if self.artist == "monet":
			stripped = re.sub(r"\s+", "", self.name.rpartition("(")[-1])

		elif self.artist == "van_gogh":
			stripped = re.sub("_", ".", self.name.rpartition("vincent-van-gogh_")[-1])
			stripped = re.sub("-", "", stripped)


		else:
			stripped = re.sub(r"\s+", "", self.name.rpartition("-")[-1])
			print(stripped)

		try:

			p = re.compile("^([0-9]+\.?[0-9]*)(cm|in)?x([0-9]+\.?[0-9]*)(cm|in)?(\))?$")
			g = p.search(stripped)
			print('g', g)

			print('name', self.name)


			self.width = float(g.group(3))
			self.height = float(g.group(1))

			self.unit = g.group(2) if g.group(2) else g.group(4) if g.group(4) else "in"
			print('unit', self.unit)


		except ValueError:
			print(
				"Oops. There are issues with the way the name of the painting was formatted."
			)

	@property
	def img(self):
		if self.__img is None:
			self.__img = cv2.imread(self.path)
		return self.__img


class Interpolate:
	"""
	Interpolates the given image to a particular grid, which can be either
	fixed (N x N pixels) or given by a fixed pixel size.

	Parameters
	----------
	type_interp: str
			The type of interpolation. Can be either "fixed", which is a fixed
			grid (N X N pixels) or "pixel", which is fixed pixel size.
	savedir: str
			Path where to save the path
	Returns
	-------
	interpimg: np.ndarray
			The interpolated image data
	"""

	def __init__(self, grid_size, pix_fixed, do_grid, savedir=None):
		self.grid_size = grid_size
		self.pix_fixed = pix_fixed
		self.do_grid = do_grid
		self.savedir = savedir

	#        self.artwork = ArtWork(path) #create an ArtWork
	#        self.artwork.load()

	def __call__(self, artwork):
		cachepath = self.__save_path(artwork)
		if self.savedir is not None and cachepath.exists():
			return np.load(cachepath)

		return (
			self.__interp_to_grid(artwork)
			if self.do_grid
			else self.__interp_fixed_pixel(artwork)
		)

	def __interp_to_grid(self, artwork):

		grid_x, grid_y = np.meshgrid(
			np.linspace(0.001, 0.999, self.grid_size),
			np.linspace(0.001, 0.999, self.grid_size),
		)

		interpimg = cv2.resize(
			artwork.img,
			dsize=(self.grid_size, self.grid_size),
			interpolation=cv2.INTER_LINEAR,
		)
		if self.savedir is not None:
			self.save(artwork, interpimg)

		return interpimg

	def __interp_fixed_pixel(self, artwork):

		img = artwork.img
		height = artwork.height
		print("height is", height)
		width = artwork.width
		print("width is", width)

		unit_scale = 1 if artwork.unit == "cm" else 2.54

		x_pix_size = height / float(img.shape[0]) * unit_scale * 10  # in mm
		y_pix_size = width / float(img.shape[1]) * unit_scale * 10  # in mm

		nx = int(x_pix_size / self.pix_fixed * img.shape[0])
		ny = int(y_pix_size / self.pix_fixed * img.shape[1])

		print('nx is', nx)
		print('ny is', ny)

		interpimg = cv2.resize(img, dsize=(ny, nx), interpolation = cv2.INTER_LINEAR)


		if self.savedir is not None:
			self.save(artwork, interpimg)

		return interpimg

	def __save_path(self, artwork):
		return self.savedir / (artwork.name + ".npy")

	def save(self, artwork, interpimg):
		np.save(self.__save_path(artwork), interpimg)
