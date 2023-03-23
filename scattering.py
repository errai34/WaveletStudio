import numpy as np
import os
import torch.nn as nn
import torch
import torch.optim as optim
import torch.fft as torch_fft
import time
import slidingwindow as sw
from sklearn.preprocessing import StandardScaler


class ScatteringTransform:
	"""Sihao's code for scattering transform."""

	def __init__(self, filters_set, J, L, do_reduced):
		
		self.M, self.N = filters_set["psi"][0][0].shape
		dtype = filters_set["psi"][0][0].dtype
		self.filters_set = torch.zeros((J, L, self.M, self.N), dtype=dtype)

		for j in range(J):
			for l in range(L):
				self.filters_set[j, l] = filters_set["psi"][j * L + l][0]
		self.do_reduced = do_reduced

	def cut_high_k_off(self, data_f, j=2):
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

	def forward(
		self,
		data,
		J,
		L,
		j1j2_criteria="j2>j1",
		mask=None,
		pseudo_coef=1,
		algorithm="fast",
	):
		M, N = self.M, self.N

		data_f = torch_fft.fftn(data, dim=(-2, -1))

		S_0 = torch.zeros(1, dtype=data.dtype)
		S_1 = torch.zeros((J, L), dtype=data.dtype)
		S_2 = torch.zeros((J, L, J, L), dtype=data.dtype) #j1, l1, j2, l2

		S_2_reduced = torch.zeros((J, J, L), dtype=data.dtype)


		S_0[0] = data.mean()

		if algorithm == "fast":

			for j1 in np.arange(J):
				if j1 >= 1:
					data_f_small = self.cut_high_k_off(data_f, j1)
					wavelet_f = self.cut_high_k_off(self.filters_set[j1], j1)
				else:
					data_f_small = data_f
					wavelet_f = self.filters_set[j1]

				_, M1, N1 = wavelet_f.shape
				I_1_temp = (
					torch_fft.ifftn(
						data_f_small[None, :, :] * wavelet_f,
						dim=(-2, -1),
					).abs()
					** pseudo_coef
				)

				S_1[j1] = I_1_temp.mean((-2, -1)) * M1 * N1 / M / N

				I_1_temp_f = torch_fft.fftn(I_1_temp, dim=(-2, -1))

				for j2 in np.arange(J):
					if eval(j1j2_criteria):

						if j1 >= 1:
							factor = j2 - j1 + 1
						else:
							factor = j2

						I_1_temp_f_small = self.cut_high_k_off(I_1_temp_f, factor)
						wavelet_f2 = self.cut_high_k_off(self.filters_set[j2], j2)

						_, M2, N2 = wavelet_f2.shape

						I_2_temp = (
							torch_fft.ifftn(
								I_1_temp_f_small[:, None, :, :]
								* wavelet_f2[None, :, :, :],
								dim=(-2, -1),
							).abs()
							** pseudo_coef
						)
						S_2[j1, :, j2, :] = I_2_temp.mean((-2, -1)) * M2 * N2 / M / N

		for l1 in range(L):
			for l2 in range(L):
				S_2_reduced[:, :, (l2 - l1) % L] += S_2[:, l1, :, l2] #mhm confusing bit
				

		S_2_reduced /= L

		if self.do_reduced is True:
			S_2_reduced = S_2_reduced.sum(-1) #maybe use this one!@?

		S = torch.cat((S_1.sum(1), S_2_reduced.flatten()[(S_2_reduced.flatten() != 0)]))
		# take S1 = S1/S0 but not for now
		# S = S_1.sum(1)
		# S = S_1
		return S


class ImageScatterer:
	def __init__(self, window_size,  do_reduced=True, savedir=None):
		self.window_size = window_size
		self.savedir = savedir
		self.do_reduced = do_reduced

	def __call__(self, name, img, filset):
		path = self.savedir / (name + "_coef.npy") if self.savedir is not None else None
		if path is not None and path.exists():
			return np.load(path)

		img = (
			StandardScaler()
			.fit_transform(img.reshape(-1, img.shape[-1]))
			.reshape(img.shape)
		)
		st = ScatteringTransform(filset[0], filset[1], filset[2], self.do_reduced)

		shape = filset[0]["psi"][0][0].shape
		coef = (
			self.__do_full_image(img, filset, st)
			if shape == img.shape[:-1]
			else self.__do_sub_image(img, filset, shape, st)
		)
		coef = np.array(coef)
		if path is not None:
			np.save(path, coef)
		return coef

	def __do_full_image(self, img, filset, st):
		# initiate array to store results
		scattering_coeff = []
		print("doing full scattering")
		J = filset[1]
		L = filset[2]

		# run different RGB channels separately
		for j in range(3):
			image_torch = torch.from_numpy(img[:, :, j])

			# run scattering transform
			scattering_coeff.append(
				st.forward(image_torch, J, L).cpu().detach().numpy()
			)

		return scattering_coeff

	def __cookie_cutter(self, img):
		M, N = self.window_size, self.window_size
		piece = img.detach().numpy()

		px = piece.shape[0]
		py = piece.shape[1]
		print(type(piece))
		res = np.zeros((M, N))
		for m in range(M // piece.shape[0] + (0 if M % piece.shape[0] == 0 else 1)):
			for n in range(N // piece.shape[1] + (0 if N % piece.shape[1] == 0 else 1)):

				print('m', m)
				print('n', n)
				filler = piece if m % 2 == 0 else np.flip(piece, 0)
				print(filler)
				filler = filler if n % 2 == 0 else np.flip(filler, 1)
				#filler = piece (no flip)

				rlimx = min((m+1)*px, M) - m*px
				rlimy = min((n+1)*py, N) - n*py
				res[m*px:rlimx+m*px, n*py:rlimy+n*py] = filler[0:rlimx, 0:rlimy]

		return torch.from_numpy(res)

	def __sliding_window_sub_image(self, image_torch, filset, shape, st):
		print('using sliding window')
		M, N = shape[0], shape[1]
		J, L = filset[1], filset[2]
		print('M, N ANNOYING IS', M, N)

		print('Size of image:', image_torch.size())

		windows = sw.generate(image_torch, sw.DimOrder.HeightWidthChannel, self.window_size, 0.2) # < 1024, do cookie cutter, 
		scattering_coeff_temp = []
		print(len(windows))
		for window in windows:
			subset = image_torch[window.indices()]
			print('subset SHAPUS', subset.shape)
			sbst = st.forward(self.__cookie_cutter(subset),
					   #image_torch[k1 * M : (k1 + 1) * M, k2 * N : (k2 + 1) * N],
					   J,
					   L,
					).cpu().detach().numpy()
			scattering_coeff_temp.append(sbst)

		return scattering_coeff_temp

	def __do_sub_image(self, img, filset, shape, st):
		J = filset[1]
		scattering_coeff = []

		# run different RGB channels separately
		for j in range(3):
			image_torch = torch.from_numpy(img[:, :, j]) #here maybe add the sliding window code, 
			# use the cookie cutter only if input image dim is smaller than window of the sliding window code.
			# also might want to move this code somehwere else?

			compute_coeff = self.__sliding_window_sub_image
			scattering_coeff_temp = compute_coeff(image_torch, filset, shape, st)
			scattering_coeff_temp = np.array(scattering_coeff_temp)
			scattering_coeff.append(np.mean(scattering_coeff_temp, axis=0))

		print(np.array(scattering_coeff).shape)
		

		return scattering_coeff
