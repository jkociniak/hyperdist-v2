import torch
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
import numpy as np
from utils import hyperbolic_dist_np, hyperbolic_volume_inverse


class HYPERDIST(Dataset):
    def __init__(self, n_samples, dim=2, curv=1, inverse_transform='euclidean', min_r=0.1, max_r=5.3):
        self.n_samples = n_samples
        self.dim = dim
        self.min_r = min_r
        self.max_r = max_r
        self.curv = curv if inverse_transform == 'hyperbolic' else 0
        self.inverse_transform = inverse_transform

        self.pairs = self.generate_hyperbolic_pairs()
        print('Generated pairs')
        self.distances = self.compute_distances()
        print('Computed dists')

    def generate_hyperbolic_pairs(self):
        # we want to sample uniformly on either Euclidean or hyperbolic disk of radius r
        # 1. sample direction (uniform sampling on n-dimensional unit sphere)
        # 2. sample norm (in a way that ensures uniform distribution on given space)
        # 3. scale the direction vector by norm

        # 1. sampling direction
        # direction is going to be sampled from multivariate normal distribution
        mean = torch.zeros(self.dim)
        cov = torch.eye(self.dim)
        distribution = MultivariateNormal(mean, cov)
        directions = distribution.sample((self.n_samples, 2))

        # we normalize directions to be unit vectors
        norms = torch.linalg.norm(directions, axis=2, keepdims=True)
        unit_directions = torch.divide(directions, norms)


        # if self.max_norm == 'fixed':
        #     max_radius = 1/self.curv - self.eps
        # elif self.max_norm == 'proportional':
        #     max_radius = 1/self.curv * (1 - self.eps)
        # else:
        #     raise ValueError('incorrect value of max_norm setting (must be "fixed" or "proportional")')

        # 2. sampling norm
        # sample from uniform distribution on (0, 1)
        distribution = Uniform(0, 1)
        radii = distribution.sample((self.n_samples, 2, 1))

        # transform the sampled radius with proper cdf
        # inverse transform used for sampling, must be normalized to have domain [0, 1]
        if self.inverse_transform == 'euclidean':
            # p = A(r) - A(r_min)/A(r_max) - A(r_min) = r^n - r_min^n / r_max^n - r_min^n
            # so inverse is r = (p * (r_max^n - r_min^n) + r_min^n)^{1/n}
            inverse_transform = lambda p: torch.pow(
                p * (self.max_r ** self.dim - self.min_r ** self.dim) + self.min_r ** self.dim, 1 / self.dim)
        elif self.inverse_transform == 'hyperbolic':
            # we define hyperbolic volume inverse in separate function
            inverse_transform = hyperbolic_volume_inverse(self.dim, self.curv, self.min_r, self.max_r)
        else:
            raise ValueError('incorrect value of inverse_transform setting (must be "euclidean" or "hyperbolic")')

        pairs = unit_directions * inverse_transform(radii)

        return pairs.float()

    def compute_distances(self):
        if self.inverse_transform == 'euclidean':
            return self.compute_edist()
        elif self.inverse_transform == 'hyperbolic':
            return self.compute_hdist()
        else:
            raise ValueError('incorrect value of inverse_transform setting (must be "euclidean" or "hyperbolic")')

    def compute_hdist(self):
        # based on https://stackoverflow.com/questions/46084656/numpy-apply-along-n-spaces
        pairs_reshaped = np.reshape(self.pairs.numpy(), (self.n_samples, 2 * self.dim))  # flatten the two last axes

        def hyperbolic_dist_wrapper(row):
            row_2d = np.reshape(row, (2, self.dim))  # unflatten
            return hyperbolic_dist_np(row_2d[0], row_2d[1], self.curv)  # compute the distance

        distances = np.apply_along_axis(hyperbolic_dist_wrapper, 1, pairs_reshaped)
        distances = distances.astype(np.float32)
        return torch.from_numpy(distances)

    def compute_edist(self):
        diff = self.pairs[:, 0, :] - self.pairs[:, 1, :]
        return torch.linalg.norm(diff, dim=1, keepdim=False)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {'pairs': self.pairs[idx], 'dist': self.distances[idx]}