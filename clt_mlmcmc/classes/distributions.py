import numpy as np
import scipy as sp
from abc import ABC, abstractmethod
from numbers import Number


class Distribution(ABC):
    @abstractmethod
    def logpdf(self, x):
        """Define log PDF of distribution for analysis."""
        pass

    @abstractmethod
    def sample(self, size):
        """Generate sample of the distribution of size 'size'."""
        pass



class Uniform(Distribution):
    def __init__(self, lower=0, upper=1):
        if isinstance(lower, Number) and isinstance(upper, Number):
            if not lower < upper:
                raise ValueError("Uniform requires lower < upper.")
            lower = np.asarray([lower], dtype=float)
            upper = np.asarray([upper], dtype=float)
        else:
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            if lower.shape != upper.shape:
                raise ValueError("Bounds must have same shape.")
            if lower.ndim != 1:
                raise ValueError("Bounds must be 1-dimensional arrays.")
            if not np.all(lower < upper):
                raise ValueError("Each lower bound must be strictly smaller than"
                                "the corresponding upper bound.")
        self.lower = lower
        self.upper = upper
        self.dim = len(lower)

    def __str__(self):
        return f"{type(self).__name__}[{self.lower}, {self.upper}]"
    
    def __repr__(self):
        return f"{type(self).__name__}[{self.lower}, {self.upper}]"

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            if self.dim != 1:
                raise ValueError("x must match distribution dimension.")
            if not (self.lower[0] <= x <= self.upper[0]):
                return -np.inf
            return -np.log(self.upper[0]-self.lower[0])
        elif x.ndim == 1:
            if x.shape != self.lower.shape:
                raise ValueError(f"x must have shape {self.lower.shape}.")
            inside = np.all((x >= self.lower) & (x <= self.upper))
            if not inside:
                return -np.inf
            else:
                return -np.sum(np.log(self.upper - self.lower))
        elif x.ndim == 2:
            if x.shape[1] != self.dim:
                raise ValueError("Each row of x must have shape "
                                 f"({self.dim}).")
            inside = np.all((x >= self.lower & self.upper), axis=1)
            log_density = -np.sum(np.log(self.upper - self.lower))
            result = np.full(x.shape[0], -np.inf)
            result[inside] = log_density
            return result
        else:
            raise ValueError("x must be a scalar, 1D or 2D array.")

    def sample(self, size):
        try:
            return np.random.uniform(low=self.lower, high=self.upper,
                                     size=(size, self.dim))
        except Exception as e:
            raise ValueError("size must be an int or tuple of ints.") from e


class Normal(Distribution):
    def __init__(self, mean=0, cov=1):
        if isinstance(mean, Number) and isinstance(cov, Number):
            if cov <= 0:
                raise ValueError("variance must be positive.")
            self.mean = mean
            self.cov = cov
            self.dim = 1
            self._dist = sp.stats.norm(loc=mean, scale=np.sqrt(cov))
        else:
            mean = np.asarray(mean)
            cov = np.asarray(cov)

            if mean.ndim != 1:
                raise ValueError("mean must be 1 dimensional array.")
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError("cov must be a square 2D array.")
            if cov.shape[0] != mean.shape[0]:
                raise ValueError("mean and cov dimensions do not match.")
            if not np.allclose(cov, cov.T):
                raise ValueError("cov is not symmetric.")
            eigvals = np.linalg.eigvalsh(cov)
            if not np.all(eigvals > 0):
                raise ValueError("cov must be positive definite.")
            self.mean = mean
            self.cov = cov
            self.dim = len(mean)
            self._dist = sp.stats.multivariate_normal(mean=self.mean,
                                                      cov=self.cov)
    
    def __str__(self):
        if self.dim == 1:
            return f"{type(self).__name__}({self.mean}, {self.cov})"
        else:
            return f"MVN(mean, cov, shape={self.mean.shape})"

    def __repr__(self):
        if self.dim == 1:
            return f"{type(self).__name__}({self.mean}, {self.cov})"
        else:
            return f"MVN(mean, cov, shape={self.mean.shape})"

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        if self.dim == 1:
            if x.ndim == 0:
                return self._dist.logpdf(x)
            elif x.ndim == 1:
                return self._dist.logpdf(x)
            elif x.ndim == 2:
                return np.array([self._dist.logpdf(xi) for xi in x])
            else:
                raise ValueError("x must be scalar, 1D, or 2D array.")
        else:
            if x.ndim == 1:
                if x.shape != self.mean.shape:
                    raise ValueError(f"x must have shape {self.mean.shape}")
                return self._dist.logpdf(x)
            elif x.ndim == 2:
                if x.shape[1] != self.dim:
                    raise ValueError("Each row of x must have shape "
                                     f"({self.dim},)")
                return self._dist.logpdf(x)
            else:
                raise ValueError("x must be a 1D or 2D array.")

        
    def sample(self, size):
        try:
            samples = self._dist.rvs(size=size)
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            return samples
        except Exception as e:
            raise ValueError("size must be an int or tuple of ints.") from e
    