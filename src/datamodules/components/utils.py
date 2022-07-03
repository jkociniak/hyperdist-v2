import numpy as np
from scipy.special import gamma
from scipy.interpolate import interp1d


def mobius_addition_np(x, y, c):
    dot_xy = np.sum(x * y)
    dot_xx = np.sum(x * x)
    dot_yy = np.sum(y * y)
    numerator = (1 + 2 * c * dot_xy + c * dot_yy) * x + (1 - c * dot_xx) * y
    denominator = 1 + 2 * c * dot_xy + c * c * dot_xx * dot_yy
    return numerator / denominator


def hyperbolic_dist_np(x, y, c):
    assert c > 0
    max_radius = 1 / np.sqrt(c)
    assert np.linalg.norm(x) < max_radius
    assert np.linalg.norm(y) < max_radius

    mobadd = mobius_addition_np(-x, y, c)
    d = np.linalg.norm(mobadd)

    # add epsilon to handle inaccurate computations
    assert d.max() <= max_radius + 1e9

    # clipping
    d = max_radius - 1e-8 if d > max_radius - 1e-8 else d

    # make runtime warnings errors
    import warnings
    warnings.simplefilter('error')

    return 2/np.sqrt(c) * np.arctanh(np.sqrt(c) * d)


def hyperbolic_volume_inverse(n, c, min_r=1e-3, max_r=5.31, n_samples=100000):
    #rs = np.logspace(np.log(min_r), np.log(max_r), num=n_samples, base=np.e)
    # rs_small = np.linspace(9e-3, 11e-3, 3*n_samples)
    # rs_med = np.linspace(11e-3, 21e-3, 3*n_samples)
    # rs_big = np.linspace(21e-3, max_r, 3*n_samples)
    # rs = np.concatenate((rs_small, rs_med, rs_big))

    # evenly spaced nodes (consider other?)
    rs = np.linspace(min_r, max_r, n_samples)

    # range of volumes
    max_v = hyperbolic_volume(n, c, max_r)
    min_v = hyperbolic_volume(n, c, min_r)

    # we have to normalize the volumes to be from (0, 1) range
    vs = np.array([(hyperbolic_volume(n, c, r) - min_v)/(max_v - min_v) for r in rs])
    _, idx = np.unique(vs, return_index=True)

    # interpolate inverse of hyperbolic volume
    inv_interp = interp1d(vs[idx], rs[idx], kind='linear')

    # compose with hyperbolic to euclidean radius transformation
    return lambda v: hr2er(inv_interp(v), c)


# volume of n-dimensional euclidean sphere
def eucl_sphere(n):
    num = 2 * np.pi ** ((n+1)/2)
    denom = gamma((n+1)/2)
    return num/denom


def er2hr(r, c):
    # rh = 2\sqrt(c) * arctgh(sqrt(c) * re)
    max_radius = 1/np.sqrt(c)
    assert (r < max_radius).all()
    return 2/np.sqrt(c) * np.arctanh(np.sqrt(c) * r)


def hr2er(r, c):
    # solve rh = 2\sqrt(c) * arctgh(sqrt(c) * re)
    # rh*sqrt(c)/2 = arctgh(sqrt(c) * re)
    # tgh(rh*sqrt(c)/2) = sqrt(c) * re
    # re = tgh(rh*sqrt(c)/2)/sqrt(c)
    assert r.min() > 0
    return np.tanh(r * np.sqrt(c) / 2) / np.sqrt(c)


def hyperbolic_volume(n, c, r, sphere_factor=False):
    # n: dimension of the hyperbolic space
    # r: hyperbolic radius of the sphere
    # c: curvature of the hyperbolic space
    #
    # we want to compute V_{n, c}(r) = e_{n-1} \int_0^r (sinh(t*sqrt(c))/sqrt(c))^{n-1} dt
    # where e_{n-1} = 2pi^{n/2}/gamma(n/2) is a volume of unit ball in ndim eucl space
    # so we use the formula
    #
    # sinh(t) = (e^x - e^{-x})/2
    #
    # and the binomial theorem to derive
    #
    # \int_0^r sinh(t)^n dt =
    # 1/(2sqrt(c))^{n-1} \sum_{k=0}^{n-1} (n-1 choose k) (-1)^{n-1-k} (e^{(2k-n)r sqrt(c)} - 1)/((2k-n)sqrt(c))
    c = np.sqrt(c)

    sign = 1 if (n-1) % 2 == 0 else -1
    sum = sign * np.expm1(-(n-1) * r * c) / ((-(n-1)) * c)

    coeff = 1
    for k in range(1, n):
        sign = -sign
        coeff *= (n-k)/k
        if 2*k - (n-1) != 0:
            sum += sign * coeff * np.expm1((2*k - (n-1)) * r * c) / ((2*k - (n-1)) * c)
        else:
            sum += sign * coeff * r

    if sphere_factor:
        return eucl_sphere(n-1) * sum / ((2*c)**(n-1))
    else:
        return sum