import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def make_psi(n, l, m):
    laguerre = sp.special.genlaguerre(n-l-1, 2*l+1)
    sph_harm = lambda theta, phi: sp.special.sph_harm_y(l, m, theta, phi)
    return lambda r, theta, phi: r**l * np.exp(-r/n) * laguerre(2*r/n) * sph_harm(theta, phi)


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)
    return r, theta, phi

def plane_cartesian_to_spherical(x, y, plane=0):
    match plane:
        case 0: point = (x, y, 0)  # XY plane
        case 1: point = (0, x, y)  # YZ plane
        case 2: point = (y, 0, x)  # ZX plane
    return cartesian_to_spherical(*point)

def spherical_to_plane_cartesian(r, theta, phi, plane=0):
    x, y, z = spherical_to_cartesian(r, theta, phi)
    match plane:
        case 0: return x, y  # XY plane
        case 1: return y, z  # YZ plane
        case 2: return z, x  # ZX plane

def plane_polar_to_spherical(r, theta, plane=0):
    match plane:
        case 0: # XY Plane
            return r, theta, np.pi / 2
        case 1: # YZ Plane
            return r, np.pi / 2, theta
        case 2: # ZX Plane
            return r, 0, theta


def sample_distribution(orbital_func, n_samples=10000, plane=0):

    # Find the max
    max = 0
    r_rands = 100 * np.sqrt(np.random.uniform(0, 1, size=100000))
    theta_rands = np.random.uniform(0, 2 * np.pi, size=100000)
    candidates = np.array([plane_polar_to_spherical(r, theta, plane) for r, theta in zip(r_rands, theta_rands)])
    values = abs(orbital_func(candidates[:,0], candidates[:,1], candidates[:,2]))**2
    max = np.max(values)

    # Find the bounds
    bound = 100
    while True:
        max_val = 0
        for theta in np.linspace(0, 2 * np.pi, 100):
            r, theta, phi = plane_polar_to_spherical(bound, theta, plane)
            value = abs(orbital_func(r, theta, phi))**2
            if value >= max * 0.0001:
                max_val = value

        if max_val >= max * 0.0001:
            bound += 1
            break
        bound -= 1

    print(max, bound)

    # Make the samples
    samples = generate_samples(n_samples, bound, max, orbital_func, plane)
    return samples


def generate_samples(n_samples, bound, max, orbital_func, plane):
    samples = [(0, 0)]
    escaper = 0
    while len(samples) < n_samples and escaper < 1000:
        r_rands = bound * np.sqrt(np.random.uniform(0, 1, size=n_samples))
        theta_rands = np.random.uniform(0, 2 * np.pi, size=n_samples)
        candidates = np.array([plane_polar_to_spherical(r, theta, plane) for r, theta in zip(r_rands, theta_rands)])

        values = abs(orbital_func(candidates[:,0], candidates[:,1], candidates[:,2]))**2
        thresholds = np.random.uniform(0, max, size=n_samples)
        accepted = values > thresholds

        points = [spherical_to_plane_cartesian(candidates[i,0], candidates[i,1], candidates[i,2], plane) for i in range(n_samples) if accepted[i]]
        samples.extend(points)
        escaper += 1

    return np.array(samples)


def sample_distribution_plane(args):
    n, l, m, n_samples, plane = args
    orbital = make_psi(n, l, m)
    return sample_distribution(orbital, n_samples=n_samples, plane=plane)