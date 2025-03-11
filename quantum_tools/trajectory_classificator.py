import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from csv import writer

def autocorrelation(x):
    """
    La función de autocorrelación para una serie finita y discreta
    """
    n = len(x)
    mean_x = np.mean(x)
    var_x = np.var(x)
    
    result = np.array([np.sum((x[:n-k] - mean_x) * (x[k:] - mean_x)) / (n * var_x) for k in range(n)])
    return result

def estimate_period(r_x):
    """
    El cálculo estimado del periodo
    """
    peaks = np.where((r_x[1:-1] > r_x[:-2]) & (r_x[1:-1] > r_x[2:]))[0] + 1 
    if len(peaks) > 0:
        return len(peaks)
    return None

def convolved_smoothing(pure_data,length):
    """
    El suavizado por convulución
    """
    data = np.concatenate((pure_data[-60:],pure_data,pure_data[:60]))
    #data = pure_data
    kernel = np.ones(length) / length
    return np.convolve(data, kernel, mode='valid')

def prom_visibility(data,peaks):
    """
    El coeficiente de visibilidad promedio
    """
    sup_peaks, low_peaks = peaks
    max_peaks_prom = np.mean(data[sup_peaks])
    low_peaks_prom = np.mean(data[low_peaks])
    return (max_peaks_prom-low_peaks_prom)/(max_peaks_prom+low_peaks_prom)

def periodicity_coefficient_energy(r_x):
    """
    El coeficiente de autocorrelación C
    """
    c = np.sum(r_x**2)
    return c/len(r_x)

def calculate_space_coeffs(dim):
    """
    Los coeficientes de autocorrelación para caracterizar el espacio
    """
    periodic = np.cos(2 * np.pi * np.arange(dim) / 1000)
    noised = periodic + 0.5 * np.random.randn(dim)
    random = np.random.randn(dim)

    r_periodic = autocorrelation(periodic)
    r_noise = autocorrelation(noised)
    r_random = autocorrelation(random)

    periodicity_energy = periodicity_coefficient_energy(r_periodic)
    noise_periodicity_energy = periodicity_coefficient_energy(r_noise)
    random_periodicity_energy = periodicity_coefficient_energy(r_random)
    
    return (periodicity_energy,noise_periodicity_energy,random_periodicity_energy)