import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re

filepath1 = "/path/to/csv/file" # Random samples from unoptimised Bell distribution
filepath2 = "/path/to/csv/file" # Random samples from optimised Bell distribution


def gauss(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def load_complex_csv(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        m = re.match(r'\s*\((-?\d+\.\d+(?:[eE][+-]?\d+)?)\+(-?\d+\.\d+(?:[eE][+-]?\d+)?)j\)', line)
        if m is not None:
            real = float(m.group(1))
            imag = float(m.group(2))
            data.append(complex(real, imag))
    return np.array(data).real

def get_bins(data):

    # Use Freedman-Diaconis rule for binning
    q3, q1 = np.percentile(data, [75, 25])
    iqr = q3 - q1
    bin_width = 2*iqr*(len(data)**(-1/3))
    num_bins = int((np.max(data) - np.min(data)) / bin_width)
    print("Number of bins:", num_bins)
    return num_bins

def guassian_fit(hist, bounds = ((0, 1.5, 0), (0.1, 3, 2))):
    # fit to gaussian:
    x_data = []
    for i in range(len(hist[1]) - 1):
        x = 0.5*(hist[1][i] + hist[1][i+1])
        x_data.append(x)
    y_data = hist[0]

    popt, pcov = curve_fit(gauss, x_data, y_data, bounds = bounds)

    return popt, pcov, x_data

def print_fit(popt, name):
    print(f"FIT PARAMETERS FOR {name}")
    a = popt[0]
    mu = popt[1]
    sigma = abs(popt[2])
    print("a:", a)
    print("mu:", mu)
    print("sigma:", sigma)
    print("stat sig", (mu - 2) / sigma)

    return None

data1 = np.genfromtxt(filepath1, delimiter = ',')
data2 = np.genfromtxt(filepath2, delimiter = ",")


plt.title(r"$\langle \it{B_{CGLMP}} \rangle$" + " Distribution for " + r"$H \rightarrow WW$", fontsize = 18)


hist1 = plt.hist(data1, get_bins(data1), weights = np.zeros_like(data1) + 1. / data1.size, alpha = 0.5)
hist2 = plt.hist(data2, get_bins(data2), weights = np.zeros_like(data2) + 1. / data2.size, alpha = 0.5)

popt1, pcov1, x1 = guassian_fit(hist1)
popt2, pvov2, x2= guassian_fit(hist2)

print_fit(popt1, "ORIGINAL")
print_fit(popt2, "OPTIMISED")


plt.plot([],[],' ', label = r"$\bf{10^4 \, Samples}$")

plt.plot(x1, gauss(x1, *popt1), color = "blue", label = "Unoptimised\nGaussian Fit")
plt.plot(x2, gauss(x2, *popt2), color = "orange", label = "Optimised\nGaussian Fit")
plt.axvline(x = popt1[1], color = "blue", linestyle = 'dashed')
plt.axvline(x = popt2[1], color = "orange", linestyle = 'dashed')
plt.axvline(x = 2, color = "green", linestyle = 'dashed', label = "CGLMP Bound")

t1 = plt.text(2.93 , 0.028, r" = " + f"{round(abs(popt1[2]), 4)}", fontsize = 16)
t2 = plt.text(2.93, 0.023, r" = " + f"{round(abs(popt2[2]), 4)}", fontsize = 16)
#t1.set_bbox(dict(facecolor='red', alpha=0.5))
#t2.set_bbox(dict(facecolor='red', alpha=0.5))


plt.xlabel(r"$\langle \it{B_{CGLMP}} \rangle_{event}$", fontsize = 16)
plt.ylabel("Frequency Density", fontsize = 16)
#plt.text(2.8, 0.025, r"$\sigma = $" + f"{round(sigma, 4)}", fontsize = 20)
plt.legend(loc = "upper right")
plt.xlim([1.8, 3.3])
plt.axhline(y=0, color = "black")
savepath = os.getcwd() + "\WW\WW_Bell_distribution10000.png"
plt.savefig(savepath, dpi = 300)

print("Plot saved to", savepath)
