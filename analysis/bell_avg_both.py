import numpy as np
import matplotlib.pyplot as plt
import os
import re
import random
import time

def main():
    tstart = time.time()
    filepath1 = "/path/to/csv/file" # Optimised full bell distribution
    filepath2 = "/path/to/csv/file" # Unoptimised full bell distribution

    data1 = np.genfromtxt(filepath1, delimiter = ',')
    data2 = load_complex_csv(filepath2)

    
    #data2 = np.array([i + 0.1 for i in data2])

    print("number of data points1:", len(data1))
    print("number of data points2:", len(data2))

    #both_plots(data1, data2)
    split_mean(data2)


def load_complex_csv(filename):
    # Use this if complex values in csv to return numpy array
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

def both_plots(data1, data2):
    mean1 = np.sum(data1) / len(data1)
    std1 = np.std(data1)
    print("Sample mean1:", mean1)
    print("Sample max1: ", np.max(data1))
    print("Sample min1:", np.min(data1))
    print("Sample std1", std1)

    mean2 = np.sum(data2) / len(data2)
    std2 = np.std(data2)
    print("Sample mean2:", mean2)
    print("Sample max2: ", np.max(data2))
    print("Sample min2:", np.min(data2))
    print("Sample std2", std2)

    plt.title(r"$\langle \it{B_{CGLMP}} \rangle$" + " Distribution for " + r"$H \rightarrow WW$", fontsize = 18)

    plt.hist(data1, get_bins(data1), alpha = 0.5, label = "Unoptimised Distribution")
    plt.hist(data2, get_bins(data2), alpha = 0.5, label='Optimised Distribution')


    plt.axvline(x = 2, color = 'green', label = "CGLMP Bound", linestyle = "dashed")
    #plt.axvline(x = mean1, color = "blue", label = "Original \nSample mean XY")
    plt.axvline(x = mean2, color = "red", label = "Optimised \nSample mean")
    plt.xlabel(r"$\langle \it{B_{CGLMP}} \rangle_{event}$", fontsize = 16)
    plt.ylabel("Frequency", fontsize = 16)
    plt.legend()

    savepath = os.getcwd() + "\WW\WW_Bell_distribution.png"
    plt.savefig(savepath, dpi = 300)

    print(f"Histogram saved to {savepath}")
    

def get_bins(data):

    # Use Freedman-Diaconis rule for binning
    q3, q1 = np.percentile(data, [75, 25])
    iqr = q3 - q1
    bin_width = 2*iqr*(len(data)**(-1/3))
    num_bins = int((np.max(data) - np.min(data)) / bin_width)
    print("Number of bins:", num_bins)
    return num_bins

def split_mean(data):
    
    M = 10000
    t_start = time.time()

    print(data)
    print(len(data))
    print(type(data))
    data_ls = data.tolist()

    L_int = 300 #(fb^-1)
    cx_WZ = 0.05375 * 10**3 #(pb --> fb)
    cx_WW = 0.02744 * 10**3 

    N = int(np.floor(L_int * cx_WW))

    print(f"Number of expected events / sample size N: {N}")
    print(f"Number of random samples M: {M}")
    sample_means = []
    for i in range(M):
        sample = random.sample(data_ls, N)
        sample_mean = np.mean(sample)
        if i % 10 == 0: print(f"{i}, time: {int(time.time() - t_start)}, time left = {int((M/(i+1) - 1) * (time.time() - t_start))}s")
        sample_means.append(sample_mean)
    np.savetxt(f"WWnopt_BellOPTIMIZED_{M}samples.csv", sample_means, delimiter = ",")

if __name__ == '__main__':
    main()