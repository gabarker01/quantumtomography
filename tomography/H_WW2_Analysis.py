import time
import matplotlib.pyplot as plt
import numpy as np
import math

from scipy import interpolate
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle

def main():
    t_start = time.time()
    print("\n")

    gmarray_sym, gmarray_asym = run_zz_tomography()
    
    print("\nSymmetric tests:")
    c_mb_squared_sym = cmb_entanglement_test(gmarray_sym)
    bell_test_sym = CGLMP_test(gmarray_sym)
    print(bell_test_sym)

    print("\nAsymmetric tests:")
    c_mb_squared_asym = cmb_entanglement_test(gmarray_asym)
    bell_test_asym = CGLMP_test(gmarray_asym)
    print(bell_test_asym)


    plot_2d_2(gmarray_sym, cm.RdYlBu, 'H_ww_sym')
    plot_2d_2(gmarray_asym, cm.RdYlBu, 'H_ww_asym')
    print(f"\n\nDone in {time.time() - t_start}\n")

def run_zz_tomography():
    import ROOT
    from H_WW2 import QuantumTomographyH_WW

    # Open Delphes simulation tree within .root file
    f = ROOT.TFile.Open('path/to/root/file', "READ")
    tree = f.Get("Delphes")

    # Initialise analysis and obtain average P values
    analysis = QuantumTomographyH_WW(tree)
    P2, P1, cov, cov_sym = analysis.tree_analysis() #W1 , W2

    print('WIGNER P W+')
    print(P1)
    print('WIGNER P W-')
    print(P2)

    gmarray_sym = np.zeros((9,9), dtype=float, order='C')
    gmarray_asym = np.zeros((9,9), dtype=float, order='C')

    # Calculate symmetric matrix
    a_sym = 0.25 * (P1 + P2)
    for i in range(8):
        gmarray_sym[0][i+1] = a_sym[i]
        gmarray_sym[i+1][0] = a_sym[i]
        for j in range(8):
            gmarray_sym[i+1][j+1] = (1/8) * cov_sym[i][j]

    print('SYMETTRIC GM MATRIX')
    print(gmarray_sym)

    # Calculate asym matrix
    for i in range(8):
        gmarray_asym[0][i+1] = 0.5 * P2[i] # W-
        gmarray_asym[i+1][0] = 0.5 * P1[i] # W+
        for j in range(8):
            gmarray_asym[i+1][j+1] = 0.25 * cov[i][j]

    print('ASYMMETRIC GM MATRIX')
    print(gmarray_asym)

    return gmarray_sym, gmarray_asym

def cmb_entanglement_test(gmarray):
    '''
    Sufficient condition test for entanglement of qutrit pair system.

    Proposed in:
    F. Mintert and A. Buchleitner, Observable entanglement measure for mixed quantum states,
    Physical Review Letters 98 (2007) https://doi.org/10.1103/physrevlett.98.140505.

    :param gmarray: bipartite density matric in GGM basis

    :returns c_mb_sqd: float value for result of test. A positive value is sufficient to demonstrate entanglement, <= 0 is inconclusive.
    
    '''

    # Sum W-, W+ and correlation entries from gmarray respectively.
    c_mb_squared = -4/9
    for i in range(1,9):
        c_mb_squared += ((-2/3) * (gmarray[0][i])**2)
        c_mb_squared += ((-2/3) * (gmarray[i][0])**2)
        for j in range(1,9):
            c_mb_squared += ((8) * (gmarray[i][j])**2)
    
    print("\nc_mb_squared:", c_mb_squared)

    if c_mb_squared > 0:
        print("System is entangled.")
    else:
        print("CM Entanglement test inconclusive.")
    
    return c_mb_squared

def CGLMP_test(gmarray):
    '''
    Performs CGLMP Bell violation test by finding the expectation value of Bell operator. Values above 2 are 
    forbidden in local realist interpretations of QM.

    :param gmarray: numpy array holding density matrix from tomography

    :returns planes: dict containing expectation value for CGLMP operator in different planes.
    
    '''
    Bell_violating = False

    tests = {}

    B_xy = -(4/3) * (math.sqrt(3) * gmarray[1][1] + math.sqrt(3)*gmarray[1][6] + math.sqrt(3)* gmarray[2][2] + math.sqrt(3)*gmarray[2][7] - 
    3 * gmarray[4][4] - 3 *gmarray[5][5] + math.sqrt(3) * (gmarray[6][1] + gmarray[6][6] + gmarray[7][2] + gmarray[7][7]))
    tests["xy"] = B_xy


    for value in tests.values():
        if value >= 2:
            Bell_violating = True
    
    if Bell_violating:
        print("\nEvidence for Bell Violation.")
    else:
        print("\nInsufficient evidence for Bell Violation.")
    
    return tests

def plot_3d(gmarray):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X = np.array([i for i in range(9)])
    Y = np.array([i for i in range(9)])
    X , Y = np.meshgrid(X,Y)
    Z = gmarray
    print(Z.shape)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.ylabel('$\Lambda_1$')
    plt.xlabel('$\Lambda_2$')
    plt.gca().xaxis.tick_bottom()
    plt.title(r'$\rho(H \rightarrow ZZ)$')
    
    plt.show()
    plt.savefig("H_WW.png")

def plot_3d_smooth(array):

    X, Y = np.mgrid[0:8:9j, 0:8:9j]
    Z = array

    xnew, ynew = np.mgrid[0:8:180j, 0:8:180j]
    tck = interpolate.bisplrep(X, Y, Z, s=0)
    znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xnew, ynew, znew, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=1, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.ylabel('$\Lambda_1$')
    plt.xlabel('$\Lambda_2$')
    plt.gca().xaxis.tick_bottom()
    plt.title(r'$\rho(H \rightarrow WW)$')
    plt.show()

def plot_2d(gmarray):

    plt.matshow(gmarray)
    plt.colorbar()

    plt.ylabel('$\Lambda_1$')
    plt.xlabel('$\Lambda_2$')
    plt.xticks()
    plt.title(r'$\rho(H \rightarrow WW)$')

    plt.show()

def plot_2d_2(array, colour, name):


    data = array
    labels_x = [""]
    labels_y = [""]

    for i in range(8):
        labels_x.append(str(i+1))
        labels_y.append(str(i+1))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(data, cmap=cm.Blues, alpha=0.7)
    
    parity = np.zeros((9,9), dtype=str, order='C')
    for a in range(9):
        for b in range(9):
            if data[a][b] > 0:
                parity[a][b] = "+"
            elif data[a][b] < 0:
                parity[a][b] = "-"
            else:
                parity[a][b] = ""
    
    
    im = ax.matshow(data, vmin=-.3, vmax=.3, interpolation='nearest', cmap=colour, aspect='auto')
    fig.colorbar(im)

    
    ax.set_xticklabels([''] + labels_x)
    ax.set_yticklabels([''] + labels_y)
    plt.gca().xaxis.tick_bottom()
    plt.gca().invert_yaxis()
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(x=j, y=i,s=parity[i, j], va='center', ha='center', fontsize='15', color = 'w')
    
    lx = ax.hlines(y=0.5, color='black', linestyle='solid', linewidth = 0.8, xmin = 0.5, xmax = 8.5)
    lx.set_zorder(5)
    ly = ax.vlines(x=0.5, color='black', linestyle='solid', linewidth = 0.8, ymin = 0.5, ymax = 8.5)
    ly.set_zorder(5)
    box = ax.add_patch(Rectangle((-0.5, -0.5), 1.01, 1.01, facecolor = 'black'))

    plt.ylabel('$W-$' + " GM Index")
    plt.xlabel('$W+$' + " GM Index")
    plt.xticks()
    plt.title(r'$\rho(H \rightarrow WW)$', fontsize=18, y = 1.04)

    savepath = f'{name}.png'
    plt.savefig(savepath, dpi=300)

def trace(array):
    '''
    Takes numpy array representing Matrix and returns trace
    
    '''
    trace = 0
    shape = array.shape
    if shape[0] == shape[1]:
        size = int(shape[0])
        for i in range(size):
            trace += array[i][i]
    else:
        print("Error: Cannot take trace of non-square matrix")
    
    return trace


if __name__ == "__main__":
    main()