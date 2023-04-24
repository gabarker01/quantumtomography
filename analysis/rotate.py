import numpy as np
import pandas as pd
import sys
import scipy


class Bell_Rotation():
    def __init__(self, gmarray):
        '''
        Class for finding the optimised CGLMP Bell operator for a given array of Gell-Mann parameters of a spin density matrix.
        '''
        
        self.gmlist = [self.gellmann(2,1,3),
            self.gellmann(1,2,3),
            self.gellmann(1,1,3),
            self.gellmann(3,1,3),
            self.gellmann(1,3,3),
            self.gellmann(3,2,3),
            self.gellmann(2,3,3),
            self.gellmann(2,2,3)]
        
        self.rho = self.gm_to_rho(gmarray)
        
    # COPY CODE FOR GENERATING GM MATRICES
    def gellmann(self, j, k, d):
        r"""Returns a generalized Gell-Mann matrix of dimension d. According to the
        convention in *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008),
        returns :math:`\Lambda^j` for :math:`1\leq j=k\leq d-1`,
        :math:`\Lambda^{kj}_s` for :math:`1\leq k<j\leq d`,
        :math:`\Lambda^{jk}_a` for :math:`1\leq j<k\leq d`, and
        :math:`I` for :math:`j=k=d`.

        :param j: First index for generalized Gell-Mann matrix
        :type j:  positive integer
        :param k: Second index for generalized Gell-Mann matrix
        :type k:  positive integer
        :param d: Dimension of the generalized Gell-Mann matrix
        :type d:  positive integer
        :returns: A genereralized Gell-Mann matrix.
        :rtype:   numpy.array

        """

        if j > k:
            gjkd = np.zeros((d, d), dtype=np.complex128)
            gjkd[j - 1][k - 1] = 1
            gjkd[k - 1][j - 1] = 1
        elif k > j:
            gjkd = np.zeros((d, d), dtype=np.complex128)
            gjkd[j - 1][k - 1] = -1.j
            gjkd[k - 1][j - 1] = 1.j
        elif j == k and j < d:
            gjkd = np.sqrt(2/(j*(j + 1)))*np.diag([1 + 0.j if n <= j
                                                else (-j + 0.j if n == (j + 1)
                                                        else 0 + 0.j)
                                                for n in range(1, d + 1)])
        else:
            gjkd = np.diag([1 + 0.j for n in range(1, d + 1)])

        return gjkd

    def numerical_bell(self, params, final = False):
        '''
        :params is input list of 16 real params
        :final is bool, if True will return optimal U and V matrices

        :returns rotated Bell operator
        '''
        if len(params) != 16:
            print(f"{len(params)} parameters! - require 16")
            sys.exit()

        # Calculate members of SU(3) using these parameters

        id = self.gellmann(3,3,3)

        U = id
        for i, param in enumerate(params[:8]):
            U = np.matmul(U, scipy.linalg.expm(1j * param * self.gmlist[i]))
        
        V = id
        for i, param in enumerate(params[8:]):
            V = np.matmul(V, scipy.linalg.expm(1j * param * self.gmlist[i]))
        
        if final:
            return U, V
        
        else:

            UV = np.kron(U,V)
            
            # Unrotated bell operator
            B = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, -2/np.sqrt(3), 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, -2/np.sqrt(3), 0, 2, 0, 0], 
                    [0, -2/np.sqrt(3), 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, -2/np.sqrt(3), 0, 0, 0, -2/np.sqrt(3), 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, -2/np.sqrt(3), 0], 
                    [0, 0, 2, 0, -2/np.sqrt(3), 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, -2/np.sqrt(3), 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])

            return np.matmul(UV.conj().T, np.matmul(B, UV, dtype = np.complex128), dtype = np.complex128)

    def trace(self, array):
        dim = array.shape[0]
        if dim != array.shape[1]:
            return "Cannot do trace of non-square."
        else:
            tr = 0
            for i in range(dim):
                tr += array[i][i]
            return tr

    def gm_to_rho(self, gmarray):
        '''
        gmarray to density matrix
        '''
        id = self.gellmann(3,3,3)
        A = (1/9) * self.gellmann(9,9,9)

        B = np.zeros((9,9), dtype=np.complex128, order='C')
        C = np.zeros((9,9), dtype=np.complex128, order='C')
        D = np.zeros((9,9), dtype=np.complex128, order='C')

        for i in range(1,9):
            B += (1/3) * gmarray[0][i] * np.kron(self.gmlist[i-1], id)
            C += (1/3) * gmarray[i][0] * np.kron(id, self.gmlist[i-1])
            for j in range(1,9):
                D += gmarray[i][j] * np.kron(self.gmlist[i-1], self.gmlist[j-1])
        
        return A + B + C + D


    # Define the black-box function
    def black_box_func(self, x):
        '''
        returns value of bell expectancy for 16 real parameters x
        '''
        # Evaluate the function at the given point and return the value
        bell_operator = self.numerical_bell(x)
        
        return self.trace(np.matmul(self.rho, bell_operator))
    
    def black_box_min(self,x):
        '''
        returns value of bell expectancy for 16 real parameters x
        '''
        # Evaluate the function at the given point and return the value
        bell_operator = self.numerical_bell(x)
        
        return -1 * self.trace(np.matmul(self.rho, bell_operator))


    def run_optimisation(self, lb, ub, method):
        if method == "pso":
            from pyswarm import pso
            # Use PSO to minimise the black-box function
            x_opt, f_opt = pso(self.black_box_min, lb, ub, swarmsize=100, maxiter=1000, minstep=1e-8, minfunc=1e-8, debug=False)
            
            bell_opt = -1 * f_opt
            U_opt, V_opt = self.numerical_bell(x_opt, True)
        
        elif method == "nm":
            from scipy.optimize import minimize
            # Uses Nelder-Mead method
            x0 = np.array([0] * 16) # Initial guess of identity

            result = minimize(self.black_box_min, x0, bounds = zip(lb,ub), method = "Nelder-Mead", options={"maxfev": 100000})
            x_opt = result.x
            bell_opt = -1 * result.fun
                

            U_opt, V_opt = self.numerical_bell(x_opt, True)
        
        elif method == "de":
            from scipy.optimize import differential_evolution
            bounds = [list(x) for x in zip(lb,ub)]
            result = differential_evolution(self.black_box_min, bounds , args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0)
            x_opt = result.x
            bell_opt = -1 * result.fun
                

            U_opt, V_opt = self.numerical_bell(x_opt, True)





        print(f"\nOPTIMAL U:\n {U_opt}")
        print(f"\nOPTIMAL V:\n {V_opt}")
        print(f'\nOPTIMAL BELL: {bell_opt}')

        return U_opt, V_opt, bell_opt

    def full_optimize(self, N, method):
        '''
        Searches through bounds from -N to N
        '''
        unrotated_bell = self.black_box_func(np.zeros(16))
        print(f"UNOPTIMIZED BELL EXPECTANCY: {unrotated_bell}")

        results = {"Trial": [0], "Bounds": [[0,0]], "U": [np.array([[1,0,0],[0,1,0],[0,0,1]])], "V": [np.array([[1,0,0],[0,1,0],[0,0,1]])], "Bell": [unrotated_bell]}

        count = 1
        for i in range(1,N):
            ub = np.array([i] * 16)
            for j in range(1,N):
                lb = np.array([-j] * 16)

                bounds = [lb[0],ub[0]]
                print(f"\n\nTRIAL{count} bounds {bounds}")
                
                U, V, bell = self.run_optimisation(lb,ub, method)
                
                # Update results dict
                results["Trial"].append(count)
                results["Bounds"].append(bounds)
                results["U"].append(U)
                results["V"].append(V)
                results["Bell"].append(bell)
                   
                count += 1

        best_bell = np.max(results["Bell"])
        ind = results["Bell"].index(best_bell)

        best_bound = results["Bounds"][ind]
        best_U = results["U"][ind]
        best_V = results["V"][ind]

        df = pd.DataFrame.from_dict(results)

        print("OPTIMIZATION SUMMARY\n")
        print(df)
        print(f"\nBEST BOUNDS:\n {best_bound}")
        print(f"\nOPTIMAL U:\n {best_U}")
        print(f"\nOPTIMAL V:\n {best_V}")
        print(f'\nOPTIMAL BELL: {best_bell}')