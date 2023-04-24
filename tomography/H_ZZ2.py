import ROOT
import numpy as np
import math

class QuantumTomographyH_ZZ():

    '''
    Class for reconstructing bipartite density matrix of a ZZ system from H decay.
    For theoretical justification, see https://arxiv.org/abs/2209.13990.
    '''

    def __init__(self, tree):
        '''
        :param tree: TTree containing event data for H and subsequent ZZ decays.

        '''
        self.tree = tree
        self.entry_no = tree.GetEntries()

        # Define objects to store WignerP values of muon and electron
        self.pZ = np.array([0] * 8)
        self.pW = np.array([0] * 8)
        self.cov = np.zeros((8,8), dtype=float, order='C')
        self.cov_sym = np.zeros((8,8), dtype=float, order='C')
        self.bell_sum = np.zeros((9,9), dtype=float, order='C')

        # Required for optimized bell tests
        self.gmlist = [self.gellmann(2,1,3),
                self.gellmann(1,2,3),
                self.gellmann(1,1,3),
                self.gellmann(3,1,3),
                self.gellmann(1,3,3),
                self.gellmann(3,2,3),
                self.gellmann(2,3,3),
                self.gellmann(2,2,3)]
    
    def tree_analysis(self):
        '''
        Analyses TTree containing event data.

        :returns self.wig_plus: numpy array of average values of wigner functions for the W+ boson.
        :returns self.wig_minus: numpy array of average values of wigner functions for the W- boson.
        :returns self.gmarray: numpy array of average bipartite density matrix.
        
        '''
        # Introduce boolean for having a mass cut
        self.cut = True
        self.mass_cut = 5
        self.ZZ_mass_cut = 1e+06

        # Get particle IDs
        pid = self.tree.GetLeaf("Particle.PID")
        
        # Look for electron/electron neutrino and muon/antimuon pairs
        ids = [11,-11,13,-13]

        # Main analysis loop
        count = 0
        bad_count = 0
        bad_events = []
        bell_events = []
        for event_no, event in enumerate(self.tree):
            self.printout = False
            if event_no < 5:
                self.printout = True
            if event_no % 1000 == 0:
                print(f"Event {event_no} / {self.entry_no} {int((event_no/self.entry_no) * 100)}% complete.")
            pid_array = np.array([pid.GetValue(i) for i in range(10)])
            good_event = self.event_type(pid_array, ids)

            if good_event:
                # Get Wigner P functions
                pZ_event, pW_event, good_event = self.analyse_event(event, pid_array)

                # Add to sum if valid event
                if not np.isnan(pZ_event).all() and not np.isnan(pW_event).all() and good_event:
                    count += 1

                    # Add event contributions to sums
                    self.pZ = np.array([self.pZ[i] + pZ_event[i] for i in range(8)])
                    self.pW = np.array([self.pW[i] + pW_event[i] for i in range(8)])
                  
                    if self.printout:
                        gmarray_asym = np.zeros((9,9), dtype=float, order='C')
                        for i in range(8):
                            gmarray_asym[0][i+1] = 0.5 * pZ_event[i] #Z
                            gmarray_asym[i+1][0] = 0.5 * pW_event[i] #W+
                            for j in range(8):
                                gmarray_asym[i+1][j+1] = 0.25 * (pW_event[i] * pZ_event[j])
                        print(f"Contribution {event_no}")
                        print(gmarray_asym)
                    
                    # Calculate event array for Bell tests
                    sym_event = np.zeros((9,9), dtype=float, order='C')
                    a_sym = 0.25 * (pZ_event + pW_event)

                    # Calculate correlations and fill bell event matrix
                    for i in range(8):
                        sym_event[0][i+1] = a_sym[i]
                        sym_event[i+1][0] = a_sym[i]
                        for j in range(8):
                            self.cov[i][j] = self.cov[i][j] + (pW_event[i] * pZ_event[j])

                            self.cov_sym[i][j] = self.cov_sym[i][j] + (pW_event[i] * pZ_event[j]) + (pW_event[j] * pZ_event[i])
                            sym_event[i+1][j+1] = (1/8) * ((pW_event[i] * pZ_event[j]) + (pW_event[j] * pZ_event[i]))
                    
                    sym_event = sym_event / 2 # EXTRA FACTOR OF 0.5
                    
                    # Add to cross check
                    self.bell_sum += sym_event

                    # Calculate bell value for event on symmetric matrix
                    event_bell = self.CGLMP_test_optimised(sym_event)
                    bell_events.append(event_bell)

                else:
                    # Invalid event due to sqrt of <0
                    bad_events.append(event_no)
                    bad_count += 1
        
        np.savetxt("ZZ_Bell_OPTIMIZED.csv", bell_events, delimiter = ",")
    
        print("SUMMARY")
        print("Count:", count)
        print("Bad count:", bad_count)
        print("Bad event numbers:", bad_events)
        print(f"Using mass cuts? {str(self.cut)}")
        if self.cut:
            print(f"Cutting at min dilepton mass {self.mass_cut} GeV and ZZ invariant mass in range 0 - {self.ZZ_mass_cut} GeV.")
        # Return average wigner and gmarray values as class variables
        self.pZ = self.pZ / count
        self.pW = self.pW / count
        self.cov = self.cov / count
        self.cov_sym = self.cov_sym / count
        
        self.bell_sum = self.bell_sum / count

        print("Symettric gmarray from method for bell (cross check):")
        print(self.bell_sum)

        return self.pZ, self.pW, self.cov, self.cov_sym


    def analyse_event(self, event, pidarray):
        '''
        Analyses single recorded event

        :param event: Event object from TTree
        :param type: int indicating decay mode
        :param pid_array: numpy array

        :returns a_sym: symetric single matrix elements, corresponding to a_i = b_i in GGM basis
        :returns gmarray_event: bipartite density matrix for event
        
        '''

        # Label neutrino as 1, muon as 2, antimuon as 3 and positron as 4
        id_1 = int(np.where(pidarray == 11)[0][0])
        id_2 = int(np.where(pidarray == 13)[0][0])
        id_3 = int(np.where(pidarray == -13)[0][0])
        id_4 = int(np.where(pidarray == -11)[0][0])
        
        # Get 4 momenta of leptons in lab frame
        lab_1 = self.get_ROOT_Lorentz_vector(event, id_1)
        lab_2 = self.get_ROOT_Lorentz_vector(event, id_2)
        lab_3 = self.get_ROOT_Lorentz_vector(event, id_3)
        lab_4 = self.get_ROOT_Lorentz_vector(event, id_4)

        # Construct Lorentz vectors of W, Z in the lab frame. Label ZMF as H.
        lab_W = lab_1 + lab_4
        lab_Z = lab_2 + lab_3
        lab_H = lab_W + lab_Z

        # Construct numpy arrays of bosons in lab frame
        lab_W_array = np.array([[lab_W.E()], [lab_W.Px()], [lab_W.Py()], [lab_W.Pz()]])
        lab_Z_array = np.array([[lab_Z.E()], [lab_Z.Px()], [lab_Z.Py()], [lab_Z.Pz()]])
        lab_H_array = np.array([[lab_H.E()], [lab_H.Px()], [lab_H.Py()], [lab_H.Pz()]])

        # Apply cuts
        good_event = True
        if self.cut:
            if lab_H.M() < 0 or lab_H.M() > self.ZZ_mass_cut:
                good_event = False
            if lab_W.M() < self.mass_cut:
                good_event = False
            if lab_Z.M() < self.mass_cut:
                good_event = False

        # Construct boost to bosonic ZMF and boost bosons
        boost_lab_to_H = self.get_Lorentz_boost(lab_H_array)
        H_W_array = boost_lab_to_H.dot(lab_W_array)
        H_Z_array = boost_lab_to_H.dot(lab_Z_array)

        # Find k - the direction of W (e neut) in ZMF
        k_raw = [float(H_W_array[i+1]) for i in range(3)]
        k_mag = three_dot(k_raw, k_raw)**0.5
        k = np.array([k_raw[i] / k_mag for i in range(3)])

        # Find p - direction of initial beam in ZMF
        beam = np.array([[1],[0],[0],[1]])
        H_beam_array = boost_lab_to_H.dot(beam)
        
        p_raw = [float(H_beam_array[i+1]) for i in range(3)]
        p_mag = three_dot(p_raw, p_raw)**0.5
        p = np.array([p_raw[i] / p_mag for i in range(3)])

        # Complete orthonormal coordinate system k, r, n
        y = three_dot(p,k)
        
        r_mag = (1 - y**2)**0.5
        r = np.array([((p[i] - y*k[i]) / r_mag) for i in range(3)])

        n = cross(p,k) * (1/r_mag)

        # Boost positive (anti-)leptons into respective W rest frames and find their directions
        boost_H_to_W = self.get_Lorentz_boost(H_W_array)
        boost_H_to_Z = self.get_Lorentz_boost(H_Z_array)

        lab_3_array = np.array([[lab_3.E()], [lab_3.Px()], [lab_3.Py()], [lab_3.Pz()]]) # amuon
        lab_4_array = np.array([[lab_4.E()], [lab_4.Px()], [lab_4.Py()], [lab_4.Pz()]]) # aelectron

        H_3_array = boost_lab_to_H.dot(lab_3_array)
        H_4_array = boost_lab_to_H.dot(lab_4_array)
        Z_3_array = boost_H_to_Z.dot(H_3_array)
        W_4_array = boost_H_to_W.dot(H_4_array)

        dir_3_raw = [Z_3_array[i+1] for i in range(3)]
        dir_4_raw = [W_4_array[i+1] for i in range(3)]
        dir_3_mag = three_dot(dir_3_raw, dir_3_raw)**0.5
        dir_4_mag = three_dot(dir_4_raw, dir_4_raw)**0.5

        dir_3 = np.array([dir_3_raw[i]/dir_3_mag for i in range(3)])
        dir_4 = np.array([dir_4_raw[i]/dir_4_mag for i in range(3)])

        # Find direction cosines for tomography

        # etax = sin(theta)cos(phi)
        etax_3 = n.dot(dir_3)
        etax_4 = n.dot(dir_4)

        # etay = sin(theta)sin(phi)
        etay_3 = r.dot(dir_3)
        etay_4 = r.dot(dir_4)

        # etaz = cos(theta)
        etaz_3 = k.dot(dir_3)
        etaz_4 = k.dot(dir_4)

        # Get generalised wigner values
        wig_3 = self.zz_wig(etax_3, etay_3, etaz_3) # amuon
        wig_4 = self.zz_wig(etax_4, etay_4, etaz_4) # aelectron

        if self.printout:
            print(f"EVENT CONTRIBUTIONS")
            print("\nLepton directions:")
            print(dir_3)
            print(dir_4)
            print("Basis nrk:", [n,r,k])
            print("Wigner amuon:")
            print(wig_3)
            print("Wigner aelectron")
            print(wig_4)

        
        return wig_3, wig_4, good_event

    def zz_wig(self, etax, etay, etaz):
        '''
        Returns wigner symbol value, generalised for case of leptonic Z decays

        :params etai: direction cosines of lepton in Z ZMF

        :returns zz_wig_event: numpy array of generalised Wigner P values for event
        
        '''
        # Assign left and right chiral couplings
        cr = 0.233
        cl = -0.273

        # Calculate standard positive wigner plus value for event
        theta = math.acos(etaz)
        wig_plus_event = np.array([(math.sqrt(2))*(5*etaz + 1)*etax,
					(math.sqrt(2))*(5*etaz +1)*etay,
					((etaz) +(15/4)*(math.cos(2*theta)) + 5/4),
					(5)*(etax**2 -etay**2),
					10*etax*etay,
					(math.sqrt(2))*(1 - 5*etaz)*etax,
					(math.sqrt(2))*(1 - 5*etaz)*etay,
					(0.25/math.sqrt(3))*((12*etaz) -15*(math.cos(2*theta)) - 5)])

        # Calulate matrix A to generalise wigner functions to reduce spin dependancy
        A = np.array([[(cr**2), 0, 0, 0, 0, (cl**2), 0, 0],
            [0, (cr**2), 0, 0, 0, 0, (cl**2), 0],
            [0, 0, (cr**2) - 0.5*(cl**2), 0, 0, 0, 0, 0.5*math.sqrt(3)*(cl**2)],
            [0, 0, 0, (cr**2) - (cl**2), 0, 0, 0, 0],
            [0, 0, 0, 0, (cr**2) - (cl**2), 0, 0, 0],
            [(cl**2), 0, 0, 0, 0, (cr**2), 0, 0],
            [0, (cl**2), 0, 0, 0, 0, (cr**2), 0],
            [0, 0, 0.5*math.sqrt(3)*(cl**2), 0, 0, 0, 0, 0.5*(cl**2) + (cr**2)]
        ])
        
        A = A / ((cr**2) - (cl**2))

        # Obtain generalised wigner functions for leptonic Z decay
        zz_wig_event = A.dot(wig_plus_event)

        return zz_wig_event

    def wig(self, etax, etay, etaz):
        '''
        Returns standard wigner symbol value.

        :params etai: direction cosines of lepton in boson ZMF

        :returns zz_wig_event: numpy array of standard Wigner P values for event
        
        '''
        # Calculate standard positive wigner plus value for event
        theta = math.acos(etaz)
        wig_plus_event = np.array([(math.sqrt(2))*(5*etaz + 1)*etax,
					(math.sqrt(2))*(5*etaz +1)*etay,
					((etaz) +(15/4)*(math.cos(2*theta)) + 5/4),
					(5)*(etax**2 -etay**2),
					10*etax*etay,
					(math.sqrt(2))*(1 - 5*etaz)*etax,
					(math.sqrt(2))*(1 - 5*etaz)*etay,
					(0.25/math.sqrt(3))*((12*etaz) -15*(math.cos(2*theta)) - 5)])

        return wig_plus_event
    
    def get_ROOT_Lorentz_vector(self, event, id):
        '''
        Returns ROOT Lorentz data given particle id.
        
        :param event: Event object from TTree
        :param id: index of particle in event

        :returns lorentz: ROOT TLorentzVector for particle id in event.

        '''
        lorentz = ROOT.TLorentzVector()

        px = float(self.tree.GetLeaf("Particle.Px").GetValue(id))
        py = float(self.tree.GetLeaf("Particle.Py").GetValue(id))
        pz = float(self.tree.GetLeaf("Particle.Pz").GetValue(id))
        e = float(self.tree.GetLeaf("Particle.E").GetValue(id))

        lorentz.SetPxPyPzE(px,py,pz,e)

        return lorentz
    
    def get_Lorentz_boost(self, array):
        '''
        Returns numpy array of a Lorentz boost matrix. Initial frame is frame of particle described by input 4-vector, boosted frame is particle's
        rest frame.
        
        :param array: numpy array of 4-momentum of particle

        :returns matrix: numpy array of lorentz transformation matrix

        '''
        # Invariant mass M^2 = E^2 - p^2; E = gamma * M
        p = (array[1]**2 + array[2]**2 + array[3]**2)**0.5
        M = (array[0]**2 - p**2)**0.5
        gamma = array[0] / M
        
        # Component-wise v_i = p_i / E
        v = [array[i+1] / array[0] for i in range(3)]
        v_mag = three_dot(v,v)**0.5

        # Construct rows of Lorentz matrix

        row_1 = [gamma, -gamma * v[0], -gamma * v[1], -gamma * v[2]]
        row_2 = [-gamma*v[0], 1 + (gamma - 1)*(v[0]**2)*(v_mag**-2), (gamma - 1) * v[0]*v[1] * (v_mag**-2), (gamma - 1) * v[0]*v[2] * (v_mag**-2)]
        row_3 = [-gamma*v[1], (gamma - 1) * v[1]*v[0] * (v_mag**-2), 1 + (gamma - 1)*(v[1]**2)*(v_mag**-2), (gamma - 1) * v[1]*v[2] * (v_mag**-2)]
        row_4 = [-gamma*v[2], (gamma - 1) * v[2]*v[0] * (v_mag**-2), (gamma - 1) * v[2]*v[1] * (v_mag**-2), 1 + (gamma - 1)*(v[2]**2)*(v_mag**-2)]

        matrix = np.array([row_1, row_2, row_3, row_4])

        return matrix.reshape((4,4))
        
    def event_type(self, array, ids):
        '''
        Determines event type by specifying all particles in a specific array must occur exactly once.
        
        :param array: array containing particle ids in event.
        :param ids: list containing id numbers of required particles.

        :returns type: Bool True if all ids in array exactly once.
        '''
        type = False
        type_count = 0

        # Require all particles to appear once only
        for id in ids:
            if np.sum(array == id) == 1:
                type_count +=1
        
        # Require all particles to be present
        if type_count == len(ids):
            type = True
        
        return type

    def CGLMP_test(self, gmarray):
        '''
        Performs CGLMP Bell violation test by finding the expectation value of Bell operator. Values above 2 are 
        forbidden in local realist interpretations of QM.

        :param gmarray: numpy array holding density matrix from tomography

        :returns bell_value = expectancy value of bell operator
        
        '''

        bell_value = -(4/3) * (math.sqrt(3) * gmarray[1][1] + math.sqrt(3)*gmarray[1][6] + math.sqrt(3)* gmarray[2][2] + math.sqrt(3)*gmarray[2][7] - 
        3 * gmarray[4][4] - 3 *gmarray[5][5] + math.sqrt(3) * (gmarray[6][1] + gmarray[6][6] + gmarray[7][2] + gmarray[7][7]))
        
        return bell_value

    def gellmann(self, j, k, d):
        """
        Returns a generalized Gell-Mann matrix of dimension d. According to the
        convention in *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008),

        """

        if j > k:
            gjkd = np.zeros((d, d), dtype=np.complex128)
            gjkd[j - 1][k - 1] = 1
            gjkd[k - 1][j - 1] = 1
        elif k > j:
            gjkd = np.zeros((d, d), dtype=np.complex128)
            gjkd[j - 1][k - 1] = -1j
            gjkd[k - 1][j - 1] = 1j
        elif j == k and j < d:
            gjkd = np.sqrt(2/(j*(j + 1)))*np.diag([1 + 0j if n <= j
                                                else (-j + 0j if n == (j + 1)
                                                        else 0 + 0j)
                                                for n in range(1, d + 1)])
        else:
            gjkd = np.diag([1 + 0j for n in range(1, d + 1)])

        return gjkd
    
    def trace(self, array):
        dim = array.shape[0]
        if dim != array.shape[1]:
            return "Cannot do trace of non-square."
        else:
            tr = 0
            for i in range(dim):
                tr += array[i][i]
            return tr

    def CGLMP_test_optimised(self, gmarray):
        '''
        optimized CGLMP test on gmarray
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
            
        rho = A + B + C + D

        B_opt_ZZ = np.array([[-0.367561 - 2.77556e-17j, 0.0578448 + 0.0706434j, 
                            0.184652 - 0.578049j, 0.0421675 - 0.0377658j, 
                            0.0724447 - 0.0126325j, 0.0720173 - 0.12949j, 
                            0.325766 + 0.531121j, 
                            0.0889856 + 0.11025j, -0.274209 + 0.0566326j], [0.0578448 - 
                            0.0706434j, -0.471755 - 2.77556e-17j, -0.0490141 - 
                            0.124177j, -0.725454 + 0.0877738j, -0.0871497 - 0.0707975j, 
                            0.315683 + 0.689787j, 
                            0.039497 + 0.0306038j, -0.48978 + 0.068238j, -0.0791048 + 
                            0.124966j], [0.184652 + 0.578049j, -0.0490141 + 0.124177j, 
                            0.839317 - 1.82146e-17j, 
                            0.0220338 + 0.0709058j, -1.09192 + 0.00527521j, 
                            0.0449822 + 0.108563j, 
                            0.797091 - 0.0181389j, -0.00691234 + 0.0785585j, 
                            0.164014 - 0.599359j], [0.0421675 + 0.0377658j, -0.725454 - 
                            0.0877738j, 
                            0.0220338 - 0.0709058j, -0.508439 - 2.77556e-17j, -0.0588711 + 
                            0.0313346j, -0.492398 + 0.0610867j, 0.0162874 + 0.116383j, 
                            0.0770381 - 0.758385j, -0.0795869 - 0.109243j], [0.0724447 + 
                            0.0126325j, -0.0871497 + 0.0707975j, -1.09192 - 
                            0.00527521j, -0.0588711 - 0.0313346j, 
                            0.975604 + 4.33681e-19j, 
                            0.0818088 + 0.0833957j, -1.07402 + 0.0108619j, 
                            0.0402002 - 0.0458998j, 
                            0.0861676 + 0.00248727j], [0.0720173 + 0.12949j, 
                            0.315683 - 0.689787j, 
                            0.0449822 - 0.108563j, -0.492398 - 0.0610867j, 
                            0.0818088 - 0.0833957j, -0.467165 - 5.55112e-17j, -0.0321866 + 
                            0.0355066j, -0.730146 - 0.0703197j, -0.0564876 - 
                            0.0704827j], [0.325766 - 0.531121j, 0.039497 - 0.0306038j, 
                            0.797091 + 0.0181389j, 
                            0.0162874 - 0.116383j, -1.07402 - 0.0108619j, -0.0321866 - 
                            0.0355066j, 0.876 + 6.93889e-18j, 0.0010263 - 0.101978j, 
                            0.307746 + 0.516962j], [0.0889856 - 0.11025j, -0.48978 - 
                            0.068238j, -0.00691234 - 0.0785585j, 0.0770381 + 0.758385j, 
                            0.0402002 + 0.0458998j, -0.730146 + 0.0703197j, 
                            0.0010263 + 0.101978j, -0.503849 - 2.77556e-17j, -0.0327947 + 
                            0.0407814j], [-0.274209 - 0.0566326j, -0.0791048 - 0.124966j, 
                            0.164014 + 0.599359j, -0.0795869 + 0.109243j, 
                            0.0861676 - 0.00248727j, -0.0564876 + 0.0704827j, 
                            0.307746 - 0.516962j, -0.0327947 - 0.0407814j, -0.372151 + 
                            2.77556e-17j]])
        
        return self.trace(np.matmul(rho, B_opt_ZZ))

# Define helper functions
def three_dot(l1, l2):
    '''
    Takes dot product of two subsciptable 3D objects

    '''
    sum = 0

    for i in range(3):
        sum += l1[i] * l2[i]
    
    return sum

def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    '''
    Returns cross product of two numpy arrays
    Work-around for numpy ReturnNone bug.
    
    '''

    return np.cross(a,b)