import ROOT
import numpy as np
import matplotlib.pyplot as plt
import math

class QuantumTomographyH_WW():

    '''
    Class for reconstructing bipartite density matrix of a W+ W- system from Higgs decay.
    For theoretical justification, see https://arxiv.org/abs/2209.13990.
    '''

    def __init__(self, tree):
        '''
        :param tree: TTree containing event data for H and subsequent WW decays.

        '''
        self.tree = tree
        self.entry_no = tree.GetEntries()
        
        # Define objects to store contributions to Wigner-P functions and GM Matrix from each event
        self.pW1 = np.array([0] * 8)
        self.pW2 = np.array([0] * 8)
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
        # Get particle IDs
        pid = self.tree.GetLeaf("Particle.PID")
        
        # Define type-1 decays to be to electron and anti-muon, and type-2 decays to be to positron and muon
        ids_1 = [-13, 14, 11, -12]
        ids_2 = [13, -14, -11, 12]

        # Main analysis loop
        type_1_count = 0
        type_2_count = 0
        bad_events = 0
        bad_count = 0
        count = 0
        bell_events = []
        for event_no, event in enumerate(self.tree):
            if event_no % 1000 == 0:
                print(f"Event {event_no} / {self.entry_no} {int((event_no/self.entry_no) * 100)}% complete.")
            pid_array = np.array([pid.GetValue(i) for i in range(10)])

            type_1 = self.event_type(pid_array, ids_1)
            type_2 = self.event_type(pid_array, ids_2)

            if type_1:
                pW1_event, pW2_event = self.analyse_event(event, 1, pid_array)
                type_1_count += 1
            elif type_2:
                pW1_event, pW2_event,  = self.analyse_event(event, 2, pid_array)
                type_2_count += 1
            
            if type_1 or type_2:
                if not np.isnan(pW1_event).all() and not np.isnan(pW2_event).all():           
                    count += 1  
                    # Add event contributions to sums
                    self.pW1 = np.array([self.pW1[i] + pW1_event[i] for i in range(8)])
                    self.pW2 = np.array([self.pW2[i] + pW2_event[i] for i in range(8)])
                    
                    # Calculate correlations
                    for i in range(8):
                        for j in range(8):
                            self.cov[i][j] = self.cov[i][j] + (pW2_event[i] * pW1_event[j])
                            self.cov_sym[i][j] = self.cov_sym[i][j] + (pW2_event[i] * pW1_event[j]) + (pW2_event[j] * pW1_event[i])
                   
                    # Calculate event contributions to matrix
                    gmarray_event = np.zeros((9,9), dtype=float, order='C')
                    for i in range(8):
                        gmarray_event[0][i+1] = 0.5 * pW1_event[i] # W-
                        gmarray_event[i+1][0] = 0.5 * pW2_event[i] # W+
                        for j in range(8):
                            gmarray_event[i+1][j+1] = 0.25 * (pW2_event[i] * pW1_event[j])

                    self.bell_sum += gmarray_event

                    # Add bell operator contribution to list
                    bell_event = self.CGLMP_test_optimised(gmarray_event)
                    bell_events.append(bell_event)

                
                else:
                    # Invalid event due to sqrt of <0
                    bad_events.append(event_no)
                    bad_count += 1
        
        np.savetxt("WW_Bell_OPTIMIZED.csv", bell_events, delimiter = ",")
        print("SUMMARY")
        print("Type 1 event count:", type_1_count)
        print("Type 2 event count:", type_2_count)
        
        # Return average wigner P values
        self.pW1 = self.pW1 / count
        self.pW2 = self.pW2/ count
        self.cov = self.cov / count
        self.cov_sym = self.cov_sym / count

        self.bell_sum = self.bell_sum / count

        print("Symettric gmarray from method for Bell:")
        print(self.bell_sum)

        return self.pW1, self.pW2, self.cov, self.cov_sym


    def analyse_event(self, event, type, pidarray):
        '''
        Analyses single recorded event

        :param event: Event object from TTree
        :param type: int indicating decay mode
        :param pid_array: numpy array

        :returns wig_plus_event: values of wigner P functions for W+ boson in event
        :returns wig_minus_event: values of wigner P functions for W- boson in event
        :returns gmarray_event: bipartite density matrix for event
        
        '''

        if type == 1:
            # Label electron as 1, anti-muon as 2. muon-neutrino as 3 and anti-electron-neutrino as 4
            id_1 = int(np.where(pidarray == 11)[0][0])
            id_2 = int(np.where(pidarray == -13)[0][0])
            id_3 = int(np.where(pidarray == 14)[0][0])
            id_4 = int(np.where(pidarray == -12)[0][0])
        elif type == 2:
            # Label muon as 1, positron as 2. electron-neutrino as 3 and anti-muon-neutrino as 4
            id_1 = int(np.where(pidarray == 13)[0][0])
            id_2 = int(np.where(pidarray == -11)[0][0])
            id_3 = int(np.where(pidarray == 12)[0][0])
            id_4 = int(np.where(pidarray == -14)[0][0])
        
        # Get 4 momenta of leptons in lab frame
        lab_1 = self.get_ROOT_Lorentz_vector(event, id_1)
        lab_2 = self.get_ROOT_Lorentz_vector(event, id_2)
        lab_3 = self.get_ROOT_Lorentz_vector(event, id_3)
        lab_4 = self.get_ROOT_Lorentz_vector(event, id_4)

        # Construct Lorentz vectors of W+ (W2), W- (W1) and Higgs (H) bosons in the lab frame
        lab_W1 = lab_1 + lab_4
        lab_W2 = lab_2 + lab_3
        lab_H = lab_W1 + lab_W2

        # Construct numpy arrays of bosons in lab frame
        lab_W1_array = np.array([[lab_W1.E()], [lab_W1.Px()], [lab_W1.Py()], [lab_W1.Pz()]])
        lab_W2_array = np.array([[lab_W2.E()], [lab_W2.Px()], [lab_W2.Py()], [lab_W2.Pz()]])
        lab_H_array = np.array([[lab_H.E()], [lab_H.Px()], [lab_H.Py()], [lab_H.Pz()]])

        # Construct boost to bosonic ZMF / Higgs rest frame and boost bosons
        boost_lab_to_H = self.get_Lorentz_boost(lab_H_array)
        H_W1_array = boost_lab_to_H.dot(lab_W1_array)
        H_W2_array = boost_lab_to_H.dot(lab_W2_array)

        # Find k - the direction of W2  in Higgs rest frame
        k_raw = [float(H_W2_array[i+1]) for i in range(3)]
        k_mag = three_dot(k_raw, k_raw)**0.5
        k = np.array([k_raw[i] / k_mag for i in range(3)])

        # Find p - direction of initial beam in Higgs rest frame
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
        boost_H_to_W1 = self.get_Lorentz_boost(H_W1_array)
        boost_H_to_W2 = self.get_Lorentz_boost(H_W2_array)

        lab_1_array = np.array([[lab_1.E()], [lab_1.Px()], [lab_1.Py()], [lab_1.Pz()]]) # amuon
        lab_2_array = np.array([[lab_2.E()], [lab_2.Px()], [lab_2.Py()], [lab_2.Pz()]]) # aelectron

        H_1_array = boost_lab_to_H.dot(lab_1_array)
        H_2_array = boost_lab_to_H.dot(lab_2_array)
        W1_1_array = boost_H_to_W1.dot(H_1_array)
        W2_2_array = boost_H_to_W2.dot(H_2_array)

        dir_1_raw = [W1_1_array[i+1] for i in range(3)]
        dir_2_raw = [W2_2_array[i+1] for i in range(3)]
        dir_1_mag = three_dot(dir_1_raw, dir_1_raw)**0.5
        dir_2_mag = three_dot(dir_2_raw, dir_2_raw)**0.5

        dir_1 = np.array([dir_1_raw[i]/dir_1_mag for i in range(3)])
        dir_2 = np.array([dir_2_raw[i]/dir_2_mag for i in range(3)])

        # Find direction cosines for tomography

        # etax = sin(theta)cos(phi)
        etax_1 = n.dot(dir_1)
        etax_2 = n.dot(dir_2)

        # etay = sin(theta)sin(phi)
        etay_1 = r.dot(dir_1)
        etay_2 = r.dot(dir_2)

        # etaz = cos(theta)
        etaz_1 = k.dot(dir_1)
        etaz_2 = k.dot(dir_2)

        # Get generalised wigner values
        wig_1 = self.wig_neg(etax_1, etay_1, etaz_1) # lepton-
        wig_2 = self.wig_plus(etax_2, etay_2, etaz_2) # lepton+
        
        return wig_1, wig_2
    
    def wig_plus(self, etax, etay, etaz):
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
    
    def wig_neg(self, etax, etay, etaz):
        '''
        Returns standard wigner negative symbol value.

        :params etai: direction cosines of lepton in boson ZMF

        :returns zz_wig_event: numpy array of standard Wigner P values for event
        
        '''
        # Calculate standard positive wigner plus value for event
        theta = math.acos(etaz)
        wig_minus_event = np.array([(-math.sqrt(2))*(-5*etaz + 1)*etax,
					(-math.sqrt(2))*(-5*etaz +1)*etay,
					0.25*((-4*etaz) +15*(math.cos(2*theta)) + 5),
					(5)*(etax**2 -etay**2),
					10*etax*etay,
					(-math.sqrt(2))*(1 + 5*etaz)*etax,
					(-math.sqrt(2))*(1 + 5*etaz)*etay,
					(0.25/math.sqrt(3))*((-12*etaz) -15*(math.cos(2*theta)) - 5)])

        return wig_minus_event
  
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

        B_opt_WW = np.array([[-0.678593 - 2.08167e-17j, -0.0128983 + 0.0268912j, 
                            0.050745 + 0.264804j, 0.100711 + 0.149546j, 
                            0.239093 + 0.0170966j, 
                            0.20388 + 0.239573j, -0.130472 - 0.270579j, -0.29918 + 0.283954j,
                            0.504154 + 0.180131j], [-0.0128983 - 0.0268912j, -0.718371 + 
                            0.j, 0.0204025 + 0.0428807j, -0.276429 + 0.13318j, -0.0799415 - 
                            0.101584j, 0.467649 - 0.673298j, -0.231724 - 0.321251j, 
                            0.0821254 + 0.00674888j, -0.20476 - 0.239956j], [0.050745 - 
                            0.264804j, 0.0204025 - 0.0428807j, 1.39696 + 2.9924e-17j, 
                            0.190551 + 0.232312j, -0.842566 - 0.000222304j, -0.0207693 - 
                            0.0479623j, 0.659601 - 0.00232018j, -0.193908 + 0.232211j, 
                            0.0483464 + 0.26383j], [0.100711 - 0.149546j, -0.276429 - 
                            0.13318j, 0.190551 - 0.232312j, -0.678633 + 1.38778e-17j, 
                            0.141928 - 0.107125j, 0.0805794 + 0.00784512j, 
                            0.121177 - 0.0818895j, 0.254101 + 0.657373j, 
                            0.29945 - 0.281431j], [0.239093 - 0.0170966j, -0.0799415 + 
                            0.101584j, -0.842566 + 0.000222304j, 0.141928 + 0.107125j, 
                            1.39699 + 3.25261e-18j, 
                            0.0791694 + 0.105139j, -0.731583 + 0.00463491j, -0.133109 + 
                            0.108585j, 0.240704 + 0.0176988j], [0.20388 - 0.239573j, 
                            0.467649 + 0.673298j, -0.0207693 + 0.0479623j, 
                            0.0805794 - 0.00784512j, 
                            0.0791694 - 0.105139j, -0.718354 + 5.55112e-17j, 
                            0.227976 - 0.323137j, -0.278488 - 0.132733j, 
                            0.0119318 - 0.026696j], [-0.130472 + 0.270579j, -0.231724 + 
                            0.321251j, 0.659601 + 0.00232018j, 
                            0.121177 + 0.0818895j, -0.731583 - 0.00463491j, 
                            0.227976 + 0.323137j, 
                            1.35723 + 1.73472e-18j, -0.12903 + 0.0802338j, -0.131324 - 
                            0.272649j], [-0.29918 - 0.283954j, 
                            0.0821254 - 0.00674888j, -0.193908 - 0.232211j, 
                            0.254101 - 0.657373j, -0.133109 - 0.108585j, -0.278488 + 
                            0.132733j, -0.12903 - 0.0802338j, -0.678616 + 0.j, -0.0995719 - 
                            0.14802j], [0.504154 - 0.180131j, -0.20476 + 0.239956j, 
                            0.0483464 - 0.26383j, 0.29945 + 0.281431j, 0.240704 - 0.0176988j,
                            0.0119318 + 0.026696j, -0.131324 + 0.272649j, -0.0995719 + 
                            0.14802j, -0.67861 - 1.38778e-17j]])
        
        return self.trace(np.matmul(rho, B_opt_WW))

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