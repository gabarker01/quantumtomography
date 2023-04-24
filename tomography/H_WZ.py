import ROOT
import numpy as np
import math

class QuantumTomographyH_WZ():

    '''
    Class for reconstructing bipartite density matrix of a WZ system from pp collision.
    For theoretical justification, see https://arxiv.org/abs/2209.13990.
    '''

    def __init__(self, tree):
        '''
        :param tree: TTree containing event data for H and subsequent WZ decays.

        '''
        self.tree = tree
        self.entry_no = tree.GetEntries()

        # Define objects to store WignerP values of muon and electron
        self.pZ = np.array([0] * 8)
        self.pW = np.array([0] * 8)
        self.cov = np.zeros((8,8), dtype=float, order='C')
        self.cov_sym = np.zeros((8,8), dtype=float, order='C')
        self.bell_sum = np.zeros((9,9), dtype=float, order='C')
    
    def tree_analysis(self):
        '''
        Analyses TTree containing event data.

        :returns self.wig_plus: numpy array of average values of wigner functions for the W+ boson.
        :returns self.wig_minus: numpy array of average values of wigner functions for the W- boson.
        :returns self.gmarray: numpy array of average bipartite density matrix.
        
        '''
        # Get particle IDs
        pid = self.tree.GetLeaf("Particle.PID")
        
        # Look for electron/electron neutrino and muon/antimuon pairs
        ids = [13,-13,12,-11]

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
                pZ_event, pW_event = self.analyse_event(event, pid_array)

                # Add to sum if valid event
                if not np.isnan(pZ_event).all() and not np.isnan(pW_event).all():
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

                    # Calculate correlations
                    for i in range(8):
                        for j in range(8):
                            self.cov[i][j] = self.cov[i][j] + (pW_event[i] * pZ_event[j])
                            self.cov_sym[i][j] = self.cov_sym[i][j] + (pW_event[i] * pZ_event[j]) + (pW_event[j] * pZ_event[i])
                
                    # Calculate event contributions to matrix
                    gmarray_event = np.zeros((9,9), dtype=float, order='C')
                    for i in range(8):
                        gmarray_event[0][i+1] = 0.5 * pZ_event[i] # W-
                        gmarray_event[i+1][0] = 0.5 * pW_event[i] # W+
                        for j in range(8):
                            gmarray_event[i+1][j+1] = 0.25 * (pW_event[i] * pZ_event[j])

                    self.bell_sum += gmarray_event

                    # Add bell operator contribution to list
                    bell_event = self.CGLMP_test(gmarray_event)
                    bell_events.append(bell_event)
                
                
                else:
                    # Invalid event due to sqrt of <0
                    bad_events.append(event_no)
                    bad_count += 1

        np.savetxt("WZ_BellXY.csv", bell_events, delimiter = ",")
        print("SUMMARY")
        print("Count:", count)
        print("Bad count:", bad_count)
        print("Bad event numbers:", bad_events)
        # Return average wigner and gmarray values as class variables
        self.pZ = self.pZ / count
        self.pW = self.pW / count
        self.cov = self.cov / count
        self.cov_sym = self.cov_sym / count
        
        self.bell_sum = self.bell_sum / count

        print("Symettric gmarray from method for Bell:")
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
        id_1 = int(np.where(pidarray == 12)[0][0])
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
        wig_4 = self.wig(etax_4, etay_4, etaz_4) # aelectron

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

        
        return wig_3, wig_4

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