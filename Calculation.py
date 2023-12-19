import math
import numpy as np
from MesData import MesData
from Calkowanie import Calkowanie


class Calculation:
    def __init__(self):
        self.calkowanie = Calkowanie()

    def L(self, x, y):
        lengths = []
        for first in range(4):
            last = first+1 if first < 3 else 0
            lengths.append( math.sqrt( ( ( x[first] - x[last] )**2 + ( y[first] - y[last] )**2 ) ) )
        return lengths
    
    def surfaces(self, x, y, pc,alfa, t_ot, data, k_t = 25):
        det =  [Calkowanie.detJ( self.calkowanie.matrix_dx_dksi_dyd_ksi(x, y, ind)) for ind in range(4)] # type: ignore
        # inv_det = [Calkowanie.detJ(np.linalg.inv(self.calkowanie.matrix_dx_dksi_dyd_ksi(x, y, ind))) for ind in range(4)]
        jakobians = [self.calkowanie.matrix_dx_dksi_dyd_ksi(x, y, i) for i in range(4) ],
        res_dict = {
            'dNdx': [np.array( self.calkowanie.mat_dN_dx(x, y, det[i])) for i in range(len(det)) ][0],
            'dNdy': [np.array( self.calkowanie.mat_dN_dy(x, y, det[i])) for i in range(len(det)) ][0],
            'Jakobian': jakobians,
            # 'Jakobian_inv': [np.linalg.inv(self.calkowanie.matrix_dx_dksi_dyd_ksi(x, y, ind)) for ind in range(4)],
            'H': sum( [self.calkowanie.H_pc_N(x,y, det[i], i, k_t, 1/det[i])  for i in range(4)] ),
            # 'H_test':  sum( [self.calkowanie.H_pc_N(x,y, det[i], i, k_t, 1/inv_det[i])  for i in range(4)] ),
            'Hbc': self.Hbc_calulation(x, y, pc, alfa),
            'P': self.P_vector(x, y, pc, alfa, t_ot),
            'C' : self.C_calulation(jakobians[0], data)
        }
        return res_dict
        
    def C_calulation(self, jak, data):
        det_j = Calkowanie.detJ(jak)
        N_range = [ self.calkowanie.N_range(pc) for pc in self.calkowanie.nodes_point()]
        N_sum = sum([ np.dot( N_range[i].reshape(4,1), N_range[i].reshape(1,4) )   for i in range( len( N_range))])
        return data['Density'] * data['SpecificHeat'] * N_sum * 1/det_j
        
    def Hbc_calulation(self, x, y, pc, alfa):
        lengths =  self.L(x, y)
        Hbc = []
        for i in range(len(pc)):
            Pcs = pc[i]
            N_range = [ self.calkowanie.N_range(pc) for pc in Pcs]
            Hbc.append( self.test_Hbc( N_range, alfa, lengths[i]/2 ) )
        return sum(Hbc)

    def test_Hbc(self, n_range, alfa, length):
        dot_nranges = [ np.dot( n_range[i].reshape(4,1), n_range[i].reshape(1,4) ) for i in range( len( n_range))]
        return alfa * sum(dot_nranges) * length

    def P_vector(self, x, y, pc, alfa, t_ot):
        lengths = self.L(x, y)
        p_vector = []
        for i in range(len(pc)):
            n_range = [self.calkowanie.N_range(p)  for p in pc[i]]
            p_vector.append(self.P(alfa, n_range, t_ot, lengths[-i]/2))
        return sum(p_vector)

    def P(self, alfa, n_ranges, t_ot, detJ):
        return sum([n.reshape(4,1)*t_ot  for n in n_ranges])*detJ*alfa
