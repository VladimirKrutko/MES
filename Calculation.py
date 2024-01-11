import math
import numpy as np
from Calkowanie import Calkowanie


class Calculation:
    def __init__(self, pc_number):
        self.pc_number = pc_number
        self.calkowanie = Calkowanie(pc_number) # type: ignore

    def L(self, x, y):
        lengths = []
        for first in range(4):
            last = first+1 if first < 3 else 0
            lengths.append( math.sqrt( ( ( x[first] - x[last] )**2 + ( y[first] - y[last] )**2 ) ) )
        return lengths
    
    def surfaces(self, x, y, pc, alfa, t_ot, data, L_ind, k_t):
        self.calkowanie.init_dNdx_dNdy(x, y)
        res_dict = {
            'dNdx': self.calkowanie.dNdx,
            'dNdy': self.calkowanie.dNdy,
            'Jakobian': self.calkowanie.j_matrixes,
            'H': self.calkowanie.H_pc_N(x, y, k_t),
            'Hbc': self.Hbc_calulation(x, y, pc, alfa, L_ind),
            'P': self.P_vector(x, y, pc, alfa, t_ot),
            'C' : self.C_calulation(x, y, data)
        }
        return res_dict

    def C_calulation(self, x, y, data):
        self.calkowanie.init_dNdx_dNdy(x, y)
        det_j = np.linalg.det(self.calkowanie.j_matrixes) # type: ignore
        point, weights = self.calkowanie.nodes_weight_combination()
        N_range = [ self.calkowanie.N_range(pc) for pc in point]
        N_dot = [ np.dot( N_range[i].reshape(4,1), N_range[i].reshape(1,4) ) for i in range( len( N_range))]
        prom_res = [ N * d * data['Density'] * data['SpecificHeat'] for N, d in zip(N_dot, det_j)]
        return sum([ weights[ind]*n for ind, n in enumerate(prom_res)])
        
    def weight_combination(self, n):
        nodes_x, weights_x = np.polynomial.legendre.leggauss(n)
        nodes_y, weights_y = np.polynomial.legendre.leggauss(n)
        weights_combinations = np.array(np.meshgrid(weights_x, weights_y)).T.reshape(-1, 2)
        return weights_combinations
            
    def PC_Weight(self):
        node, weights = self.calkowanie.nodes_weight_combination()
        weights= self.weight_combination(self.pc_number)
        nodes_weight = [ np.append(node[ind], w) for ind, w in enumerate(weights)]
        subarrays = np.array_split(nodes_weight, len(nodes_weight) // self.pc_number)
        PC = []
        for i in range(4):
            if i ==0:
                PC.append([[1.0, s[1], s[-1]] for s in subarrays[-1]])
            elif i == 1:
                PC.append([[s[0][0], 1.0, s[0][-2]] for s in [[s[-1] for s in subarrays]]] )
            elif i == 2:
                PC.append( [[-1.0, s[1], s[-1]] for  s in subarrays[0]] )
            elif i == 3:
                PC.append([[s[0], -1.0, s[-2]] for s in [s[0] for s in subarrays]] )
        return PC
    
    def Hbc_calulation(self, x, y, PC, alfa, L_ind):
        lengths =  self.L(x, y)
        Hbc = []
        for i in range(len(PC)):
            Pcs = PC[i]
            N_range = [ self.calkowanie.N_range([pc[0], pc[1]]) for pc in Pcs]
            Hbc.append( self.test_Hbc( N_range, alfa, lengths[L_ind[i]]/2, [p[-1] for p in Pcs] ) )
        return sum(Hbc)
    
    def test_Hbc(self, n_range, alfa, length, weights ):
        dot_nranges = [ weights[i] * np.dot( n_range[i].reshape(4,1), n_range[i].reshape(1,4) ) for i in range( len( n_range))]
        return  alfa* sum(dot_nranges) * length

    def P_vector(self, x, y, pc, alfa, t_ot):
        lengths = self.L(x, y)
        p_vector = []
        for i in range(len(pc)):
            n_range = [self.calkowanie.N_range([p[0], p[1]])  for p in pc[i]]
            p_vector.append(self.P(alfa, n_range, t_ot, lengths[-i]/2, [p[-1] for p in pc[i]]))
        return sum(p_vector)

    def P(self, alfa, n_ranges, t_ot, detJ, weights):
        print()
        return sum([weights[ind] * n.reshape(4,1)*t_ot  for ind, n in enumerate(n_ranges)])*detJ*alfa
