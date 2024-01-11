import math
import numpy as np
from numpy.polynomial.legendre import leggauss

class Calkowanie:
    def __init__(self, pc_number):
        self.pc_number = pc_number
        self.default_x_y = {'x': [0, 0.025, 0.025, 0], 'y':[0,0,0.025, 0.025]}
        self.dNdKsi = []
        self.dNdEta = []
        self.det_j = 6400
    
    def ksi(self, ksi):
        ksis = []
        a = 1
        b = 1
        for i in range(4):
            a = -1 if i in [0,3] else 1
            b = -1 if i in [0,1] else 1
            ksis.append( a*0.25 * (1 + (b*ksi)) )
        return ksis

    def eta(self, eta):
        etas = []
        a = 1
        b = 1
        for i in range(4):
            a = -1 if i in [0,1] else 1
            b = -1 if i in [0,3] else 1
            etas.append( a*0.25 * (1 + (b*eta)) )
        return etas
    
    def integration_point_weight(self):
        return leggauss(self.pc_number)
    
    def compute_dN_dxi_deta(self, xi, eta):
        dN_dxi = np.array([-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)])
        dN_deta = np.array([-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)])
        return dN_dxi, dN_deta
    
    def computedXi_dEta(self):
        self.dNdKsi = []
        self.dNdEta = []
        gauss_nodes,  gauss_weights = self.integration_point_weight()
        
        for xi, weight_xi in zip(gauss_nodes, gauss_weights):
            for eta, weight_eta in zip(gauss_nodes, gauss_weights):
                dN_dxi, dN_deta = self.compute_dN_dxi_deta(xi, eta)
                self.dNdKsi.append(dN_dxi)
                self.dNdEta.append(dN_deta)
        self.dNdKsi[1], self.dNdKsi[-2] = self.dNdKsi[-2], self.dNdKsi[1]
        self.dNdEta[1], self.dNdEta[2] = self.dNdEta[2], self.dNdEta[1]
        
    def N_range(self, pc):
        """
        pc => [ksi, eta]
        n_range => [N1, N2, N3, N4]
        """
        n_range = [
            0.25*(1-pc[0])*(1-pc[1]),
            0.25*(1+pc[0])*(1-pc[1]),
            0.25*(1+pc[0])*(1+pc[1]),
            0.25*(1-pc[0])*(1+pc[1]),
        ]
        return np.array(n_range)
        
    def Hbc(self, N_ranges, alfa, det):
        """
        N_range => [[N1,N2,N3,N4],.... ] dla punktów calkowania
        """
        matrix_sum =sum([np.dot(mat.reshape(4,1), mat.reshape(1,4))  for mat in N_ranges ])
        return alfa * matrix_sum * det
    
    def P_vec(self, N_ranges, t_ot, alfa, det):
        """
        N_ranges => [[N1,N2,N3,N4],.... ] dla punktów calkowania
        t_ot => temperatura otoczenia
        """
        matrix_sum = sum( [mat.reshape(4,1) * t_ot for mat in N_ranges] )
        return alfa * matrix_sum * det

    def nodes_weight_combination(self):
        nodes_x, weights_x = self.integration_point_weight()
        nodes_y, weights_y = self.integration_point_weight()
        nodes_combinations = np.array(np.meshgrid(nodes_x, nodes_y)).T.reshape(-1, 2)
        weights_combinations = np.outer(weights_x, weights_y).reshape(-1)
        return nodes_combinations, weights_combinations
    
    def jacobian_matrix(self, x,y, pc):
        self.computedXi_dEta()
        dy_dKsi = sum([ self.dNdKsi[pc][i]* y[i] for i in range(len(y))])
        dx_dKsi = sum([ self.dNdKsi[pc][i]* x[i] for i in range(len(y))])
        dy_dEta = sum([ self.dNdEta[pc][i]* y[i] for i in range(len(y))])
        dx_dEta = sum([ self.dNdEta[pc][i]* x[i] for i in range(len(y))])
        return [[dy_dEta, -1 * dy_dKsi ], [-1*dx_dEta, dx_dKsi]]
    
    def init_dNdx_dNdy(self, x, y):
        self.dNdx = []
        self.dNdy = []
        j_matrix =  [ self.jacobian_matrix(x,y, i) for i in range(self.pc_number**2)]
        self.j_matrixes = j_matrix
        for j_ind, matJ in enumerate(j_matrix):
            x = []
            y = []
            det_j = np.linalg.det( np.linalg.inv(matJ)) # type: ignore
            for ksi, eta in zip( np.array(self.dNdKsi[j_ind]), np.array(self.dNdEta[j_ind]) ):
                ksi_eta = np.array([ksi, eta]).reshape(2,1)
                prom_res = det_j * np.dot(matJ, ksi_eta) # type: ignore
                x.append(prom_res[0][0])
                y.append(prom_res[1][0])
            self.dNdx.append(x)
            self.dNdy.append(y)
        
    def mat_dN_dx(self, x, y):
        self.computedXi_dEta()
        self.init_dNdx_dNdy(x, y)
        return self.dNdx
    
    def mat_dN_dy(self, x, y):
        self.computedXi_dEta()
        self.init_dNdx_dNdy(x, y)
        return self.dNdy
    
    def H_pc_N(self, x, y, k_t):
        dN_dx =  np.array( self.mat_dN_dx(x, y))
        dN_dy = np.array( self.mat_dN_dy(x,y))
        jakobians = self.j_matrixes
        H_by_pc = []
        nodes_point, nodes_weight = self.nodes_weight_combination()
        for ind in range(len(nodes_point)):
            dot_dNdnx = np.dot( dN_dx[ind].reshape(4,1), dN_dx[ind].reshape(1,4) ) 
            dot_dNdy = np.dot( dN_dy[ind].reshape(4,1), dN_dy[ind].reshape(1,4) )
            H_by_pc.append( k_t * (dot_dNdnx + dot_dNdy) * nodes_weight[ind] * np.linalg.det(jakobians[ind]) ) # type: ignore
        return sum(H_by_pc)

    def print_matrix(matrix):
        for i in range(len(matrix)): # type: ignore
            print(", ".join( map(str, matrix[i]))) # type: ignore