import math
import numpy as np
from copy import deepcopy
from operator import le

from sympy import true

class Calkowanie:
    def __init__(self, pc_number):
        self.pc_number = pc_number
        self.dNdKsi4_4, self.dNdEta4_4 = self.matrixKsiEta()
        # self.dNdEta4_4 = []
        self.dNdKsi9_4 = [[0.0 for _ in range(4)] for _ in range(9)]
        self.dNdEta9_4 = [[0.0 for _ in range(4)] for _ in range(9)]
        self.default_x_y = {'x': [0, 0.025, 0.025, 0], 'y':[0,0,0.025, 0.025]}
        self.det_j = 6400
        
    def _split_by_columns(self):
        start = 0
        step = self.pc_number
        wall_nodes = []
        i_plus = 1 if self.pc_number == 2 else 2
        for i in range(1, len(self.nodes_combinations[0])+i_plus):
            wall_nodes.append(self.nodes_point()[start:step*i])
            start += step
        return wall_nodes
    
    def PC_weight(self, wall = true):
        wall_nodes = self._split_by_columns()
        walls = [[] for _ in range(4) ]
        for i in range(4):
            if i == 0:
                prom = deepcopy(wall_nodes[-1])
                if wall:
                    prom[:, 0] = 1.0
                walls[0] = prom
            elif i == 1:
                prom = deepcopy(wall_nodes)
                if wall:
                    for i in range(len(prom)):
                        prom[i][-1][1] = 1.0
                walls[1] = [prom[i][-1] for i in range(len(wall_nodes))]
            elif i == 2:
                prom = deepcopy(wall_nodes[1])
                if wall:
                    prom[:, 0] = -1.0
                walls[2] = prom
            elif i == 3:
                prom = deepcopy(wall_nodes)
                if wall:
                    for i in range(len(prom)):
                        prom[i][0][1] = -1.0
                walls[3] = [prom[i][0] for i in range(len(wall_nodes))]
        return walls
            
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
    
    def nodes_point(self):
        self.nodes_x, self.weights_x = np.polynomial.legendre.leggauss(self.pc_number)
        nodes_y, weights_y = np.polynomial.legendre.leggauss(self.pc_number)
        self.nodes_combinations = np.array(np.meshgrid(self.nodes_x, nodes_y)).T.reshape(-1, 2)
        self.weights_combinations = np.outer(self.weights_x, weights_y).reshape(-1)

        nodes_weight = np.zeros((len(self.nodes_combinations),3))
        for i in range(len(self.nodes_combinations)):
            nodes_weight[i][0], nodes_weight[i][1] = self.nodes_combinations[i][0], self.nodes_combinations[i][1]
            nodes_weight[i][2] = self.weights_combinations[i]
            
        return nodes_weight
    
    def PC(self):
        pc = [
            [[-1/math.sqrt(3), -1], [1/math.sqrt(3), -1]],
            [[1, -1/math.sqrt(3)], [1, 1/math.sqrt(3)]],
            [[1/math.sqrt(3), 1], [-1/math.sqrt(3), 1]],   
            [[-1, 1/math.sqrt(3)], [-1, -1/math.sqrt(3)]],
            ]
        # pc = [ [-1/math.sqrt(3), -1],  #11
        #        [1/math.sqrt(3), -1],   #12
        #        [1, -1/math.sqrt(3)],   #21
        #        [1, 1/math.sqrt(3)],    #22
        #        [1/math.sqrt(3), 1],    #31
        #        [-1/math.sqrt(3), 1],   #32
        #        [-1, 1/math.sqrt(3)],     #41
        #        [-1, -1/math.sqrt(3)]   #42 
        # ]
        return pc

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
    
    def N_range_eta(self, ):
        N_range = []
        for node in self.nodes_point():
            eta = node[0]
            N_range.append( [ -0.25 * (1 - eta ),
                              -0.25 * (1 + eta ),
                               0.25 * (1 + eta ),
                               0.25 * (1 - eta )] )
            
        return N_range
    
    def jacobian_matrix(self, xi, eta, x,y):
        # N1 = 0.25 * (1 - xi) * (1 - eta)
        # N2 = 0.25 * (1 + xi) * (1 - eta)
        # N3 = 0.25 * (1 + xi) * (1 + eta)
        # N4 = 0.25 * (1 - xi) * (1 + eta)

        dN1_dxi = -0.25 * (1 - eta)
        dN2_dxi = 0.25 * (1 - eta)
        dN3_dxi = 0.25 * (1 + eta)
        dN4_dxi = -0.25 * (1 + eta)

        dN1_deta = -0.25 * (1 - xi)
        dN2_deta = -0.25 * (1 + xi)
        dN3_deta = 0.25 * (1 + xi)
        dN4_deta = 0.25 * (1 - xi)

        dx_dxi = x[0]*dN1_dxi + x[1]*dN2_dxi + x[2]*dN3_dxi + x[3]*dN4_dxi
        dx_deta = x[0]*dN1_deta + x[1]*dN2_deta + x[2]*dN3_deta + x[3]*dN4_deta
        dy_dxi = y[0]*dN1_dxi + y[1]*dN2_dxi + y[2]*dN3_dxi + y[3]*dN4_dxi
        dy_deta = y[0]*dN1_deta + y[1]*dN2_deta + y[2]*dN3_deta + y[3]*dN4_deta
        J = np.array([
            [dx_dxi, dx_deta],
            [dy_dxi, dy_deta]
        ])
        return J
    
    def J_matrixies(self, x,y):
        J_array = []
        for xi, weight_xi in zip(self.nodes_x, self.weights_x):
            for eta, weight_eta in zip(self.nodes_x, self.weights_x):
                J_array.append( self.jacobian_matrix(xi, eta, x, y) )
        return J_array
    
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

    def compute_dN_dxi_deta(self, xi, eta):
        dN_dxi = np.array([-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)])
        dN_deta = np.array([-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)])
        return dN_dxi, dN_deta
    
    def matrixKsiEta(self):
        self.nodes_point()
        ksi_matrix = []
        eta_matrix = []
        for xi, weight_xi in zip(self.nodes_x, self.weights_x):
            for eta, weight_eta in zip(self.nodes_x, self.weights_x):
                dN_dxi, dN_deta = self.compute_dN_dxi_deta(xi, eta)
                eta_matrix.append(dN_deta)
                ksi_matrix.append(dN_dxi)
        # self.dNdKsi4_4[1], self.dNdKsi4_4[-1] = self.dNdKsi4_4[-1], self.dNdKsi4_4[1]
        # self.dNdKsi4_4[1], self.dNdKsi4_4[-2] = self.dNdKsi4_4[1], self.dNdKsi4_4[-2]
        eta_matrix[1], eta_matrix[2] = eta_matrix[2], eta_matrix[1]
        ksi_matrix[1], ksi_matrix[2] = ksi_matrix[2], ksi_matrix[1]
        return ksi_matrix, eta_matrix
    
    def nodes_point9_4(self):
        points = [
                [-math.sqrt(3.0 / 5.0), -math.sqrt(3.0 / 5.0)],
                [0, -math.sqrt(3.0 / 5.0)],
                [math.sqrt(3.0 / 5.0), -math.sqrt(3.0 / 5.0)],
                [-math.sqrt(3.0 / 5.0), 0],
                [0, 0],
                [math.sqrt(3.0 / 5.0), 0],
                [-math.sqrt(3.0 / 5.0), math.sqrt(3.0 / 5.0)],
                [0, math.sqrt(3.0 / 5.0)],
                [math.sqrt(3.0 / 5.0), math.sqrt(3.0 / 5.0)]
                ]
        return points

    def matrix9_4(self):
        for i in range(9):
            for j in range(4):
                # pdb.set_trace()
                ksi = self.ksi(self.nodes_point9_4()[i][0])
                eta = self.eta(self.nodes_point9_4()[i][1])
                self.dNdEta9_4[i][j] = eta[j]
                self.dNdKsi9_4[i][j] = ksi[j]

    def matrix_dx_dksi_dyd_ksi(self, x,y, pc):
        dy_dKsi = sum([ self.dNdKsi4_4[pc][i]* y[i] for i in range(len(y))])
        dx_dKsi = sum([ self.dNdKsi4_4[pc][i]* x[i] for i in range(len(y))])
        dy_dEta = sum([ self.dNdEta4_4[pc][i]* y[i] for i in range(len(y))])
        dx_dEta = sum([ self.dNdEta4_4[pc][i]* x[i] for i in range(len(y))])
        return [[dy_dEta, -1 * dy_dKsi ], [-1*dx_dEta, dx_dKsi]]
    
    def init_dNdx_dNdy(self, x, y):
        self.dNdKsi4_4[1], self.dNdKsi4_4[-2] = self.dNdKsi4_4[1], self.dNdKsi4_4[-2]
        self.dNdx = []
        self.dNdy = []
        j_matrix =  self.J_matrixies(x, y)
        for j_ind, matJ in enumerate(j_matrix):
            x = []
            y = []
            det_j = 1/np.linalg.det(matJ) # type: ignore
            for ksi, eta in zip( np.array(self.dNdKsi4_4[j_ind]), np.array(self.dNdEta4_4[j_ind]) ):
                ksi_eta = np.array([ksi, eta]).reshape(2,1)
                prom_res = det_j * np.dot(matJ, ksi_eta) # type: ignore
                x.append(prom_res[0][0])
                y.append(prom_res[1][0])
            self.dNdx.append(x)
            self.dNdy.append(y)

    def mat_dN_dx(self, x, y, det):
        # self.matrix4_4()
        self.init_dNdx_dNdy(x, y)
        return self.dNdx
    
    def mat_dN_dy(self, x, y, det):
        # self.matrix4_4()
        self.init_dNdx_dNdy(x, y)
        return self.dNdy
    
    def H_pc_N(self, x, y, det, N, k_t, dV):
        dN_dx =  np.array( self.mat_dN_dx(x, y, det))
        dN_dy = np.array( self.mat_dN_dy(x,y, det))
        H = np.dot( dN_dx[N].reshape(4,1), dN_dx[N].reshape(1,4) ) + np.dot( dN_dy[N].reshape(4,1), dN_dy[N].reshape(1,4) )
        return k_t * H * dV

    def detJ(matrix):
        np_matrix = np.array(matrix)
        return 1/np.linalg.det(np_matrix)
    
    def invMatrix(self, matrix):
        return np.linalg.inv(matrix)

    def print_matrix(matrix):
        for i in range(len(matrix)): # type: ignore
            print(", ".join( map(str, matrix[i]))) # type: ignore
            