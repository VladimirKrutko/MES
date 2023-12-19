import math
import numpy as np

class Calkowanie:
    def __init__(self, pc_number):
        self.wagi = { i: sorted(self.get_weight()[i], key=lambda x: x['point']) for i in range(1,6) }
        self.dNdKsi4_4 = [[0.0 for _ in range(4)] for _ in range(4)]
        self.dNdEta4_4 = [[0.0 for _ in range(4)] for _ in range(4)]
        self.dNdKsi9_4 = [[0.0 for _ in range(4)] for _ in range(9)]
        self.dNdEta9_4 = [[0.0 for _ in range(4)] for _ in range(9)]
        self.default_x_y = {'x': [0, 0.025, 0.025, 0], 'y':[0,0,0.025, 0.025]}
        self.pc_number = pc_number
        self.det_j = 6400

    def get_weight(self):
        return {1: [ {'point': 0, 'weight': 2} ], 
         2: [ {'point': num*((1/3.0)**(1/2)), 'weight':1 } for num in [-1,1]],
         3: [ {'point': 0, 'weight': 8/9.0 },
              {'point': -1*((3/5.0)**(1/2)), 'weight': 5/9.0 },
              {'point': ((3/5.0)**(1/2)), 'weight': 5/9.0 }
             ],
        4: [[ {'point': num*( (3/7.0 - 2/7.0*(6/5.0)**(1/2) )**(1/2) ) , 'weight': (18+30**1/2)/36} for num in [-1,1]] 
        + [ {'point': num*( (3/7.0 + 2/7.0*(6/5.0)**(1/2) )**(1/2) ) , 'weight': (18-30**1/2)/36} for num in [-1,1] ]][0],
        5: [{'point': 0, 'weight': 128/225}] 
        + [[ {'point': num*1/3*( (5-2*(10/7)**(1/2)))**(1/2), 'weight': (322+13*(70**(1/2)))/900}  for num in [-1,1] ] 
            + [ {'point': num*1/3*( (5+2*(10/7)**(1/2)))**(1/2), 'weight': (322-13*(70**(1/2)))/900}  for num in [-1,1] ]][0]}
    
    def gauss_1d_integration(self, f, points):
        integral = sum( node['weight']  * f(node['point']) for node in self.wagi[points])
        return integral

    def gauss_2d_integration(self, f, points):
        nodes_x = self.wagi[points]
        nodes_y = self.wagi[points]
        integral = sum(node_x['weight'] * node_y['weight'] * f(node_x['point'], node_y['point']) for node_x in nodes_x for node_y in nodes_y)
        return integral

    def f(self, x):
        return (1/4)*(1-x)

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
        nodes_x, weights_x = np.polynomial.legendre.leggauss(self.pc_number)
        nodes_y, weights_y = np.polynomial.legendre.leggauss(self.pc_number)
        nodes_combinations = np.array(np.meshgrid(nodes_x, nodes_y)).T.reshape(-1, 2)
        weights_combinations = np.outer(weights_x, weights_y).reshape(-1)

        nodes_weight = np.zeros((len(nodes_combinations),3))
        for i in range(len(nodes_combinations)):
            nodes_weight[i][0], nodes_weight[i][1] = nodes_combinations[i][0], nodes_combinations[i][1]
            nodes_weight[i][2] = weights_combinations[i]
            
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

    def matrix4_4(self):
        for i in range(4):
            ksi = self.ksi(self.nodes_point()[i][0])
            eta = self.eta(self.nodes_point()[i][1])
            for j in range(4):
              self.dNdKsi4_4[i][j] = ksi[j]
              self.dNdEta4_4[i][j] = eta[j]  

        self.dNdKsi4_4[1], self.dNdKsi4_4[-1] = self.dNdKsi4_4[-1], self.dNdKsi4_4[1]
        self.dNdEta4_4[1], self.dNdEta4_4[2] = self.dNdEta4_4[2], self.dNdEta4_4[1]
    
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
        self.matrix4_4()
        dy_dKsi = sum([ self.dNdKsi4_4[pc][i]* y[i] for i in range(len(y))])
        dx_dKsi = sum([ self.dNdKsi4_4[pc][i]* x[i] for i in range(len(y))])
        dy_dEta = sum([ self.dNdEta4_4[pc][i]* y[i] for i in range(len(y))])
        dx_dEta = sum([ self.dNdEta4_4[pc][i]* x[i] for i in range(len(y))])
        return [[dy_dEta, -1 * dy_dKsi ], [-1*dx_dEta, dx_dKsi]]
    
    def init_dNdx_dNdy(self, x, y):
        self.dNdx = []
        self.dNdy = []
        j_matrix =  [ self.matrix_dx_dksi_dyd_ksi(x,y, i) for i in range(4)]
        for j_ind, matJ in enumerate(j_matrix):
            x = []
            y = []
            det_j = 1 / np.linalg.det(matJ) # type: ignore
            for ksi, eta in zip( np.array(self.dNdKsi4_4[j_ind]), np.array(self.dNdEta4_4[j_ind]) ):
                ksi_eta = np.array([ksi, eta]).reshape(2,1)
                prom_res = det_j * np.dot(matJ, ksi_eta) # type: ignore
                x.append(prom_res[0][0])
                y.append(prom_res[1][0])
            self.dNdx.append(x)
            self.dNdy.append(y)

    def mat_dN_dx(self, x, y, det):
        self.matrix4_4()
        self.init_dNdx_dNdy(x, y)
        return self.dNdx
    
    def mat_dN_dy(self, x, y, det):
        self.matrix4_4()
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