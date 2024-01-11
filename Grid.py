import numpy as np
from Node import Node
from Calkowanie import Calkowanie
from Calculation import Calculation


class Grid:           
    def __init__(self, data, pc_number):
        self.data = data
        self.calculation = Calculation(pc_number)
        self.pc_number = pc_number
        self.elemnt_xy = self.data["leaf_member"]
        self.size = [int(self.data['Elements number']**0.5), int(self.data['Elements number']**0.5)]
        self.calkowanie = Calkowanie(pc_number)
        self.grid = self.create_grig()
        self.H_glob = self.create_H_glob()
        self.P_glob = self.create_P_glob()
        self.C_glob = self.create_C_glob()
        self.TempinTime = self.temperatures_in_time()

        
    def weight_combination(self, n):
        nodes_x, weights_x = np.polynomial.legendre.leggauss(n)
        nodes_y, weights_y = np.polynomial.legendre.leggauss(n)
        weights_combinations = np.array(np.meshgrid(weights_x, weights_y)).T.reshape(-1, 2)
        return weights_combinations
            
    # def PC_Weight(self):
    #     node, weights = self.calkowanie.nodes_weight_combination()
    #     weights= self.weight_combination(self.pc_number)
    #     nodes_weight = [ np.append(node[ind], w) for ind, w in enumerate(weights)]
    #     subarrays = np.array_split(nodes_weight, len(nodes_weight) // self.pc_number)
    #     PC = []
    #     for i in range(4):
    #         if i ==0:
    #             PC.append([[1.0, s[1], s[-1]] for s in subarrays[-1]])
    #         elif i == 1:
    #             PC.append([[s[0][0], 1.0, s[0][-2]] for s in [[s[-1] for s in subarrays]]] )
    #         elif i == 2:
    #             PC.append( [[-1.0, s[1], s[-1]] for  s in subarrays[0]] )
    #         elif i == 3:
    #             PC.append([[s[0], -1.0, s[-2]] for s in [s[0] for s in subarrays]] )
    #     return PC
    
    def create_grig(self):
        PC = self.calculation.PC_Weight()
        grid = []
        last = 0
        for i in range(self.size[1]):
            grid_row = []
            for j in range(self.size[0]):
                # pc_x_y = {'x':[], 'y':[]}
                L_ind =[]
                if j == 0 and i == 0:
                    pc = [PC[-2], PC[-1]]
                    L_ind = [0, -1]
                                        
                elif i == 0 and j == self.size[0]-1:
                    pc = [PC[0], PC[-1]]
                    L_ind = [0, 1]
        
                elif i == self.size[1]-1 and j == 0:
                    pc = [PC[1], PC[-2]]
                    L_ind = [2, 3]
                    
                elif i == self.size[1]-1 and j == self.size[0]-1:
                    pc = [PC[0], PC[1]]
                    L_ind = [1,2]
    
                elif i == 0:
                    pc = [PC[-1]]
                    L_ind = [0]
                
                elif j == 0:
                    pc = [PC[-2]]
                    L_ind = [3]
                    
                elif j == self.size[0]-1:
                    pc = [PC[0]]
                    L_ind = [1]
                    
                elif i == self.size[1]-1:
                    pc = [PC[1]]
                    L_ind = [2]
    
                else:
                    pc = []

                x_y = self.elemnt_xy[last]
                last+=1
                grid_row.append( Node(x=x_y['x'], y=x_y['y'], pc=pc, data=self.data, L_ind=L_ind, pc_number=self.pc_number))
            grid.append(grid_row)
        return grid
    
    def create_H_glob(self):
        H_locs = H_locs = [self.grid[i][j].surface['H'] +self.grid[i][j].surface['Hbc'] for i in range(self.size[0]) for j in range(self.size[0])]
        H_glob = np.zeros((int(self.data['Nodes number']), int(self.data['Nodes number'])))
        for el_ind, g_index in enumerate(self.data['Element']):
            for i in range(4):
                for j in range(4):
                    H_glob[g_index[i]-1][g_index[j]-1] +=  H_locs[el_ind][i][j]
        return H_glob
    
    def create_P_glob(self):
        P_locs = [self.grid[i][j].surface['P'] for i in range(self.size[0]) for j in range(self.size[0])]
        P_glob = [0 for _ in range(int(self.data['Nodes number']))]
        for el_ind, g_index in enumerate(self.data['Element']):
            for ind, g_i in enumerate(g_index):
                try:
                    P_glob[g_i-1] += P_locs[el_ind][ind][0]
                except:
                    P_glob[g_i-1] += 0
        return P_glob
    
    def create_C_glob(self):
        C_locs = [self.grid[i][j].surface['C'] for i in range(self.size[0]) for j in range(self.size[0])]
        C_glob = np.zeros((int(self.data['Nodes number']), int(self.data['Nodes number'])))
        for el_ind, g_index in enumerate(self.data['Element']):
            for i in range(4):
                for j in range(4):
                    C_glob[g_index[i]-1][g_index[j]-1] +=  C_locs[el_ind][i][j]
        return C_glob
    
    def temperatures_in_time(self):
        H_ = self.H_glob + self.C_glob/self.data['SimulationStepTime']
        t0 = np.array([self.data['InitialTemp'] for _ in range( int(self.data['Nodes number']))])
        t1 = [t0]
        t_in_time = []
        for _ in range(int(self.data['SimulationTime']/self.data['SimulationStepTime'])):
            P_ =  np.dot( self.C_glob/self.data['SimulationStepTime'], t0) + self.P_glob
            t1 = np.linalg.solve(H_, P_)
            t_in_time.append(t1)
            t0 = t1
        return t_in_time
