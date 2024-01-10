from second.Calculation import Calculation

class Node:
    def __init__(self, x,y, pc, data, L_ind, pc_number):
        self.x = x
        self.y = y
        self.pc = pc
        self.L_ind = L_ind
        self.data = data
        self.surface = Calculation(pc_number).surfaces(self.x, self.y, self.pc, self.data['Alfa'], 
                                              self.data['Tot'], self.data, self.L_ind, self.data['Conductivity'])

    def __str__(self):
        return f"x => {' '.join(self.x)} \n y=> { ' '.join(self.y)}"
    
    def print_surface(self):
        # print(self.__str__())
        for key in self.surface.keys():
            print(f"***{key}***")
            print(self.surface[key])
