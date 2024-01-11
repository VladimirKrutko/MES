from shutil import which


def add_newline_and_join(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, list) and all(isinstance(item, str) for item in result):
            result.append('\n')
            joined_result = ''.join(result)
            return joined_result
        else:
            return result
    return wrapper

class ParaviewFile:     
    
    def __init__(self, data, temperature) -> None:
        self.data = data
        self.TempinTime = temperature
            
    @add_newline_and_join
    def file_header(self):
        return ['# vtk DataFile Version 2.0\n',
                'Unstructured Grid Example\n',
                'ASCII\n',
                'DATASET UNSTRUCTURED_GRID\n']

    @add_newline_and_join
    def points(self):
        points = [f"POINTS { int(self.data['Nodes number'])} float\n"]
        for x_y in self.data['Node']:
            if x_y[0] == 0.0:
                points.append(f"{int(x_y[0])} {x_y[1]} 0\n" )
            elif x_y[1] == 0.0:
                points.append(f"{x_y[0]} {int(x_y[1])} 0\n" )
            else:
                points.append(f"{x_y[0]} {x_y[1]} 0\n" )
        return points    

    @add_newline_and_join
    def cells(self):
        # el_size = sum([i for i in range(int(self.data['Elements number'])+1)])
        nodes_number = [f"CELLS {int(self.data['Elements number'])} { int(self.data['Elements number']) * 5 }\n"]
        for element in self.data['Element']:
            nodes_number.append( '4 '+' '.join( [str(i-1) for i in element] )+'\n' )
        return nodes_number

    @add_newline_and_join
    def cell_types(self):
        el_number = int(self.data['Elements number'])
        cell_types = [f"CELL_TYPES {el_number}\n"]
        [ cell_types.append(f'9\n') for el in range(el_number) ]
        return cell_types


    def clarifying_data(self):
        return "".join([f"POINT_DATA {int(self.data['Nodes number'])}\n",
                "SCALARS Temp float 1\n",
                "LOOKUP_TABLE default\n"])
        
    def temperature(self, t_vector):
        return "".join([f"{str(t)}\n" for t in t_vector])
    
    def to_file(self, path):
        for ind, t_vector in enumerate(self.TempinTime):
            file_data = "".join([self.file_header(), self.points(), self.cells(), 
                                 self.cell_types(), self.clarifying_data(), 
                                 self.temperature(t_vector)]) # type: ignore
            with open(f"{path}Foo{ind+1}.vtk", 'w') as f:
                f.write(file_data)