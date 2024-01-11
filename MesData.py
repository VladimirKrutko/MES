class MesData:

    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.data =  self.parse_data()
        self.data['leaf_member'] = self.elements()

    
    def parse_data(self):
        data = self.get_metadata()
        data['Node'] = self.get_matrix('*Node', '*Element, type=DC2D4')
        data['Element'] = self.get_matrix_element('*Element, type=DC2D4', '*BC')
        data['BC'] = self.get_bc()
        return data
    
    def file_data(self):
        with open(self.file_name, 'r') as f:
            file_text = f.read()
        return file_text
    
    def splited_text(self):
        return self.file_data().split('\n')

    def get_metadata(self):
        data = {}
        for row in self.splited_text()[0:self.splited_text().index('*Node')]:
            split_row = row.split(' ')
            data[' '.join( split_row[0:-1], )] = float(split_row[-1])
        return data
    
    def get_matrix(self, start, stop):
        nodes = []
        for number_row in self.splited_text()[self.splited_text().index(start)+1:self.splited_text().index(stop)]:
            numbers = number_row.split(',')
            nodes.append( list( map(float, numbers) )[1:3])
        return nodes
    
    def get_matrix_element(self, start, stop):
        nodes = []
        for number_row in self.splited_text()[self.splited_text().index(start)+1:self.splited_text().index(stop)]:
            numbers = number_row.split(',')
            nodes.append( list( map(int, numbers) )[1:])
        return nodes

    def get_bc(self):
        last_numbers = self.splited_text()[-1]
        return list(map(float, last_numbers.split(',')))
    
    def elements(self):
        elements = []
        for el in self.data['Element']:
            elements.append( { 'x': [ self.data['Node'][i-1][0] for i in el ], 
                               'y': [ self.data['Node'][i-1][1] for i in el ] } )
        return elements

            
    

if __name__ == '__main__':
    FILES = { 'Test1_4_4.txt', 'Test3_31_31_kwadrat.txt'}
    DATA = {}
    for file_name in FILES:
        DATA[file_name] = MesData(file_name)
        