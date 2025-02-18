class A1:
    def __init__(self):
        input_shape = 10
        self.hh = 1

    def get_hh(self):
        return self.hh
    
class A2(A1):
    def __init__(self):
        super(A2, self).__init__()
        a = self.input_shape


a2 = A2()
print(a2.a)
