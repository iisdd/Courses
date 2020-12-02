class FileObject:
    def __init__(self , file):
        self.f = open(file)

    def xs(self):
        print(self.f.read())
    def __del__(self):
        self.f.close()
