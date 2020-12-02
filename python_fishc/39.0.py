class A:
    count = 0
    def __init__(self):
        A.count += 1
    def __del__(self):
        A.count -= 1

        
