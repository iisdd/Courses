class Queue:
    def __init__(self, size=100):
        self.queue = [0 for _ in range(size)]
        self.size = size
        self.front = 0                              # 队头指针
        self.rear = 0                               # 队尾指针

    def push(self, element):
        if not self.is_filled():
            self.rear = (self.rear+1) % self.size
            self.queue[self.rear] = element
        else:
            raise IndexError

    def pop(self):
        if not self.is_empty():
            self.front = (self.front+1) % self.size
            return self.queue[self.front]
        else:
            raise IndexError

    def is_empty(self):
        return self.rear == self.front

    def is_filled(self):
        return (self.rear+1) % self.size == self.front

queue = Queue(10)
print(queue.is_empty())
for i in range(9):
    queue.push(i)
print(queue.queue)
print(queue.is_filled())            # 长度为10的队列可以装9个元素
for i in range(6):
    print(queue.pop())
print('front指针位置: ', queue.front)










