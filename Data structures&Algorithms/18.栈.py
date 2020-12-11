# 定义一个类,实现栈的push,pop,get_top功能
class Stack:                            # 类名一般大写
    def __init__(self):
        self.stack = []

    def push(self, element):
        self.stack.append(element)

    def pop(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        else:
            return None

    def get_top(self):
        if len(self.stack) > 0:
            return self.stack[-1]
        else:
            return None

stack = Stack()
stack.push('1')
stack.push('2')
print(stack.pop())
print(stack.get_top())

