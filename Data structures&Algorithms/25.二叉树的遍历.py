# 定义二叉树中的节点
class BiTreeNode:
    def __init__(self, data):
        self.data = data
        self.lchild = None                              # 左孩子节点
        self.rchild = None                              # 右孩子节点

a = BiTreeNode('A')
b = BiTreeNode('B')
c = BiTreeNode('C')
d = BiTreeNode('D')
e = BiTreeNode('E')
f = BiTreeNode('F')
g = BiTreeNode('G')

e.lchild = a
e.rchild = g
a.rchild = c
g.rchild = f
c.lchild = b
c.rchild = d

root = e
print(root.lchild.rchild.data)                          # C

def pre_order(root):                                    # 前序遍历树的节点,使用递归实现
    if root:
        print(root.data, end=',')
        pre_order(root.lchild)
        pre_order(root.rchild)
print('前序遍历: ', end='')
pre_order(root)


def in_order(root):
    if root:
        in_order(root.lchild)
        print(root.data, end=',')
        in_order(root.rchild)
print('\n中序遍历: ', end='')
in_order(root)


def post_order(root):
    if root:
        post_order(root.lchild)
        post_order(root.rchild)
        print(root.data, end=',')
print('\n后序遍历: ', end='')
post_order(root)


from collections import deque
def level_order(root):
    queue = deque([root])
    while len(queue) > 0:
        Node = queue.popleft()
        print(Node.data, end=',')
        if Node.lchild:
            queue.append(Node.lchild)
        if Node.rchild:
            queue.append(Node.rchild)
print('\n层次遍历: ', end='')
level_order(root)















