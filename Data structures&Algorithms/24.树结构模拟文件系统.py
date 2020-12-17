# 用树结构模拟文件系统,每个文件就是一个节点


class Node:
    def __init__(self, name, type = 'dir'):
        self.name = name
        self.type = type                                        # 'dir' or 'file'
        self.children = []                                      # 孩子节点放在一个列表里, Ex: n1.children.append(n2)
        self.parent = None                                      # 父亲节点只能有一个, Ex: n2.parent = n1
    def __str__(self):
        return self.name


class FileSystemTree:
    def __init__(self):
        self.root = Node('/')                                   # 根目录
        self.now = self.root                                    # 当前目录

    def mkdir(self, name):                                      # 创建路径,名字必须合法
        if name[-1] != '/':
            name += '/'
        node = Node(name)
        # 子目录与上级目录建立联系
        self.now.children.append(node)
        node.parent = self.now

    def ls(self):
        return self.now.children

    def cd(self, name):
        if name[-1] != '/':
            name += '/'
        if name == '../':
            self.now = self.now.parent
        for child in self.now.children:
            if name == child.name:
                self.now = child
        else:
            raise ValueError('invalid dir')

