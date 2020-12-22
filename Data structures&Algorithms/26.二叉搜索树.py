


class BiTreeNode:
    def __init__(self, data):
        self.data = data
        self.lchild = None                              # 左孩子节点
        self.rchild = None                              # 右孩子节点
        self.parent = None


class BST:
    def __init__(self, li=None):
        self.root = None
        if li:
            for val in li:
                self.insert_no_rec(val)

############################################## 插入功能 ################################################
    def insert(self, node, val):
        if not node:                                    # 当前节点为None,就改变这个位置的值
            node = BiTreeNode(val)
        elif node.data > val:                           # 如果值改变了那就与左孩子建立联系,如果没改变就当说了句废话
            node.lchild = self.insert(node.lchild, val) # 如果node.lchild有值就接着比,没有就落户了
            node.lchild.parent = node
        elif node.data < val:                           # 不考虑插入相同元素的情况
            node.rchild = self.insert(node.rchild, val)
            node.rchild.parent = node
        return node

    def insert_no_rec(self, val):                       # 非递归形式的插入
        p = self.root
        if not p:                                       # 空树
            self.root = BiTreeNode(val)
            return
        while 1:
            if p.data > val:
                if p.lchild:                            # 存在左孩子
                    p = p.lchild
                else:                                   # 左边没有节点,捏一个节点
                    p.lchild = BiTreeNode(val)
                    p.lchild.parent = p
                    return
            elif p.data < val:
                if p.rchild:
                    p = p.rchild
                else:
                    p.rchild = BiTreeNode(val)
                    p.rchild.parent = p
                    return

############################################## 插入功能 ################################################
############################################## 查询功能 ################################################
    def query(self, node, val):                         # 查询功能,递归版本
        if not node:
            return None
        if node.data < val:
            return self.query(node.rchild, val)
        elif node.data > val:
            return self.query(node.lchild, val)
        else:
            return node

    def query_no_rec(self, val):
        p = self.root
        while p:
            if p.data > val:
                p = p.lchild
            elif p.data < val:
                p = p.rchild
            else:
                return p

############################################## 查询功能 ################################################

###################################### 遍历打印功能 #######################################
    def pre_order(self, root):                          # 前序遍历树的节点,使用递归实现
        if root:
            print(root.data, end=',')
            self.pre_order(root.lchild)
            self.pre_order(root.rchild)

    def in_order(self, root):
        if root:
            self.in_order(root.lchild)
            print(root.data, end=',')
            self.in_order(root.rchild)

    def post_order(self, root):
        if root:
            self.post_order(root.lchild)
            self.post_order(root.rchild)
            print(root.data, end=',')

###################################### 遍历打印功能 #######################################

###################################### 删除功能 #######################################
    def __remove_node_1(self, node):                    # 情况1: 删除的节点是叶子节点,两个下划线表示类内方法
        if not node.parent:                             # node是根节点
            self.root = None
        elif node == node.parent.lchild:                # node是它父节点的左孩子
            node.parent.lchild = None
        else:                                           # node是它父节点的右孩子
            node.parent.rchild = None

    def __remove_node_21(self, node):                   # 情况2.1: 删除的节点不是叶子节点,且其只有左孩子
        if not node.parent:                             # node是根节点
            self.root = node.lchild
            node.lchild.parent = None
        elif node == node.parent.lchild:                # node是其父节点的左孩子节点
            node.parent.lchild = node.lchild
            node.lchild.parent = node.parent
        else:                                           # node是其父节点的右孩子节点
            node.parent.rchild = node.rchild
            node.rchild.parent = node.parent

    def __remove_node_22(self, node):                   # 情况2.2: 删除的节点非叶子节点,且其只有右孩子
        if not node.parent:
            self.root = node.rchild
            node.rchild.parent = None
        elif node == node.parent.lchild:                # node是其父节点的左孩子节点
            node.parent.lchild = node.rchild
            node.rchild.parent = node.parent
        else:                                           # node是其父节点的右孩子节点
            node.parent.rchild = node.rchild
            node.rchild.parent = node.parent

    def delete(self, val):
        if self.root:                                   # 不是空树
            node = self.query_no_rec(val)
            if not node:
                return False                            # 没找到要删除的节点
            if not node.lchild and not node.rchild:     # 情况1:叶子节点
                self.__remove_node_1(node)
            elif not node.rchild:                       # 情况2.1:只有左孩子节点
                self.__remove_node_21(node)
            elif not node.lchild:                       # 情况2.2:只有右孩子节点
                self.__remove_node_22(node)
            else:                                       # 情况3:有两个节点,找右孩子的最小节点
                min_node = node.rchild
                while min_node.lchild:
                    min_node = min_node.lchild
                node.data = min_node.data
                if min_node.rchild:
                    self.__remove_node_22(min_node)
                else:
                    self.__remove_node_1(min_node)





###################################### 删除功能 #######################################


tree = BST([4,6,7,9,2,1,3,5,8])
tree.pre_order(tree.root)
print('')
tree.in_order(tree.root)        # 升序的
print('\n', tree.query_no_rec(4).data)
print(tree.query_no_rec(11))

tree.delete(4)
tree.delete(1)
tree.delete(8)
tree.in_order(tree.root)