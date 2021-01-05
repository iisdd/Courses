from bst import BiTreeNode, BST

class AVLNode(BiTreeNode):                              # 继承树节点
    def __init__(self, data):
        BiTreeNode.__init__(self, data)
        self.bf = 0                                     # balance factor就是左子树的深度减右子树的深度

class AVLTree(BST):
    def __init__(self, li=None):
        BST.__init__(self, li)

    def rotate_left(self, p, c):                        # 左旋,对右孩子右子树插入,p为父亲节点,c为孩子节点
        # 甩包袱
        s2 = c.lchild
        p.rchild = s2
        if s2:                                          # s2节点不为空
            s2.parent = p
        # 上位
        c.lchild = p
        p.parent = c
        p.bf = 0
        c.bf = 0
        return c                                        # 返回新树的根节点

    def rotate_right(self, p, c):                       # 右旋,对左孩子左子树插入
        s2 = c.rchild
        p.lchild = s2
        if s2:
            s2.parent = p
        c.rchild = p
        p.parent = c
        p.bf = 0
        c.bf = 0
        return c

    def rotate_right_left(self, p, c):
        g = c.lchild                                    # g: grandchild
        ############################ 右旋 ###########################
        # 甩包袱给c
        s3 = g.rchild
        c.lchild = s3
        if s3:                                          # 反着连回去就要判断空
            s3.parent = c
        # g上位
        c = g.rchild
        c.parent = g
        ############################ 右旋 ###########################
        ############################ 左旋 ###########################
        s2 = g.lchild
        # 甩包袱给p
        p.rchild = s2
        if s2:
            s2.parent = p
        # g上位
        g.lchild = p
        p.parent = g
        ############################ 右旋 ###########################
        # 更新bf
        if g.bf > 0:                                    # 插在s2的位置上
            p.bf = 0
            c.bf = -1
        elif g.bf < 0:                                  # 插在s3的位置上
            p.bf = 1
            c.bf = 0
        else:                                           # g.bf=0只可能有一种情况,即插在g的位置上
            p.bf = 0
            c.bf = 0

        g.bf = 0
        return g

    def rotate_left_right(self, p, c):
        g = c.rchild
        ############################ 左旋 ###########################
        # 甩包袱给c
        s2 = g.lchild
        c.rchild = s2
        if s2:
            s2.parent = c
        # g上位
        g.lchild = c
        c.parent = g
        ############################ 左旋 ###########################

        ############################ 右旋 ###########################
        # 甩包袱给p
        s3 = g.rchild
        p.lchild = s3
        if s3:
            s3.parent = p
        # g上位
        g.rchild = p
        p.parent = g
        ############################ 右旋 ###########################
        # 更新bf
        if g.bf > 0:                                    # 插在s2
            c.bf = 0
            p.bf = -1
        elif g.bf < 0:                                  # 插在s3
            c.bf = 1
            p.bf = 0
        else:                                           # g.bf=0 -> 插在g
            c.bf = 0
            p.bf = 0

        g.bf = 0
        return g

    def insert_no_rec(self, val):                       # 非递归形式的插入
        # 1.和BST一样,插入
        p = self.root
        if not p:                                       # 空树
            self.root = AVLNode(val)
            return
        while 1:
            if p.data > val:
                if p.lchild:                            # 存在左孩子
                    p = p.lchild
                else:                                   # 左边没有节点,捏一个节点
                    p.lchild = AVLNode(val)
                    p.lchild.parent = p
                    node = p.lchild
                    break
            elif p.data < val:
                if p.rchild:
                    p = p.rchild
                else:
                    p.rchild = AVLNode(val)
                    p.rchild.parent = p
                    node = p.rchild
                    break
            else:                                       # 已经有一个相同的节点了,就不插入了
                return

        # 2.更新bf,这里用的右子树深度-左子树深度,和前面相反
        while node.parent:                              # node还有父节点,即没到根节点
            if node.parent.lchild == node:              # node是其父节点的左孩子,即左边更沉,node.parent.bf -= 1
                if node.parent.bf < 0:                  # 原来是-1,现在左边又插入一个变成-2了,那就得旋转了
                    # 做旋转,看node哪边沉,分两种情况: 左左->右旋, 左右->左旋右旋
                    n = node.parent.parent              # 为了连接旋转后的子树,n为根节点,g为子树的根节点
                    x = node.parent                     # 旋转前node的parent
                    if node.bf > 0:                     # 左右
                        g = self.rotate_left_right(node.parent, node)
                    else:                               # 右旋
                        g = self.rotate_right(node.parent, node)
                elif node.parent.bf > 0:                # 原来是1,现在左边插入一个变成0,可以在这停止了
                    node.parent.bf = 0
                    break
                else:                                   # 原来是0,现在变-1,还能接受,再往上一层看看,到-2就算崩了
                    node.parent.bf = -1
                    node = node.parent
                    continue

            else:                                       # node是其父节点的右孩子,即右边更沉,node.parent.bf += 1
                if node.parent.bf > 0:                  # 原来是1,现在右边插入一个变成2,崩了,旋转!
                    # 看哪边沉,分两种情况: 右右->左旋, 右左->右旋左旋
                    n = node.parent.parent
                    x = node.parent                     # 旋转前node的parent
                    if node.bf < 0:                     # 右左->右旋左旋
                        g = self.rotate_right_left(node.parent, node)
                    else:                               # 右右->左旋
                        g = self.rotate_left(node.parent, node)
                elif node.parent.bf < 0:                # 原来是-1,现在右边插入变成0,到此为止
                    node.parent.bf = 0
                    break
                else:                                   # 原来是0,现在变成1,还能接受,再往上一层看看,到2才停
                    node.parent.bf = 1
                    node = node.parent
                    continue

            # 记得连起来
            # 走到这里时AVL树肯定已经平衡了,bf=-1的情况会continue,只有旋转完或者bf=0才会走到这
            g.parent = n
            if n:                                       # 根节点不为空
                if x == n.lchild:
                    n.lchild = g
                else:
                    n.rchild = g
            else:                                       # 根节点为空
                self.root = g
            break  # AVL树已经调整平衡了,跳出while循环


tree = AVLTree([9,8,7,6,5,4,3,2,1])
tree.pre_order(tree.root)
print('')
tree.in_order(tree.root)




