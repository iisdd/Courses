# 用队列实现寻找迷宫从起点到终点的路径,广度优先算法

from collections import deque
# 1代表墙,0代表路
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

def print_path(li):                                                       # 根据节点列表打印路径
    curNode = li[-1]
    (x, y, idx) = curNode
    path = []
    while idx != -1:
        path.append((x, y))
        curNode = li[idx]
        (x, y, idx) = curNode
    curNode = li[0]
    (x, y, idx) = curNode
    path.append((x, y))
    return path


def maze_path_queue(x1, y1, x2, y2):
    queue = deque()
    queue.append((x1, y1, -1))                                            # 存长度为三的元组,前两个是坐标,第三个存出身
    Node = []                                                             # 记录每个节点的坐标&下标
    maze[x1][y1] = -1
    while len(queue) > 0:
        curNode = queue.popleft()                                         # 当前节点出队
        (x, y, idx) = curNode
        Node.append(curNode)
        for next in [(x-1,y), (x,y+1), (x+1,y), (x,y-1)]:                 # 把从当前节点出发的所有可能的走法记录进队列
            if maze[next[0]][next[1]] == 0:                               # 除了0(路)就是1(墙)和-1(走过的路)
                queue.append((next[0], next[1], len(Node)-1))             # 记录它的上一节点的idx
                maze[next[0]][next[1]] = -1
            if next[0] == x2 and next[1] == y2:                           # 找到终点!!!
                Node.append((next[0], next[1], len(Node)-1))
                print('最短路径: ', print_path(Node)[::-1])
                return True
    return False

print(maze_path_queue(1,1,8,8))

print(maze_path_queue(1,1,9,9))
