# 用栈实现寻找迷宫从起点到终点的路径,深度优先算法

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

def maze_path(x1, y1, x2, y2):                                      # 起点与终点坐标,x为行数,y为列数
    path = [[x1, y1]]                                               # 创建栈记录走过的路径
    maze[x1][y1] = -1
    while len(path) > 0:
        curNode = path[-1]
        nextNode = None                                             # 如果这是None,就说明走到死路了
        x, y = curNode[0], curNode[1]
        for next in [[x-1,y], [x,y+1], [x+1,y], [x,y-1]]:           # 按上右下左的顺序试探能不能走得通
            if maze[next[0]][next[1]] == 0 and next not in path:    # 除了0(路)就是1(墙)和-1(走过的路)
                # 不能走死路也不能走回头路
                maze[next[0]][next[1]] = -1
                nextNode = next
                path.append(nextNode)
                break

        if path[-1] == [x2, y2]:                                    # 抵达终点
            return path

        if nextNode == None:                                        # 走到死胡同了,回溯
            [tmp_x, tmp_y] = path.pop()
            # maze[tmp_x][tmp_y] = -1                                 # 把这个点变成走过的点(-1)

    return None

print(maze_path(1, 1, 8, 8))
print(maze)

# print(maze_path(1, 1, 9, 9))









