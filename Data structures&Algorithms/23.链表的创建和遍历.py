# 手动实现链表功能,包括头插生成链表,尾插生成链表,顺序打印链表

class Node:
    def __init__(self, item):
        self.item = item
        self.next = None

def create_linklist_head(li):                                           # 头插法,通过列表创建链表,链表顺序与列表相反
    head = Node(li[0])
    for element in li[1:]:
        next_node = Node(element)
        next_node.next = head                                           # 往前一个一个加
        head = next_node
    return head

def create_linklist_tail(li):
    head = Node(li[0])
    tail = head                                                         # 注意这里不能写tail = Node(li[0]),相当于重开了
    for element in li[1:]:
        next_node = Node(element)
        tail.next = next_node                                           # 往后一个一个加
        tail = next_node
    return head

def print_linklist(head):
    while head:
        print(head.item, end=',')
        head = head.next


lk = create_linklist_head([1,2,3])
print_linklist(lk)
lk = create_linklist_tail([1,2,3])
print_linklist(lk)

