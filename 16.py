import _thread
import math
import threading
import time

import numpy as np


'''
================================================
State类
8数码问题中的节点类
包含以下属性与函数
------------------------------------------------
属性：
deepcost:深度 d（n）
cost:总代价函数 f（n）
fin：目标状态
parent；父节点
max_deep：深度最大值
method：启发函数编号 
        #启发函数1：错位启发
        #启发函数2：距离启发
        #启发函数3：逆转惩罚
        #启发函数4：错位启发+3倍逆转惩罚
------------------------------------------------
函数：
__init__ 初始化函数
get_fin get函数，获取节点目标状态
get_method get函数，获取节点启发函数方式
get_cost get函数，获取节点f(n)（代价函数）
clc_cost函数 计算代价函数f（n）其中f（n）= d（n）+ h（n）
get_open函数，根据当前节点状态，计算出子节点，作为open集的待选集
is_equ函数：判断节点自身状态是否与对比状态相等
position_cost函数：计算错位代价，并返回。错位代价为与目标错位数
distance_cost函数：计算当前状态与目标状态的距离并返回
reverse_cost函数：计算当前状态与目标状态之间的逆转情况数并返回
position_3reverse_cost函数：返回错位代价与3倍逆转数
position_distance_3reverse_cost函数：返回错位代价与距离代价与3倍逆转数
__dis函数：计算两点的坐标的城区距离
print函数：打印当前节点状态
================================================
'''

COL = 4  # 棋盘的行数
ROW = 4  # 棋盘的列数


class State:
    deepcost = 0  # 深度 d（n）
    cost = 0  # 总代价函数 f（n）
    max_deep = 200  # 深度最大值

    """
    __init__ 初始化函数
    input：
        state节点状态
        deepcost深度
        parent(=None)父节点
        fin(=None)目标状态
        method(=None)启发函数方式   
    output：
        无output
    """

    def __init__(self, state, deepcost, parent=None, fin=None, method=None):  # 初始化函数
        self.state = state
        self.deepcost = deepcost
        # 下为初始节点父节点为空的处理方式
        self.parent = parent  # 父节点
        self.fin = parent.get_fin() if fin is None else fin  # 未设定时，子节点与父节点目标状态相同
        self.method = parent.get_method() if method is None else method  # 未设定时，子节点与父节点启发函数方式相同

    """
    get_fin get函数，获取节点目标状态
    output：fin目标状态  
    """

    def get_fin(self):
        return self.fin  # 节点目标状态

    """
    get_method get函数，获取节点启发函数方式
    output：method目标状态  
    """

    def get_method(self):
        return self.method  # 节点启发函数方式

    """ 
    clc_cost函数 计算代价函数f（n）
        包含f（n）= d（n）+ h（n）
    无input，output
    """

    def clc_cost(self):
        fin_state = self.fin  # 目标状态
        h_cost = 0  # 启发代价 h（n）

        method = self.method % 7
        is_deepcost = self.method > 7
        is_use_log = self.method > 14

        if method == 100:  # 随机搜索
            h_cost = 0
        elif method == 1:
            h_cost = self.position_cost(fin_state)  # 启发函数1：错位启发
        elif method == 2:
            h_cost = self.distance_cost(fin_state)  # 启发函数2：距离启发
        elif method == 3:
            h_cost = self.reverse_cost(fin_state)  # 启发函数3：逆转惩罚
        elif method == 4:
            h_cost = self.position_cost(fin_state) + self.distance_cost(fin_state) # 启发函数4：错位启发+距离启发
        elif method == 5:
            h_cost = self.position_cost(fin_state) + 3 * self.reverse_cost(fin_state)  # 启发函数5：错位启发+3倍逆转惩罚
        elif method == 6:
            h_cost = self.distance_cost(fin_state) +  3 * self.reverse_cost(fin_state)  # 启发函数6：距离启发+3倍逆转惩罚
        elif method == 0:
            h_cost = self.position_distance_3reverse_cost(fin_state)  # 启发函数7：错位启发+距离启发+3倍逆转惩罚

        deepcost = math.log(self.deepcost + 1) if is_use_log else self.deepcost

        self.cost = deepcost + h_cost if is_deepcost else h_cost

        # 纯启发函数 f(n)=d(n)+h(n) 其中d(n)=0
        # self.cost = self.deepcost + h_cost #此处为A算法，启发




    """ 
    get_cost get函数，获取节点f(n)（代价函数）
    output：节点代价  
    """

    def get_cost(self):
        self.clc_cost()  # 计算cost
        return self.cost

    """ 
        get_deepcost get函数，获取节点d(n)
        output：节点深度  
    """
    def get_deepcost(self):
        return self.deepcost
    """ 
    get_open函数，根据当前节点状态，计算出子节点，作为open集的待选集
    input：
        无
    output：
        sub_States节点子节点集合
    """

    def get_open(self):
        sub_States = []  # 子节点集合
        directions = ['left', 'right', 'up', 'down']  # 空白（0）可移动的方向
        row, col = np.where(self.state == 0)  # 获取空白（0）的坐标位置
        if self.deepcost >= self.max_deep:  # 如果深度大于规定最大深度，返回空，目的为避免深度一直拓展
            return sub_States
        for direction in directions:
            if 'left' == direction and col > 0:  # 左移
                s = self.state.copy()
                s[row, col], s[row, col - 1] = s[row, col - 1], s[row, col]  # 移动
                new = State(s, self.deepcost + 1, self)  # 初始化节点，移动后的状态，深度加1，父节点指向self
                sub_States.append(new)
            if 'up' == direction and row > 0:  # 上移
                s = self.state.copy()
                s[row, col], s[row - 1, col] = s[row - 1, col], s[row, col]
                new = State(s, self.deepcost + 1, self)
                sub_States.append(new)
            if 'down' == direction and row < ROW - 1:  # 下移
                s = self.state.copy()
                s[row, col], s[row + 1, col] = s[row + 1, col], s[row, col]
                new = State(s, self.deepcost + 1, self)
                sub_States.append(new)
            if 'right' == direction and col < COL - 1:  # 右移
                s = self.state.copy()
                s[row, col], s[row, col + 1] = s[row, col + 1], s[row, col]
                new = State(s, self.deepcost + 1, self)
                sub_States.append(new)
        return sub_States

    """ 
    is_equ函数：判断节点自身状态是否与对比状态相等，相等返回True，否则返回False
    input：
        state：对比状态
    output：
        True或False
    """

    def is_equ(self, state):
        if self.position_cost(state) == 0:
            return True
        else:
            return False

    """ 
    print函数：打印当前节点状态
    无input，output 
    """

    def print(self):
        for i in range(ROW):
            for j in range(COL):
                print(self.state[i, j], end='   ')
            print("\n")

    """
    position_cost函数：计算错位代价，并返回。错位代价为与目标错位数
    input：
        state：目标状态 np.array(3*3)
    output：
        cost：错位代价 
    """

    def position_cost(self, state):
        count = 0  # 正确位计数器
        for i in range(ROW):
            for j in range(COL):
                if self.state[i, j] == state[i, j] and state[i, j] > 0:  # 对应位相等且非空白点
                    count += 1
        return ROW * COL - 1 - count  # 错位计算

    """  
    distance_cost函数：计算当前状态与目标状态的距离并返回
        这个距离为每一个点到目标正确位置的城区距离
    input：
        state：目标状态  np.array(3*3)
    output：
        cost：距离代价  
    """

    def distance_cost(self, state):
        dis = 0  # 距离
        position1 = {}  # 数：位置
        position2 = {}
        for i in range(ROW):
            for j in range(COL):
                position1[state[i, j]] = [i, j]  # 记录目标转态中每个数对应的坐标
                position2[self.state[i, j]] = [i, j]  # 记录当前转态中每个数对应的坐标
        # 计算当前状态与目标状态，每个对应数的距离
        for n in range(ROW * COL):
            if n == 0:
                continue
            dis += self.__dis(position2[n], position1[n])  # 城区距离
        return dis

    """
    __dis函数：计算两点的坐标的城区距离
    input：
        a:点1[x1,y1]
        b:点2[x2,y2]
    output：
        dis：距离  
    """

    def __dis(self, a, b):  # 曼哈顿城区距离
        dis = abs(a[0] - b[0]) + abs(a[1] - b[1])
        return dis

    def __dis2(self, a, b):
        dis = (a[0] - b[0]) ^ 2 + (a[1] - b[1]) ^ 2
        return dis

    """ 
    reverse_cost函数：计算当前状态与目标状态之间的逆转情况数并返回
    input：
        state：目标状态  np.array(3*3)
    output：
        count: 逆转数  
    """

    def reverse_cost(self, state):  # 逆转
        wrong = [[i, j] for i in range(ROW) for j in range(COL) if
                 self.state[i, j] != state[i, j] and state[i, j] != 0]  # 获取不匹配的点
        count = 0  # 逆转数
        for i in range(len(wrong)):
            for j in range(i + 1, len(wrong)):
                if self.__dis(wrong[i], wrong[j]) == 1:  # 相邻
                    if (self.state[tuple(wrong[i])], self.state[tuple(wrong[j])]) == (
                            state[tuple(wrong[j])], state[tuple(wrong[i])]):  # 逆转
                        count += 1
        return count  # 逆转数越少越好

    """  
    position_3reverse_cost函数：返回错位代价与3倍逆转数
    input：
        state：目标状态  np.array(3*3)
    output：
        错位代价与3倍逆转数代价
    """

    def position_3reverse_cost(self, state):
        return self.position_cost(state) + 3 * self.reverse_cost(state)

    """  
    position_distance_3reverse_cost函数：返回错位代价与距离代价与3倍逆转数
    input：
        state：目标状态  np.array(3*3)
    output：
        错位代价与距离代价与3倍逆转数代价
    """

    def position_distance_3reverse_cost(self, state):
        return self.position_cost(fin_state) + self.distance_cost(fin_state) + 3 * self.reverse_cost(fin_state)


"""  
get_cost函数：返回State的cost
input：
    State:状态类
output:
    State的cost
"""


def get_cost(State):
    return State.get_cost()



"""  
get_cost函数：返回State的deepcost
input：
    State:状态类
output:
    State的deepcost
"""
def get_deepcost(State):
    return State.get_deepcost()



"""    
solve函数：8数码寻路函数，利用不同方式找到初始8数码状态到最终8数码状态的路径
input: 
    ini_state:初始状态,类型：np.array，3*3
    fin_state:最终状态,类型：np.array，3*3
    method:解决问题方式
        方式1：#启发函数1：错位启发
        方式2：#启发函数2：距离启发
        方式3：#启发函数3：逆转惩罚
        方式4：#启发函数4：错位启发 + 距离启发
        方式5：#启发函数5：错位启发+3倍逆转惩罚
        方式6：#启发函数6：距离启发 + 3倍逆转惩罚
        方式7：#启发函数7：错位启发 + 距离启发 + 3倍逆转惩罚
        ----------------------
output：
    [a,b] = [close_list,len(close_list)-1]形式
    a（close_list）：最终路径节点集合
    b:步数
"""


def solve(ini_state, fin_state, method):
    close_list = []  # 闭集
    open_list = []  # 开集
    ini = State(ini_state, 0, None, fin_state, method)  # 初始化节点
    open_list.append(ini)
    i = 0

    T1 = time.time()
    none_count = 0
    while True:
        i += 1
        # print(i,"迭代")
        if len(open_list) == 0:  # 如果开集一开始为空，则失败，退出
            print("you fail!")
            break
        res = sorted(open_list, key=get_cost)  # 开集根据节点的cost（f(n)）来升序排列
        # res[0].print()
        min_cost_State = res[0]
        close_list.append(res[0])  # 将重排后的开集第一个，也就是代价最小的放入闭集
        open_list.remove(res[0])  # 将代价最小的移出开集
        if min_cost_State.is_equ(fin_state):  # 如果达到目标状态，打印结果，结束
            # print_inf(close_list)
            break
        temp = min_cost_State.get_open()  # 获取这个新加入闭集的节点的子集


        if len(temp) == 0:
            none_count += 1
            if none_count > 100:
                print("fail! ")
                return [[None], -1, -1]

        for a in temp:
            count = 0  # 是否在开集与闭集中出现过的标志位
            if not open_list is None:
                for b in open_list:
                    if a.is_equ(b.state):  # 是否于开集某一个节点状态相等
                        count = 1
                        if a.get_cost() < b.get_cost():
                            open_list.remove(b)
                            open_list.append(a)
            for c in close_list:
                if a.is_equ(c.state):  # 是否于闭集某一个节点状态相等
                    count = 1
                    if a.get_cost() < c.get_cost():
                        close_list.remove(c)
                        close_list.append(a)
            if count == 1:  # 已存在，则跳过
                continue
            else:  # 未出现，则加入开集
                open_list.append(a)
        T2 = time.time()
        if T2-T1 >= 5*60:
            print("=====超时：启发函数%d运行时间超过5分钟，自动退出，此次结果无效===="%method)
            break
    return [close_list, len(close_list) - 1, len(open_list)]  # 返回闭集与路径步数


def print_inf1(res):
    for i, each in enumerate(res):
        print(i)
        print("cost:", each.cost)
        each.print()


"""  
print_inf函数：打印从初始状态到达目标转态的路径
input：
    res：结果集，类型为[State,...]
无output
"""

def print_inf(res):
    print("########################")
    for i, each in enumerate(res):
        if i == 0:
            print("第%d步：" % i)
        else:
            print("=>第%d步：" % i)
        each.print()
        # print(each.cost)
    print("总共%d步" % (len(res) - 1))
    print("########################")




if __name__ == "__main__":

    fin_res = []
    ini_1 = [[5, 1, 2, 4], [9, 6, 3, 8], [13, 15, 10, 11], [14, 0, 7, 12]]
    ini_2 = [[11, 9, 4, 15], [1, 3, 0, 12], [7, 5, 8, 6], [13, 2, 10, 14]]
    fin = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]

    ini_state_1 = np.array(ini_1)
    ini_state_2 = np.array(ini_2)
    fin_state = np.array(fin)

    """
    ======================================================
    |method为搜索方式
    |   method = 1 启发函数1：错位启发
    |   method = 2 启发函数2：距离启发
    |   method = 3 启发函数3：逆转惩罚
    |   method = 4 启发函数4：错位启发 + 距离启发
    |   method = 5 启发函数5：错位启发 + 3倍逆转惩罚
    |   method = 6 启发函数6：距离启发 + 3倍逆转惩罚
    |   method = 7 启发函数7：错位启发 + 距离启发 + 3倍逆转惩罚
    |   method = 8 A算法：广度优先 + 启发函数1：错位启发
    |   method = 9 A算法：广度优先 + 启发函数2：距离启发
    |   method = 10 A算法：广度优先 + 启发函数3：逆转惩罚
    |   method = 11 A算法：广度优先 + 启发函数4：错位启发 + 距离启发
    |   method = 12 A算法：广度优先 + 启发函数5：错位启发 + 3倍逆转惩罚
    |   method = 13 A算法：广度优先 + 启发函数6：错位启发 + 3倍逆转惩罚
    |   method = 14 A算法：广度优先 + 启发函数7：错位启发 + 3倍逆转惩罚
    |   method = 100 盲目搜索
    ======================================================
    """



    method = 0

    for method in range(1):
        # if method == 1 or method == 3 or method == 6:
        #     continue
        #
        # if method == 1 or method == 4 or method == 7:
        #     continue
        method = 1+7*2
        T1 = time.time()
        res = solve(ini_state1, fin_state, method+1)



        # print("启发函数%d：共计%d步" % (method, res[1]))
        best_list = []
        temp = res[0][-1]
        while (not temp is None):
            best_list.append(temp)
            temp = temp.parent
        new_res = sorted(best_list, key=get_deepcost, reverse=False)
        print_inf(new_res)
        T2 = time.time()

        # print("初始状态：",ini_state)
        print("启发函数%d：" % (7 if (method+1)%7==0 else (method+1)%7))
        if method<=7-1:
            print("评价函数：f(n) = h(n)")
        elif method<=14-1:
            print("评价函数：f(n) = d(n) + h(n)")
        else:
            print("评价函数：f(n) = log(d(n)) + h(n)")

        print('程序运行时间:%s秒' % (T2 - T1))
        print("使用步数：%d步" % len(best_list))
        print("探索节点：%d 个（其中 open表 %d 个，close表 %d 个）" % (res[1] + res[2], res[1], res[2]))
        print("#######################################################")

