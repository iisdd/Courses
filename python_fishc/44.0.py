'''0. 按照课堂中的程序，如果开始计时的时间是（2022年2月22日16:30:30），
停止时间是（2025年1月23日15:30:30），那按照我们用停止时间减开始时间的
计算方式就会出现负数，你应该对此做一些转换'''

import time as t

class Mytimer():

    def __init__(self , hs , num = 1000000):
        self.unit = ['年','月','日','小时','分钟','秒']
        self.prompt = '未开始计时！'
        self.lasted = 0
        self.begin = 0
        self.end = 0
        self.rate = (0 , 12 , 31 , 24 , 60 , 60)
        self.default_timer = t.perf_counter
        self.func = hs
        self.num = num
    def __str__(self):
        
        return self.prompt

    __repr__ = __str__
    def __add__(self , other):
        prompt = '总共运行了：'
        result = []
        for index in range(6):
            result.append(self.lasted[index] + other.lasted[index])
            if result[index]:
                prompt +=( str (result[index]) + self.unit[index])
        return prompt
    
    #  开始计时
    def start(self):
        self.begin = t.localtime()
        self.prompt = '提示：请先调用 stop()停止计时！'
        print('计时开始...')

    #  停止计时
    def stop(self):
        if not self.begin:
            print ('提示：请先调用 start()开始计时！')
        else:
            self.end = t.localtime()
            
            self._calc()
            print('计时结束...')



    #  内部方法,计算运行时间，内部方法用下划线开头
    
    def _calc(self):
        self.lasted = []
        self.prompt = ''
        
        for index in range(6):   #时间元祖的索引值一共6个
            self.lasted.append(self.end[index] - self.begin[index])

        for index in range(6):  #  更正负数
            while  self.lasted[5 - index] < 0:
                self.lasted[4 - index] -= 1
                self.lasted[5 - index] += self.rate[5 - index]
            if self.lasted[5 - index]:
                self.prompt = str(self.lasted[5 - index]) + self.unit[5 - index] + self.prompt # 打印
        
        #  为下一轮计时初始化
        self.begin = 0
        self.end = 0
        print('总运行时间：' + self.prompt)

    #   设置默认计时器   
    def set_timer(self , timer):
        if timer == 'perf_counter':
            self.default_timer = t.perf_counter()
        elif timer == 'process_time':
            self.default_timer = t.process_time()
        else:
            print('输入有误，请输入perf_counter 或者 process_time')

    def timing(self):   #   计算函数运行时间
        self.start()
        for i in range(self.num):
            self.func()
        self.stop()
        


def test():
    text = "I love FishC.com!"
    char = 'o'
    if char in text:
        pass
t1 = Mytimer(test , 100000000)
t2 = Mytimer(test)


            
