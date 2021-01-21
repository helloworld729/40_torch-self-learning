# multiprocess:跨平台的多进程模块  该模块提供process类代表进程对象
from multiprocessing import Process
import os

# 子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))
    for i in range(10):
        print(i)

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))  # 创建Process实例，只需传入函数和函数参数
    print('Child process will start.')
    p.start()  # 进程启动
    p.join()   # 等待子进程结束后再继续运行，一般用于进程同步,否则先print下一句，再执行子进程
    print('Child process end.')