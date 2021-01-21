import argparse
import sys

# print('system argument：{}'.format(sys.argv))
# print('system version： {:<10}'.format(sys.version))
# print('max size： {:<10}'.format(sys.maxsize))
####################################################  基础  ###########################################################################
parser = argparse.ArgumentParser()
# parser.description = '输入两个数:输出乘积'  # 脚本描述信息
# parser.add_argument('-a', '--parta', help='I am a', type=int, default=2)  # 有引号，但是是一个变量，加--表示不确定是否输入，-a为简称
# parser.add_argument('-b', '--partb', help='I am b', type=int, default=3)
# args = parser.parse_args()

# if args.parta:
#     print('get a:{}'.format(args.parta))  # 不可以写a
# if args.partb:
#     print('get b:{}'.format(args.partb))
# if args.partb and args.partb:
#     print('get a and b:{}{}'.format(args.parta, args.partb))
#     print(args.parta * args.partb)
###################################################位置参数########################################################################
# parser.add_argument("echo", help='input the value of echo, I will output it')
# args = parser.parse_args()
# print(args.echo)

################################################### 指定type ################################################################
# parser.add_argument("squre", help='正方形的变长', type=int)
# args = parser.parse_args()
# answer = args.squre ** 2
# print(answer)
################################################### 可选参数 ###############################
# parser.add_argument("--verbosity", help='Add output verbosity')
# args = parser.parse_args()
# if args.verbosity:
#     print('verbosity turned on')
################################################### Action参数 ###############################
# parser.add_argument("-v", "--verbose", help='Add verbose', action='store_true')
# args = parser.parse_args()
# if args.verbose:
#     print('verbose turned on')
################################################### 位置+可选 ###############################
# parser.add_argument("-v", "--verbose", help='Add verbose', action='store_true')
# parser.add_argument("square", type=int, help='输入正方形的变长：')
# args = parser.parse_args()
# answer = args.square ** 2
# if args.verbose:
#     print('变长为：{}的正方形，面积是：{}'.format(args.square, answer))
# else:
#     print(answer)
################################################### choice参数 ##############################
# parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2])
# parser.add_argument("square", type=int, help='正方形的边长')
# args = parser.parse_args()
# answer = args.square ** 2
# if args.verbose == 2:
#     print('变长为：{}的正方形，面积是：{}'.format(args.square, answer))
# elif args.verbose == 1:
#     print("{}^2 == {}".format(args.square, answer))
# else:
#     print(answer)
################################################### 冲突处理 ###############################
# group = parser.add_mutually_exclusive_group()
# group.add_argument("-v", "--verbose", action='store_true')
# group.add_argument("-q", "--quite",   action='store_true')
# parser.add_argument("square", type=int, help='正方形的边长')
# args = parser.parse_args()
# answer = args.square ** 2
# if args.verbose:
#     print('变长为：{}的正方形，面积是：{}'.format(args.square, answer))
# elif args.quite == 1:
#     print("{}^2 == {}".format(args.square, answer))
# else:
#     print(answer)
################################################### 注释信息 ###############################
parser = argparse.ArgumentParser(description='输入正方形的变长，输出面积信息...')
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action='store_true')
group.add_argument("-q", "--quite",   action='store_true')
parser.add_argument("--square", type=int, help='正方形的边长', default=5)
args = parser.parse_args()
answer = args.square ** 2
if args.verbose:
    print('变长为：{}的正方形，面积是：{}'.format(args.square, answer))
elif args.quite == 1:
    print("{}^2 == {}".format(args.square, answer))
else:
    print(answer)