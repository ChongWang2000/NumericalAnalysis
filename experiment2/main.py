from cmath import exp
import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy import dot
from prettytable import PrettyTable


def generate(data):
    data1 = data.loc[:, ['1.你的性别', '2.身高: ________:1[题目填空]', '3.体重: ________:1[题目填空]']]

    boy = pd.DataFrame(columns=['H', 'W'])
    girl = pd.DataFrame(columns=['H', 'W'])
    for index, row in data1.iterrows():
        if (row[0] == 'A.男生'):
            temp = pd.DataFrame([[row[1], row[2]]], columns=['H', 'W'])
            boy = boy.append(temp)
        elif (row[0] == 'B.女生'):
            temp = pd.DataFrame([[row[1], row[2]]], columns=['H', 'W'])
            girl = girl.append(temp)

    para = {'data': data1,
            'boy': boy,
            'girl': girl}

    return para


def sout(c, H, W):
    c1 = dot(dot(inv(dot(H.T, H)), H.T), W)  # 最小二乘法公式
    para = {'mean': np.mean(c)[0],
            'var': np.var(c)[0],
            'median': np.median(c),
            'r': c1[0][0]}
    return para


def table(p1, p2, p3):
    x = PrettyTable([" ", "girl", "boy", "all"])
    x.align[" "] = "l"  # Left align city names
    x.padding_width = 1
    x.add_row(["mean", p1['mean'], p2['mean'], p3['mean']])
    x.add_row(["variance", p1['var'], p2['var'], p3['var']])
    x.add_row(["median", p1['median'], p2['median'], p3['median']])
    x.add_row(["leastsquare", p1['r'], p2['r'], p3['r']])
    print(x)


def cH2(para):
    data = para['data']
    boy = para['boy']
    girl = para['girl']

    H = girl.loc[:, ['H']]
    H = np.multiply(H, H)
    W = girl.loc[:, ['W']]
    c = np.true_divide(W, H)
    p1 = sout(c, H, W)

    H = boy.loc[:, ['H']]
    H = np.multiply(H, H)
    W = boy.loc[:, ['W']]
    c = np.true_divide(W, H)
    p2 = sout(c, H, W)

    H = data.loc[:, ['2.身高: ________:1[题目填空]']]
    H = np.multiply(H, H)
    W = data.loc[:, ['3.体重: ________:1[题目填空]']]
    c = np.true_divide(W, H)
    p3 = sout(c, H, W)

    table(p1, p2, p3)


def cH3(para):
    data = para['data']
    boy = para['boy']
    girl = para['girl']

    H = girl.loc[:, ['H']]
    H2 = np.multiply(H, H)
    H3 = H * H2
    W = girl.loc[:, ['W']]
    c = np.true_divide(W, H3)
    p1 = sout(c, H3, W)

    H = boy.loc[:, ['H']]
    H2 = np.multiply(H, H)
    H3 = H * H2
    W = boy.loc[:, ['W']]
    c = np.true_divide(W, H3)
    p2 = sout(c, H3, W)

    H = data.loc[:, ['2.身高: ________:1[题目填空]']]
    H2 = np.multiply(H, H)
    H3 = H * H2
    W = data.loc[:, ['3.体重: ________:1[题目填空]']]

    c = np.true_divide(W, H3)
    p3 = sout(c, H3, W)

    table(p1, p2, p3)


def cHc(para):
    data = para['data']
    boy = para['boy']
    girl = para['girl']

    H = girl.loc[:, ['H']]
    H = np.log(H)
    H.insert(0, 'num', np.ones(25))
    W = girl.loc[:, ['W']]
    W = np.log(W)
    print("女生")
    c = dot(dot(inv(dot(H.T, H)), H.T), W)  # 最小二乘法公式
    c1 = exp(c[0][0])
    c2 = c[1][0]

    print(c1.real, c2)
    girlC1 = c[0][0]
    girlC2 = c[1][0]

    H = boy.loc[:, ['H']]
    H = np.log(H)
    H.insert(0, 'num', np.ones(104))
    W = boy.loc[:, ['W']]
    W = np.log(W)
    print("男生")
    c = dot(dot(inv(dot(H.T, H)), H.T), W)  # 最小二乘法公式
    c1 = exp(c[0][0])
    c2 = c[1][0]

    print(c1.real, c2)
    boyC1 = c[0][0]
    boyC2 = c[1][0]

    H = data.loc[:, ['2.身高: ________:1[题目填空]']]
    H = np.log(H)
    H.insert(0, 'num', np.ones(129))
    W = data.loc[:, ['3.体重: ________:1[题目填空]']]
    W = np.log(W)
    print("总体")

    c = dot(dot(inv(dot(H.T, H)), H.T), W)  # 最小二乘法公式

    c1 = exp(c[0][0])
    c2 = c[1][0]

    print(c1.real, c2)
    allC1 = c[0][0]
    allC2 = c[1][0]

    parameters = {"boyC1": boyC1,
                  "boyC2": boyC2,
                  "girlC1": girlC1,
                  "girlC2": girlC2,
                  "allC1": allC1,
                  "allC2": allC2}
    return parameters


def cHc2(para, parameters):
    data = para['data']
    boy = para['boy']
    girl = para['girl']

    boyC = parameters['boyC2']
    girlC = parameters['girlC2']
    allC = parameters['allC2']

    H = girl.loc[:, ['H']]
    H = np.power(H, girlC)
    W = girl.loc[:, ['W']]
    c = np.true_divide(W, H)
    p1 = sout(c, H, W)

    H = boy.loc[:, ['H']]
    H = np.power(H, boyC)
    W = boy.loc[:, ['W']]
    c = np.true_divide(W, H)
    p2 = sout(c, H, W)

    H = data.loc[:, ['2.身高: ________:1[题目填空]']]
    H = np.power(H, allC)
    W = data.loc[:, ['3.体重: ________:1[题目填空]']]

    c = np.true_divide(W, H)
    p3 = sout(c, H, W)

    table(p1, p2, p3)


def c1Hc(para, parameters):
    data = para['data']
    boy = para['boy']
    girl = para['girl']

    boyC = parameters['boyC1']
    girlC = parameters['girlC1']
    allC = parameters['allC1']

    H = girl.loc[:, ['H']]
    H = np.log(H)
    W = girl.loc[:, ['W']]
    W = np.log(W)
    W = W - girlC
    c = np.true_divide(W, H)
    p1 = sout(c, H, W)

    H = boy.loc[:, ['H']]
    H = np.log(H)
    W = boy.loc[:, ['W']]
    W = np.log(W)
    W = W - boyC
    c = np.true_divide(W, H)
    p2 = sout(c, H, W)

    H = data.loc[:, ['2.身高: ________:1[题目填空]']]
    H = np.log(H)
    W = data.loc[:, ['3.体重: ________:1[题目填空]']]
    W = np.log(W)
    W = W - allC
    c = np.true_divide(W, H)
    p3 = sout(c, H, W)

    table(p1, p2, p3)


if __name__ == '__main__':
    data = pd.read_csv("data.csv")  # 读取csv文件
    d = generate(data)
    cH2(d)
    print('\n')
    cH3(d)
    print('\n')
    para = cHc(d)
    print('\n')
    cHc2(d, para)
    print('\n')
    c1Hc(d, para)


    x=np.eye(5,dtype=int)
    print(x[0][0])

    y=np.diag([1,2,3])
    print(y)
    w,v=np.linalg.eig(y)
    print(w)
    print(v)


    z=np.arange(9).reshape(3,3)
    print(z)
    z=np.append(z,[1,1,1]).reshape(3,4)
    print(z)


    v=np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(v)

    a=np.array([1,2,3,4]).reshape(1,4)
    print(a.T)

    x=np.random.randn(4,3)
    x=np.sort(x)
    print(x)
    print(np.var(x))

    x=np.arange(36).reshape(6,6)
    print(x[:,2])


    x=np.array([[0,1,2],[1,0,3],[4,-3,8]])
    print(np.linalg.inv(x))

    df = pd.DataFrame({'A': np.random.randint(1, 100, 4), 'B': pd.date_range(start='20130101', periods=4, freq='D'),
                       'C': pd.Series([1, 2, 3, 4], index=['zhang', 'li', 'zhou', 'wang'], dtype='float32'),
                       'D': np.array([3] * 4, dtype='int32'), 'E': pd.Categorical(["test", "train", "test", "train"]),
                       'F': 'foo'})
    print(df)
    dates=pd.date_range("20200105",periods=3)
    df=pd.DataFrame({'A':np.array([3]*3),'B':[1,5,6]},index=dates)
    print(df)
    print(df.index)
    print(df.columns)

    df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3'], 'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']}, index=[0, 1, 2, 3])
    df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'], 'B': ['B4', 'B5', 'B6', 'B7'], 'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']}, index=[4, 5, 6, 7])
    df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'], 'B': ['B8', 'B9', 'B10', 'B11'], 'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']}, index=[8, 9, 10, 11])
    a=pd.concat([df1, df2])
    print(a)


    # np.array([[1,1],[2,2]])