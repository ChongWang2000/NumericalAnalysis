import time
import numpy as np
from numpy.linalg import inv
import math
from prettytable import PrettyTable
import copy


def tableprint(vector):
    s = "["
    print("[", end="")
    for i in range(np.shape(vector)[0]):
        if i != 0:
            # s += ","
            # s += "\n"
            if (i) % 2 == 0:
                s += ","
                s += "\n"
            else:
                s += ",    "
        if isinstance(vector[i], np.ndarray):
            s += ("{0.real:.5f}+{0.imag:.5f}j".format(vector[i, 0]))
        else:
            s += ("%.5f" % vector[i])
        if i != 0 and i % 6 == 0:
            s += ""
    s += "]"
    return s


# 利用无穷范数归一化的幂法迭代
def power_method(A, iterationnum, min_error):
    m = A.shape[0]
    x0 = np.ones((m, 1))
    y0 = x0
    x1 = A.dot(y0)
    k = 0
    start = time.process_time_ns()
    while k < iterationnum and np.abs(np.max(np.abs(x0)) - np.max(np.abs(x1))) > min_error:
        y1 = x1 / (x1.flat[abs(x1).argmax()])
        x1 = A.dot(y1)
        k += 1
    pld = x1.flat[abs(x1).argmax()]
    env = x1 / (x1.flat[abs(x1).argmax()])
    end = time.process_time_ns()
    runtime = (end - start) / 1000000
    para = {"pld": pld,
            "env": env,
            "runtime": runtime}
    return para


# 利用第二范数归一化的幂法迭代
def power_method1(A, iterationnum, min_error):
    m = A.shape[0]
    x1 = np.zeros((m, 1))
    x1[0][0]=1
    # y0 = x0
    # x1 = A.dot(y0)
    k = 0
    start = time.process_time_ns()
    while k < iterationnum :
        y1 = x1 / np.linalg.norm(x1)
        x1 = A.dot(y1)
        pld = np.dot(y1.T, x1)
        k += 1
    env = x1 / np.linalg.norm(x1)
    end = time.process_time_ns()
    runtime = (end - start) / 1000000
    para = {"pld": pld,
            "env": env,
            "runtime": runtime}
    return para


# 计算角度
def cal_theta(d):
    a = math.atan(d) / 2
    cos = math.cos(a)
    sin = math.sin(a)
    return [cos, sin]


# 找非对角线元素的最大值
def find_max(A):
    [row, column] = np.shape(A)
    ipos = 0
    jpos = 0
    max = -100000
    for i in range(row):
        for j in range(i + 1, column):
            if abs(A[i][j]) > max:
                max = abs(A[i][j])
                ipos = i
                jpos = j
    return [max, ipos, jpos]


# jacobi方法
def jacobi_method(A, iterationnum, min_error):
    now_iteration = 0
    [row, column] = np.shape(A)
    start = time.process_time_ns()
    [max, p, q] = find_max(A)
    while now_iteration < iterationnum and max >= min_error:
        now_iteration += 1
        if abs(A[p][p] - A[q][q]) < min_error:
            cos = math.cos(math.pi / 4)
            sin = math.cos(math.pi / 4)
        else:
            d = (2 * A[p][q]) / (A[p][p] - A[q][q])
            [cos, sin] = cal_theta(d)
        [row, column] = np.shape(A)
        P = np.eye(row)
        P[p][p] = cos
        P[p][q] = -sin
        P[q][p] = sin
        P[q][q] = cos
        A = P.T @ A @ P
        [max, p, q] = find_max(A)
    end = time.process_time_ns()
    runtime = (end - start) / 1000000
    result = [A[i][i] for i in range(row)]
    result = np.sort(result)
    para = {"res": result,
            "runtime": runtime}
    return para


# 高斯相似变换求上hessenberg矩阵
def gauss_hessen(A):
    min_error = 1e-20
    [row, column] = np.shape(A)
    k = 0
    while k < (column - 2):
        G = np.eye(row)
        x = A[:, k][k + 1]
        if abs(x) >= min_error:
            for i in range(row - (k + 2)):
                a = -(A[:, k][k + 2 + i] / x)
                G[(k + 2 + i)][(k + 1)] = a
            A = G @ A @ inv(G)
        else:
            lie = A[:, k]
            now = k + 2
            while now < row:
                if (abs(lie[now]) >= min_error):
                    A[[k + 1, now], :] = A[[now, k + 1], :]
                    A[:[k + 1, now]] = A[:[now, k + 1]]
                    for i in range(row - (k + 2)):
                        a = -(A[:, k][k + 2 + i] / x)
                        G[(k + 2 + i)][(k + 1)] = a
                    A = G @ A @ inv(G)
                    break
                now += 1
        k += 1
    return A


def qr_aux(vector, A, i, j, iterationnum, min_error):
    m = A.shape[0]
    if m == 1:
        vector[i] = A[0, 0]
        return
    elif m == 2:
        b = -(A[0, 0] + A[1, 1])
        c = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        delta = b * b - 4 * c
        if delta >= 0:
            x1 = -b / 2 + np.sqrt(delta) / 2
            x2 = -b / 2 - np.sqrt(delta) / 2
        else:
            x1 = np.complex(-b / 2, np.sqrt(-delta) / 2)
            x2 = np.complex(-b / 2, -np.sqrt(-delta) / 2)
        vector[i], vector[i + 1] = x1, x2
        return
    else:
        a = gauss_hessen(A)
        for k in range(1, m):
            if np.abs(a[k, k - 1]) < min_error:
                qr_aux(vector, a[:k, :k], i, i + k, iterationnum, min_error)
                qr_aux(vector, a[k:, k:], i + k, j, iterationnum, min_error)
                return
    iters = 0
    while iters < iterationnum and np.max(np.abs(a[m - 1])) >= min_error:
        iters += 1
        u = a[m - 1, m - 1] * np.eye(m)
        q, r = np.linalg.qr(a - u)
        a = r @ q + u
        a = gauss_hessen(a)
    if iters < iterationnum:
        vector[j - 1] = a[m - 1, m - 1]
        qr_aux(vector, a[:m - 1, :m - 1], i, j - 1, iterationnum, min_error)
    else:
        disc = (a[m - 2, m - 2] - a[m - 1, m - 1]) ** 2 + \
               4 * a[m - 1, m - 2] * a[m - 2, m - 1]
        if disc >= 0:
            vector[j - 1] = (a[m - 2, m - 2] +
                             a[m - 1, m - 1] + np.sqrt(disc)) / 2
            vector[j - 2] = (a[m - 2, m - 2] +
                             a[m - 1, m - 1] - np.sqrt(disc)) / 2
        else:
            vector[j - 1] = np.complex((a[m - 2, m - 2] +
                                        a[m - 1, m - 1]) / 2, np.sqrt(-disc) / 2)
            vector[j - 2] = np.complex((a[m - 2, m - 2] +
                                        a[m - 1, m - 1]) / 2, -np.sqrt(-disc) / 2)
        qr_aux(vector, a[:m - 2, :m - 2], i, j - 2, iterationnum, min_error)


# QR算法
def QR_method(A, iterationnum=500, min_error=1e-4):
    m = A.shape[0]
    res = np.zeros((m, 1), dtype=np.complex)
    start = time.process_time_ns()
    qr_aux(res, A, 0, m, iterationnum, min_error)
    end = time.process_time_ns()
    para = {"res": res,
            "runtime": (end - start) / 1000000}
    return para


if __name__ == '__main__':
    A = np.array([[611, 196, -192, 407, -8, -52, -49, 29],
                  [196, 899, 113, -192, -71, -43, -8, -44],
                  [-192, 113, 899, 196, 61, 49, 8, 52],
                  [407, -192, 196, 611, 8, 44, 59, -23],
                  [-8, -71, 61, 8, 411, -599, 208, 208],
                  [-52, -43, 49, 44, -599, 411, 208, 208],
                  [-49, -8, 8, 59, 208, 208, 99, -911],
                  [29, -44, 52, -23, 208, 208, -911, 99]],
                 dtype=float)

    B = np.ones((10, 10))
    for i in range(10):
        for j in range(10):
            B[i][j] = 1 / (i + j + 1)

    C = np.ones((12, 12))
    for i in range(12):
        for j in range(12):
            k = max(i, j)
            C[i][j] = float(12 - k)

    D = np.ones((20, 20))
    for i in range(1, 21):
        for j in range(1, 21):
            D[i - 1][j - 1] = math.sqrt(2 / 21) * math.sin(i * j * math.pi / 21)

    E = np.ones((50, 50))
    for i in range(50):
        for j in range(50):
            if i == j or j == 49:
                E[i][j] = 1
            elif i < j:
                E[i][j] = 0
            else:
                E[i][j] = -1

    polynomial = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], ], dtype=float)

    # 用np.linalg.eig求解特征值
    [rpa, pae] = np.linalg.eig(A)
    rpa = np.sort(rpa)
    [rpb, pbe] = np.linalg.eig(B)
    rpb = np.sort(rpb)
    [rpc, pce] = np.linalg.eig(C)
    rpc = np.sort(rpc)
    [rpd, pde] = np.linalg.eig(D)
    rpd = np.sort(rpd)
    [rpe, pee] = np.linalg.eig(E)
    rpe = np.sort(rpe)
    x = PrettyTable([" ", "Eigenvalue"])
    x.align[" "] = "l"
    x.padding_width = 1
    x.add_row(["A", (rpa)])
    x.add_row(["B", (rpb)])
    x.add_row(["C", (rpc)])
    x.add_row(["D", (rpd)])
    x.add_row(["E", (rpe)])
    print(x)

    # 用幂法（无穷范数归一化）求解特征值
    pa = power_method(A, 1000, 1e-4)
    pb = power_method(B, 1000, 1e-4)
    pc = power_method(C, 1000, 1e-4)
    pd = power_method(D, 1000, 1e-4)
    pe = power_method(E, 1000, 1e-4)
    x = PrettyTable([" ", "Max Eigenvalue", "Eigenvector", "error", "Runtime", "remarks"])
    x.align[" "] = "l"
    x.padding_width = 1
    x.add_row(
        ["A", pa["pld"], tableprint(pa['env']), np.linalg.norm(np.dot(A, pa['env']) - pa["pld"] * pa['env'])/np.linalg.norm(pa['env']),
         pa['runtime'], "  "])
    x.add_row(
        ["B", pb["pld"], tableprint(pb['env']), np.linalg.norm(np.dot(B, pb['env']) - pb["pld"] * pb['env'])/np.linalg.norm(pb['env']),
         pb["runtime"], "  "])
    x.add_row(
        ["C", pc["pld"], tableprint(pc['env']), np.linalg.norm(np.dot(C, pc['env']) - pc["pld"] * pc['env'])/np.linalg.norm(pc['env']),
         pc["runtime"], "  "])
    x.add_row(
        ["D", pd["pld"], tableprint(pd['env']), np.linalg.norm(np.dot(D, pd['env']) - pd["pld"] * pd['env'])/np.linalg.norm(pd['env']),
         pd["runtime"], "  "])
    x.add_row(
        ["E", pe["pld"], tableprint(pe['env']), np.linalg.norm(np.dot(E, pe['env']) - pe["pld"] * pe['env'])/np.linalg.norm(pe['env']),
         pe["runtime"], "  "])
    print(x)

    print(tableprint(pa['env']))
    print(tableprint(pb['env']))
    print(tableprint(pc['env']))
    print(tableprint(pd['env']))
    print(tableprint(pe['env']))

    # 用幂法（二范数归一化）求解特征值
    pa = power_method1(A, 1000, 1e-4)
    pb = power_method1(B, 1000, 1e-4)
    pc = power_method1(C, 1000, 1e-4)
    pd = power_method1(D, 1000, 1e-4)
    pe = power_method1(E, 1000, 1e-4)
    x = PrettyTable([" ", "Max Eigenvalue", "Eigenvector", "error", "Runtime", "remarks"])
    x.align[" "] = "l"
    x.padding_width = 1
    x.add_row(
        ["A", pa["pld"], tableprint(pa['env']), np.linalg.norm(np.dot(A, pa['env']) - pa["pld"] * pa['env'])/np.linalg.norm(pa['env']),
         pa['runtime'], "  "])
    x.add_row(
        ["B", pb["pld"], tableprint(pb['env']), np.linalg.norm(np.dot(B, pb['env']) - pb["pld"] * pb['env'])/np.linalg.norm(pb['env']),
         pb["runtime"], "  "])
    x.add_row(
        ["C", pc["pld"], tableprint(pc['env']), np.linalg.norm(np.dot(C, pc['env']) - pc["pld"] * pc['env'])/np.linalg.norm(pc['env']),
         pc["runtime"], "  "])
    x.add_row(
        ["D", pd["pld"], tableprint(pd['env']), np.linalg.norm(np.dot(D, pd['env']) - pd["pld"] * pd['env'])/np.linalg.norm(pd['env']),
         pd["runtime"], "  "])
    x.add_row(
        ["E", pe["pld"], tableprint(pe['env']), np.linalg.norm(np.dot(E, pe['env']) - pe["pld"] * pe['env'])/np.linalg.norm(pe['env']),
         pe["runtime"], "  "])
    print(x)

    # 用jacobi方法求解特征值
    pa = jacobi_method(A, 1000, 1e-4)
    pb = jacobi_method(B, 1000, 1e-4)
    pc = jacobi_method(C, 1000, 1e-4)
    pd = jacobi_method(D, 1000, 1e-4)
    x = PrettyTable([" ", "Eigenvalue", "Runtime", "remarks"])
    x.align[" "] = "l"
    x.padding_width = 1
    x.add_row(["A", tableprint(pa['res']), pa['runtime'], "  "])
    x.add_row(["B", tableprint(pb['res']), pb['runtime'], "  "])
    x.add_row(["C", tableprint(pc['res']), pc['runtime'], "  "])
    x.add_row(["D", tableprint(pd['res']), pd['runtime'], "  "])
    print(x)

    print(tableprint(pa['res']))
    print(tableprint(pb['res']))
    print(tableprint(pc['res']))
    print(tableprint(pd['res']))


    # 用平移QR算法求解特征值
    pa = QR_method(A)
    pb = QR_method(B)
    pc = QR_method(C)
    pd = QR_method(D)
    pe = QR_method(E)
    x = PrettyTable([" ", "Eigenvalue", "Runtime", "remarks"])
    x.align[" "] = "l"
    x.padding_width = 1
    x.add_row(["A", tableprint(pa['res']), pa['runtime'], "  "])
    x.add_row(["B", tableprint(pb['res']), pb['runtime'], "  "])
    x.add_row(["C", tableprint(pc['res']), pc['runtime'], "  "])
    x.add_row(["D", tableprint(pd['res']), pd['runtime'], "  "])
    x.add_row(["E", tableprint(pe['res']), pe['runtime'], "  "])
    print(x)


    print(tableprint(pa['res']))
    print(tableprint(pb['res']))
    print(tableprint(pc['res']))
    print(tableprint(pd['res']))
    print(tableprint(pe['res']))

    # 计算方程的解
    print(" 使用 numpy 的函数求出的根为：")
    [px, pxe] = np.linalg.eig(polynomial)
    print(px)
    print(" 使用 平移QR算法 求出的根为：")
    px1 = QR_method(polynomial)
    px1 = px1["res"].reshape((11,))
    print(px1)
