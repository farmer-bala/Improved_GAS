#standard QAOA
#import networkx as nx
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
from scipy.interpolate import make_interp_spline
import numpy as np
import math
nqubits = 10
s_full = [(1,9),(2,8),(3,7),(4,6),(5,10),(2,9),(3,8),(4,7),(5,6),(1,10),(2,1),(3,9),(4,8),(5,7),(6,10),(3,1),(4,9),(5,8),(6,7),(2,10),
            (3,2),(4,1),(5,9),(6,8),(7,10),(4,2),(5,1),(6,9),(7,8),(3,10),(4,3),(5,2),(6,1),(7,9),(8,10),(5,3),(6,2),(7,1),(8,9),(4,10),
            (5,4),(6,3),(7,2),(8,1),(9,10)]

legal_string = ['0000000001','0000000010','0000000100','0000001000','0000010000','0000100000','0001000000','0010000000','0100000000','1000000000']

desire_vector = [0]*(2**nqubits)
#初态制备要注意
for i in range(nqubits):
    desire_vector[2**i] = 1/math.sqrt(nqubits)*complex(1,0)

iteration = 0
max_iter = 0
np.set_printoptions(threshold = np.inf)
Test = -1
lambda1 = 0.5
#第一问用传统QAOA算法beta对应的一项是传统泡泡利x门
'''
def nbit_Sfull():
    list1 = []
    for i in range(11):
        list1.append([])
    for i in range(1,11):
        for j in range(i,11):
            k = (i+j)%n
            list1[k].append((i,j))
    
    #对每一行K进行排序

'''
def H_F():
    #在量子电路图上定义beta符号
    gamma = Parameter("$\\gamma$")
    #构造4比特量子电路模块
    qc_f = QuantumCircuit(nqubits)
    
    for i in range(nqubits):
        qc_f.rz(2 * gamma,i)
    
    return qc_f

#可能是门电路不对导致其他比特也出现了
def H_fullmixer():
    #问题一只需要
    #在量子电路图上定义gama符号
    beta = Parameter("$\\beta$")
    #构造4比特量子电路模块
    qc_p = QuantumCircuit(nqubits)
    for pair in list(s_full):  # pairs of nodes
        qc_p.rxx(2 * beta, pair[0]-1, pair[1]-1)
        qc_p.ryy(2 * beta, pair[0]-1, pair[1]-1)
        #加上虚拟栅分割模块
    
    return qc_p

def Init_gate():

    qc_0 = QuantumCircuit(nqubits)
    qc_0.initialize(desire_vector,list(range(0,nqubits))) 

    #qc_0.gates_to_uncompute().inverse()
    return qc_0
#纠正counts中存在的不合理的地方比如有多个1
def abstract_counts(counts):

    n_counts = {}
    for bit_string in legal_string:

        n_counts[bit_string] = counts[bit_string]

    return n_counts 


def create_Acircuit(theta,reward_e):

    n_layers = len(theta)//2  
    beta = theta[:n_layers]
    gamma = theta[n_layers:]

    qc = QuantumCircuit(nqubits)
    r = 0.5
    # 先计算初始状态
    #qc.h(range(nqubits))
    qc.initialize(desire_vector,list(range(0,nqubits)))
    
    #print(".........",n_layers)
    
    #qc.gates_to_uncompute().inverse()
    #利用for循环不断叠加，有多少theta就设置多少层
    for layer_index in range(n_layers):
        # 问题酉矩阵运算
        for i in range(nqubits):
            #此处每一个gamma前面都要适配一个系数
            qc.rz(2 * lambda1 * reward_e[i] * gamma[layer_index], i)

        for pair in list(s_full):  # pairs of nodes
            qc.rxx(2 * beta[layer_index], pair[0]-1, pair[1]-1)
            qc.ryy(2 * beta[layer_index], pair[0]-1, pair[1]-1)

    #print("para:",theta)
    qc.measure_all()
    return qc

#全局变量存储tsv文件
def Credit_reward(solution,data):
    #t通过率h坏账率,data 是一维20列数据
    reward = 0

    #solution代表量子比特n位量子比特的值，经过QAOA算法每一位可能为0或者1代表被分到不同的组
    for index,value in enumerate(solution):

  
        #这里index代表某一张卡
        if value == '1':
            reward = -data[2*(9-index)] + (1+data[2*(9-index)])*data[2*(9-index)+1]
            break
    
    return reward


def get_expectation(data,reward_e,shots=512):

    backend = Aer.get_backend('qasm_simulator')
    #执行512次
    #backend.shots = shots
    #利用闭包特性封存上下文，这样调用minimize就不用传*args
    def execute_circ(theta):
        qc = create_Acircuit(theta,reward_e)
        counts = backend.run(qc, seed_simulator=10,
                             shots=shots).result().get_counts()
        
        #此处counts得处理
        
        counts = abstract_counts(counts)
        #print("counts:", counts)
        return compute_expectation(counts,data)

    return execute_circ

def compute_expectation(counts,data):
    #counts 多次改变beta，gama后的结果包含比特串和最后对应的值
    avg = 0
    sum_count = 0
    #这样计算count
    
    for bit_string,count in counts.items():
        #如果bit_string 等于那十个

        reward = Credit_reward(bit_string,data)
    
        avg += reward * count
        sum_count += count
        
    #print("total",avg/sum_count)
    return avg/sum_count
#读tsv文件
def count_n(theta_i):
    global iteration
    iteration = iteration + 1

def run(data,theta_0):
    global iteration
    global max_iter
    #已经处理好的data
    reward_e = []
    for s in legal_string:
        reward_e.append(-Credit_reward(s,data))

    expectation = get_expectation(data,reward_e)
    #[1.0,1.0]theta 和 gama
    
    res = minimize(expectation,theta_0,method='COBYLA',callback = count_n )

    if iteration > max_iter:
        max_iter = iteration
    iteration = 0


    backend = Aer.get_backend('qasm_simulator')
    #backend.shots = 512

    qc = create_Acircuit(res.x,reward_e)
    counts = backend.run(qc, seed_simulator=10,
                             shots=512).result().get_counts()
    #此处counts也得处理
    
    counts = abstract_counts(counts)
    return counts


def read_tsv(path):

    with open(path,encoding = "utf-8") as f:
        data = np.loadtxt(f,delimiter = ",",skiprows = 1)
    return data

def step1(data,theta_0):
    #先处理数据选出每列的行然后返回10个行的值
    index = []
    global Test
    for i in range(100):
        tmp = data[np.arange(0,10,1),2*i:2*i+2]
        tmp = tmp.flatten()

        Test = i
        counts = run(tmp,theta_0)
        
        #找出counts对应的最大bit的行值
        '''
        if i == 58:
            plot_histogram(counts)
            plt.show()
        '''
        index.append(find_max(counts))
    
    for i,value in enumerate(index):
        print("cardId = " + str(i+1) + " card阈值= " + str(value+1))

    return index,max_iter


def step2(data,index,theta_0):
#100对阈值做归并选出最大的一个
    index = np.repeat(index,2)
    #行值 列值
    n_data = data[index,np.arange(0,np.size(data,1),1)]
    res_index = []
    #长度为200的一维向量
    for i in range(10):
        tmp = n_data[20*i:20*i+20]
        counts = run(tmp,theta_0)
        t_index = find_max(counts)
        #返回的是相对位置所以得处理一下
        t_index = 2*t_index + 20*i
        res_index.append(t_index)
        res_index.append(t_index+1)

    e_data = n_data[res_index]

    counts = run(e_data,theta_0)
    

    #plot_histogram(counts)
    #plt.show()
    e_index = find_max(counts)

    #返回第几张卡
    return res_index[e_index*2]//2,max_iter

def find_max(counts):
    #按道理说满足条件的最大值应该出现在满足条件的初态里
    max_count = 0
    index = 0
    bs = '' 
    for bit_string,count in counts.items():
        if count > max_count:
            max_count = count
            bs = bit_string
    
    for i,val in enumerate(bs):
        if val != '0':
            index = 9-i   
    
    return index

def traditional_count(data):
    Max_reward = 0
    card_id = 0
    card_threhold = 0
    for i in range(10):
        for j in range(100):
            t_res = data[i][j*2]-(1+data[i][j*2])*data[i][j*2+1]
            if Max_reward < t_res:
                Max_reward = t_res
                card_id = j
                card_threhold = i

    return card_id,card_threhold,Max_reward

def draw(y,value1 = None,x = np.array([1,2,3,4,5,6,7,8])):
    plt.title("QAOA")
    plt.xlabel("p")
    plt.ylabel("f(X)")
    
    x_smooth = np.linspace(0, 9, 200)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    y_smooth = make_interp_spline(x, y)(x_smooth)
    plt.plot(x_smooth, y_smooth,color='blue', label='qaoa_max', linestyle='-.')

    if value1 != None: 
        x2 = range(0,9)
        y2 = [value1 for _ in x2]
        plt.plot(x2,y2,color = 'red',label='max')

    plt.legend()
    plt.show()

def qaoa(data,theta_0):
    index = step1(data,theta_0)
    card_id = step2(data,index,theta_0)[0]+1
    print("最终结果cardId = " + str(card_id),"card阈值 = " + str(index[card_id-1] + 1))


    value = data[index[card_id-1]][2*(card_id-1)] - (1+data[index[card_id-1]][2*(card_id-1)])*data[index[card_id-1]][(card_id-1)*2+1]
   
    return value

if __name__ == "__main__":
    
    q_circuit = QuantumCircuit(nqubits)
    H_op = Init_gate()
    hf_op = H_F()
    hmix_op = H_fullmixer()
    q_circuit.append(H_op, range(nqubits))
    q_circuit.append(hf_op, range(nqubits))
    q_circuit.append(hmix_op, range(nqubits))
    #默认用文本绘制但在vscode无法显示 所以用matplotlib绘图
    
    #q_circuit.draw('mpl')
    print(q_circuit)
    
    plt.show()  
    
    
    data = read_tsv(".\data_100.csv")
    card_id2,card_threhold,Max_reward = traditional_count(data)
    value1 = data[card_threhold][2*(card_id2)] - (1+data[card_threhold][2*(card_id2)])*data[card_threhold][(card_id2)*2+1]
    print("暴力搜索最终结果cardId = " + str(card_id2+1),"card阈值 = " + str(card_threhold+1))
    y = []
    
    
    for i in range(1,9):

        #theta_0 = np.ones(2*i)
        theta_0 = np.around(np.random.rand(2*i),1)
        print(theta_0)
        y.append(qaoa(data,theta_0))
    
    draw(y,value1)
    #1、打印最后结果cardid和阈值
    #2、打印中间结果每个step每次归并选出哪些卡
    #3、传统算法和量子算法对比，写出传统算法暴力求解
    #把QAOA和GAS结合起来的要点在于 先用QAOA跑出大致解然后划定阈值最后在用
    