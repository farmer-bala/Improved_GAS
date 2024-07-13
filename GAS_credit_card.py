
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.extensions import Initialize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.circuit.library import MCXGate, QFT
# distribution preparation
from scipy.stats import lognorm, norm, multivariate_normal

import Algorithm
from scipy.optimize import minimize
from scipy.special import comb, perm
import pandas as pd
import numpy as np
from itertools import combinations
import time
from mpl_toolkits.mplot3d import Axes3D
import credit_card as Cc

'''
def input_utils():
    #处理的是前面的系数这个 10个比特多少种组合
'''
total_r1 = []
total_r2 = []
y_1 = []
y_2 = []
y_3 = []
def draw_p(y1,y2):
    #插值必须是长度为2的二元组
    plt.title("Mix-alogrithm-creditcard,p=2")
    plt.xlabel("times")
    plt.ylabel("total_r")

    t = len(y1)
    x = range(1,t+1)
    #x_smooth = np.linspace(1, t+1, 200)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    #y_smooth = Cc.make_interp_spline(x, y1)(x_smooth)
    #plt.plot(x_smooth, y_smooth,color='red', label='QAOA_GAS',linestyle='-.')
    plt.plot(x, y1,color='red',marker='*',linestyle='-',label = 'QAOA_GAS')
    
    #x_smooth = np.linspace(1, t+1, 200)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    #y_smooth = Cc.make_interp_spline(x, y2)(x_smooth)
    #plt.plot(x_smooth, y_smooth,color='blue', label='GAS', linestyle='-')
    plt.plot(x, y2,color='blue',marker='v',linestyle='-.',label = 'GAS')

    plt.legend()
    plt.show()

def draw_v(y1,y2,y3):
    #插值必须是长度为2的二元组
    plt.title("Mix-alogrithm-creditcard,p=2")
    plt.xlabel("times")
    plt.ylabel("fx")

    t = len(y1)
    x = range(1,t+1)
    #x_smooth = np.linspace(1, t+1, 200)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    #y_smooth = Cc.make_interp_spline(x, y1)(x_smooth)
    #plt.plot(x_smooth, y_smooth,color='red', label='QAOA_GAS',linestyle='-.')
    plt.plot(x, y1,color='red',marker='*',linestyle='-',label = 'QAOA_GAS')
    
    #x_smooth = np.linspace(1, t+1, 200)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    #y_smooth = Cc.make_interp_spline(x, y2)(x_smooth)
    #plt.plot(x_smooth, y_smooth,color='blue', label='GAS', linestyle='-')
    plt.plot(x, y2,color='blue',marker='v',linestyle='-.',label = 'GAS')
    plt.plot(x, y2,color='yellow',marker='o',linestyle=':',label = 'QAOA')
    plt.legend()
    plt.show()

def matrix_generator(data,lAbmda,y,B,shots = 200,qaoa_on = 0):
    #每组数据 最后用map把数据找出来
    global total_r1,total_r2,y_1,y_2,y_3
    o_choice = np.zeros(data.shape[1]//2)
    y_init = 10
    for i in range(data.shape[1]//2):
        print("------------------------------------------")
        mu = np.zeros(data.shape[0])
        for j in range(data.shape[0]):

            mu[j] = data[j][2*i] - (1+data[j][2*i])*data[j][2*i+1]
            
        #coffcient 和另一个相加constrainti相加 y-lAbmda*B^2
        
        mu = (mu-0.6)/0.02*1.5
        
        
        mu = mu + np.full(data.shape[0],2*B*lAbmda)
        print(".....",y)
        print("muuuuuu",mu)
        #正则化
        

        sigma = np.full((data.shape[0],data.shape[0]),-2*lAbmda)
        row, col = np.diag_indices_from(sigma)
        sigma[row,col] = -lAbmda
        sigma = np.triu(sigma , k=0)

        print("sigmaaaaa:",sigma)

        loss_function = Loss_function(data[::,2*i:2*i+2])
        
        gas1 = GAS(data.shape[0],loss_function,mu,sigma)
        gas2 = GAS(data.shape[0],loss_function,mu,sigma)
        #gas1.show()
        
        opt_choice,_,total_R1,iter_time_depth1,y1 = gas1.solver(shots,y+lAbmda*B*B)
        opt_choice,_,total_R2,iter_time_depth2,y2 = gas2.solver(shots,y_init+lAbmda*B*B)
        #这里画图 value1 = none
        #怎么改参数
        #opt_choice 返回的是01000..类似的10位字符串 需要判断哪一位最大
        total_r1.append(iter_time_depth1)
        total_r2.append(iter_time_depth2)
        y_1.append(y1)#qaoa-gas
        y_2.append(y2)#gas
        y_3.append(y)#qaoa
        for index,ele in enumerate(opt_choice[0]):
            if ele == '1':
                o_choice[i] = index
        
        total_r1[-1] = total_r1[-1] + qaoa_on
        print("QAOA_GAS:",total_r1)
        print("GAS:",total_r2)
        print("qaoa_on:",qaoa_on)
        if(len(total_r1)>3):
            draw_p(total_r1,total_r2)
            draw_v(y_1,y_2,y_3)

    
    return o_choice
    
def Loss_function(data):
    #首先判断哪一位是1
    def loss_function(opt_choice):
        pos = 0
        for i in range(len(opt_choice[0])):
            if opt_choice[0][i] == '1':
                pos = i
                break
        return data[pos][0]-(1+data[pos][0])*data[pos][1]
    #只要算出值就好了也不用算出其他的

    return loss_function

class GAS:
    def __init__(self, n, loss_function, loss_mu, loss_sigma,  constraint=None, constraint_mu=None, constraint_sigma=None, eps=0.01):
        """

        :param n:
        :param loss_mu:损失函数一阶项
        :param loss_sigma: 损失函数二阶项
        :param constraint: 约束条件常数项，可、有多个，d*n 按[c1, c2]形式表示 没有就填0
        :param constraint_mu: 约束条件一阶项，可有多个, d*n*n,  [[c11, c12, ..,c1n], [c21,c22,...,c2n], ...]
        :param constraint_sigma: 约束条件二阶项，可有多个, d*n,  [[C1],[C2],..[Cd]]
        :param eps: GAS涉及到参数的整数化，eps是其精度，影响总体线路宽度
        """ 

        self.n = n
        self.loss_mu = loss_mu
        self.loss_sigma = loss_sigma
        self.constant_int = 0
        self.constraint = constraint
        self.constraint_mu = constraint_mu
        self.constraint_sigma = constraint_sigma
        self.loss_function = loss_function
        self.eps = eps
        
        #self.num_value

        self.loss_mu_int = np.zeros(self.loss_mu.shape)
        self.loss_sigma_int = np.zeros(self.loss_sigma.shape)
        if self.constraint is not None:
            self.constraint_int = 0
            self.constraint_mu_int = np.zeros(self.constraint_mu.shape)
            self.constraint_sigma_int = np.zeros(self.constraint_sigma.shape)
        
    def coefficients_trans(self,y):
        """
        Reconstruct the value function, since in GAS we load the value function in qubits register
        (Rather than in the phase angle as in QAOA), we need to integerize the coefficients in the
        loss function

        :param num_value: number of qubits needs for load the value function
        :return:
        """



        m = int(np.ceil(np.log2(1/self.eps)))
        
        def trans(mu, sigma, constant=None):
            
            if constant == None:
                mu_max = np.max(np.abs(mu))
                sigma_max = np.max(np.abs(sigma))
                max = np.max([mu_max, sigma_max])
                # print('max:', max)
                mu_int = np.round(mu/max * 2**m)
                sigma_int = np.round(sigma/max * 2**m)
                # print('mu_int:', mu_int)
                self.max = max
                self.multiplier = max * 2**m
                return mu_int, sigma_int
            else:
                mu_max = np.max(np.abs(mu))
                sigma_max = np.max(np.abs(sigma))
                constant_max = np.max(np.abs(constant))
                max = np.max([mu_max, sigma_max, constant_max])
                mu_int = np.round(mu / max * 2 ** m)
                sigma_int = np.round(sigma / max * 2 ** m)
                constant_int = np.round(constant / max * 2 ** m)
                self.max = max
                self.multiplier = max * 2**m
                
                return constant_int, mu_int, sigma_int



        self.constant_int,self.loss_mu_int, self.loss_sigma_int = trans(self.loss_mu,self.loss_sigma,constant = y)

        #合并

        if self.constraint is not None:
            '''
            if len(self.constraint) == 1:  # 只有一个约束条件
                self.constraint_int, self.constraint_mu_int, self.constraint_sigma_int \
                    = trans(self.constraint_mu, self.constraint_sigma, self.constraint)
            elif len(self.constraint) > 1:
            '''
                #self.constraint_int = np.zeros(self.constraint)
            
            for d in range(len(self.constraint)):
               
                self.constraint_int[d], self.constraint_mu_int[d, : ], self.constraint_sigma_int[d,:,:] \
                    = trans(self.constraint_mu[d, :], self.constraint_sigma[d,:,:],constant = self.constraint[d])

        # todo scale sqrt(n)
        scale1 = int(np.ceil(np.sqrt(self.n))) + 2
        print("scale1:", scale1)
        scale2 = int(np.ceil(np.log2(self.n ** 2 * self.max)))
        print("scale2:", scale2)
        # scale = self.n + 1
        scale = scale2
        #self.num_value = scale + m
        self.num_value = m+6

    def cal_theta(self, a):
        return a * 2 * np.pi / 2 ** self.num_value

    def ug_gate(self, a):
       
        qc = QuantumCircuit(self.num_value)
        theta = self.cal_theta(a)

        #把p门理解为论文中R（）这里有问题这里构造的时候没有反转
        for i in range(self.num_value):
            theta_i = 2 ** i * theta
            
            qc.p(theta_i, i)
        return qc.to_gate(label=f'Ug({theta}), a={a}')

    # def gas_ug_loss_function_gate(self):
    #     x = QuantumRegister(self.n, name='x')
    #     z = QuantumRegister(self.num_value, name='z')
    #     qc = QuantumCircuit()
    #     qc.add_register(x)
    #     qc.add_register(z)
    #     for qubit in range(qc.num_qubits):
    #         qc.h(qubit)
    #     for i in range(self.n):
    #         ug = self.ug_gate(self.loss_mu_int[i])
    #         qc.append(ug.control(1), [x[i] + z[:]])
    #     for i in range(self.n):
    #         for j in range(self.n):
    #             if i == j: # for x_i^2, since x_1 = 0, 1 we have x_i^2 = x_i
    #                 ug = self.ug_gate(self.loss_sigma_int[i, j])
    #                 qc.append(ug.control(1), [x[i]] + z[:])
    #             else:
    #                 ug = self.ug_gate(self.loss_sigma_int[i, j])
    #                 qc.append(ug.control(2), [x[i]]+[x[j]]+z[:])
    #     return qc.to_gate(label='Ug_loss_function')

    # def gas_ug_constraint_gate(self, first, second, label=' '):
    #     """
    #     The Ug gate of constraint, all constraints is given in the form of c(x)+y<0
    #     :param constraint:
    #     :param first: first order
    #     :param second: second order
    #     :param name: label of the constraint
    #     :return:
    #     """
    #     x = QuantumRegister(self.n, name='x')
    #     z = QuantumRegister(self.num_value, name='z')
    #     qc = QuantumCircuit()
    #     qc.add_register(x)
    #     qc.add_register(z)
    #     # for qubit in range(qc.num_qubits):
    #     #     qc.h(qubit)
    #     for i in range(self.n):
    #         ug = self.ug_gate(first[i])
    #         qc.append(ug.control(1), [x[i] + z[:]])
    #     for i in range(self.n):
    #         for j in range(self.n):
    #             if i == j: # for x_i^2, since x_1 = 0, 1 we have x_i^2 = x_i
    #                 ug = self.ug_gate(second[i, j])
    #                 qc.append(ug.control(1), [x[i]] + z[:])
    #             else:
    #                 ug = self.ug_gate(second[i, j])
    #                 qc.append(ug.control(2), [x[i]]+[x[j]]+z[:])
    #     return qc.to_gate(label=f'Ug_constraint{label}_function')

    def gas_calculate_gate(self, y, first, second, label):
        
        """
        The Gates that computes f(x)+y
        :param y:
        :param first: represent mu such as x1 x2
        :param second:represent sigma such as x1x2 x1x1
        :param label:
        :return:
        """
        #这里需要改变一下如果对应为0则不生成r门
        
        x = QuantumRegister(self.n, name='x')
        z = QuantumRegister(self.num_value, name='z')
        qc = QuantumCircuit()
        qc.add_register(x)
        qc.add_register(z)
        for qubit in range(self.num_value):
            qc.h(z[qubit])
        for i in range(self.n):
            
            #如果second是0则不用造门
            ug = self.ug_gate(first[i])
                #ug 有m条线路
            qc.append(ug.control(1), [x[i]] + z[:])

        for i in range(self.n):
            for j in range(self.n):
               
                if i == j:  # for x_i^2, since x_1 = 0, 1 we have x_i^2 = x_i
                    
                        
                    ug = self.ug_gate(second[i, j])
                    qc.append(ug.control(1), [x[i]] + z[:])
                else:
                    ug = self.ug_gate(second[i, j])
                    qc.append(ug.control(2), [x[i]] + [x[j]] + z[:])
        
        #这里出了问题
       
        qc.append(self.ug_gate(y), z[:])
        qc.append(QFT(self.num_value).inverse().to_gate(), z[:])
        return qc.to_gate(label=f'{label}_ug_gate:{y}')

    def gas_no_constraint_Ay_gate(self, y):
        x = QuantumRegister(self.n, name='x')
        z = QuantumRegister(self.num_value, name='z')
        qc = QuantumCircuit()
        qc.add_register(x)
        qc.add_register(z)
        # Prepare the superpostion of all possible outcomes
        for i in range(self.n):
            qc.h(x[i])
        qc.append(self.gas_calculate_gate(y, first=self.loss_mu_int, second=self.loss_sigma_int, label='Loss'), x[:]+z[:])
        return qc.to_gate(label=f'Ay, y={y}')
    #****************************** -------改
    def gas_constraint_Ay_gate(self, y):
        x = QuantumRegister(self.n, name='x')
        z = QuantumRegister(self.num_value, name='z')
        c = QuantumRegister(len(self.constraint) + 2)
        qc = QuantumCircuit()
        qc.add_register(x)
        qc.add_register(z)
        qc.add_register(c)
        # If we have constraints, the oracle O which is a CNOT gate is controlled by o[-1]
        # Therefore we need to apply (f(x)+y)^dagger and (c(x)+y)^dagger to x+z
        # Gate of [f(x)+y, cnot, (f(x)+y)^dagger]
      
        fy_gate = self.gas_calculate_gate(y=y, first=self.loss_mu_int, second=self.loss_sigma_int, label='Loss')
        qc.append(fy_gate, x[:]+z[:])
        qc.cnot(z[-1], c[0])
        qc.append(fy_gate.inverse(), x[:]+z[:])
        # Gate of [c(x)+c, cnot, (c(x)+c)^dagger]
        for d in range(len(self.constraint)):
            #可能又多个常数项代表多个约束
            
            cy_gate = self.gas_calculate_gate(y=self.constraint_int[d,0], first=self.constraint_mu_int[d, :],
                                              second=self.constraint_sigma_int[d, :, :], label=f'Constraint{d}')
            qc.append(cy_gate, x[:]+z[:])
            qc.cnot(z[-1], c[1+d])
            qc.append(cy_gate.inverse(), x[:]+z[:])
        qc.mcx(control_qubits=c[:-2], target_qubit=c[-1])
        return qc.to_gate(label=f'Ay, y={y}')

    def gas_Gy_gate(self, y):
        # Gy = AyDAy^daggerO
        x = QuantumRegister(self.n, name='x')
        z = QuantumRegister(self.num_value, name='z')
        o = QuantumRegister(1, name='o')
        qc = QuantumCircuit()
        qc.add_register(x)
        qc.add_register(z)
        qc.add_register(o)

        if self.constraint == None:
            # In this case, the oracle O, a CNOT gate is controlled on the last qubit z
            # We only need Ay

            # O
            # qc.x(z[-1])
            qc.cnot(z[-1], o[:])
            qc.z(o)
            qc.cnot(z[-1], o[:])
            # qc.x(z[-1])
            # Ay^dagger
            ay_gate = self.gas_no_constraint_Ay_gate(y)
            qc.append(ay_gate.inverse(), x[:]+z[:])
            # D
            d_gate = Algorithm.reflect_0_gate(nqubits=qc.num_qubits)
            qc.append(d_gate, x[:]+z[:]+o[:])
            #Ay
            qc.append(ay_gate, x[:]+z[:])
        else:
            # We need c register to restore the comparison results including loss and constraints
            # In this case, oracle O becomes a multi-controlled Z gate
            c = QuantumRegister(len(self.constraint)+2, name='c')
            #c不在线路中
            #O
            qc.add_register(c)
            qc.cnot(c[-1], o[:])
            qc.z(o)
            qc.cnot(c[-1], o[:])
            # Ay^dagger
            ay_gate = self.gas_constraint_Ay_gate(y)
            qc.append(ay_gate.inverse(), x[:]+z[:]+c[:])
            d_gate = Algorithm.reflect_0_gate(nqubits=qc.num_qubits)
            qc.append(d_gate, range(qc.num_qubits))
            # Ay
            qc.append(ay_gate, x[:]+z[:]+c[:])
        return qc.to_gate(label='G')

    def get_value(self, string):
        if int(string, 2) >= 2 ** (self.num_value - 1):
            opt_value = int(string, 2) - 2 ** self.num_value
        else:
            opt_value = int(string, 2)
        return opt_value

    def gas_step(self, r, y, shots):
        """
        One step of gas, A grover search corresponds to y
        :param y:
        # :param s: dicide the iteration number by r=np.floor(np.pi/4*np.sqrt(N/s))
        :param r: the iteration time of search
        :return:
        """
        qc = QuantumCircuit()
        x = QuantumRegister(self.n, name='x')
        z = QuantumRegister(self.num_value, name='z')
        qc.add_register(x)
        qc.add_register(z)

        #print("qc.numbit",qc.num_qubits)
        if self.constraint is not None:
            c = QuantumRegister(len(self.constraint) + 2, name='c')
            qc.add_register(c)
        o = QuantumRegister(1, name='o')
        qc.add_register(o)

        if self.constraint is not None:
            ay_gate = self.gas_constraint_Ay_gate(y)
        else:
            ay_gate = self.gas_no_constraint_Ay_gate(y)
        qc.append(ay_gate, range(qc.num_qubits-1))

        for _ in range(r):
            qc.append(self.gas_Gy_gate(y), range(qc.num_qubits))
        meas = ClassicalRegister(self.num_value + self.n)
        qc.add_register(meas)
        for qubit in range(self.n):
            qc.measure(x[qubit], meas[qubit])
        for qubit in range(self.num_value):
            qc.measure(z[qubit], meas[self.n + qubit])
        backend = Aer.get_backend('qasm_simulator')
        qc = transpile(qc, backend)

        print('shots:',shots)
        print("qc.width:",qc.width())
        print("qc.depth:",qc.depth())
        print("qc.numbit",qc.num_qubits)
        counts = backend.run(qc, shots=shots).result().get_counts()
        step_result = [(k[self.num_value:][::-1], self.get_value(k[:self.num_value])-y, v) for k, v in counts.items()]
        print("counts:",step_result)
        # 判断 
        l_counts = np.sort(list(counts.values()))

        max_prob = l_counts[-1]
        
        smax_prob = l_counts[-2]
        
        #opt= [k for k, v in counts.items() if v == max_prob][0]
        opt_orign = [k for k, v in counts.items() if v == max_prob]
        opt1 =[k for k, v in counts.items() if v == smax_prob][0]


        signal = '0'*self.n
        index = -1
        #
        for opt_ele in opt_orign:
            opt_ele = opt_ele[self.num_value:]
            index += 1
            if opt_ele != signal:
                break
        
        if opt_orign[index][self.num_value:] == signal:
            list1 = list(counts.values())
            arr_= np.argmax(list1)
            list1[arr_] = -list1[arr_]
            max_prob = np.max(list1)
            opt= [k for k, v in counts.items() if v == max_prob][0]
            

        else:
            opt = opt_orign[index]

        #print("max_prob------:",max_prob)
        # todo
        # opt_choice = opt[:self.n][::-1]
        # opt_value_bin = opt[self.n:]
        # 如果代表信用卡的后几位是0则舍去找第二大的
        opt_choice = opt[self.num_value:][::-1]
        opt_value_bin = opt[:self.num_value]

        opt_value_bin2 = opt1[:self.num_value]

        if int(opt_value_bin2, 2) >= 2 ** (self.num_value-1):
            opt_value2 = int(opt_value_bin2, 2) - 2 ** self.num_value
        else:
            opt_value2 = int(opt_value_bin2, 2)
        #翻得次数不对
        prob_is_max = False
        if max_prob/shots > 0.7:
            print('max_prob:', max_prob/shots)
            prob_is_max = True
        if int(opt_value_bin, 2) >= 2 ** (self.num_value-1):
            opt_value = int(opt_value_bin, 2) - 2 ** self.num_value
        else:
            opt_value = int(opt_value_bin, 2)

        return opt_choice, [opt_value-y,opt_value2-y,max_prob/shots,smax_prob/shots,qc.depth()], prob_is_max, counts

    def solver(self, shots,y_1,lam=1.1, max_iter=50):

        self.coefficients_trans(y_1)
        k = 1
        i = 1
        opt_change = [True, True]
        #
        opt_value_old = self.constant_int
        print("mmmmmmmmmmmmmmuuuuuuuuuuu:",self.loss_mu_int)
        print("sigmaaaaaaaaaaaaaaaaaaaaa:",self.loss_sigma_int)
        
        y = self.constant_int

        prob_is_max = False
        total_search_number = 0
        while i < max_iter and (opt_change[0] or opt_change[1]) and not prob_is_max:
            s = 2
            if np.ceil(k-1) == 0:
                r = 0
            else:
                r1 = np.random.randint(low=np.floor(k/2), high=np.ceil(k-1), size=1)
                r2 = np.floor(np.pi/4 * np.sqrt(2**self.n/s))
                r = int(np.min([r1, r2]))
            print('r:', r)
            print('i:', i)
            print('y:', y)
            #plt.ion()
            opt_choice, opt_values, prob_is_max, counts = self.gas_step(r=r, y=-y, shots=shots)
            #plt.clf()
            #plot_histogram(counts)
            #plt.pause(2)
            #plt.ioff()
            #plt.show()
            #print("12123",opt_choice)
            opt_value = opt_values[0]
            opt_value1 = opt_values[1]
            print(f'y:{y}, Choice: {opt_choice}, Value:{opt_value}')
            # if opt_value == opt_value_old:
            #     if not opt_change[0]:
            #         opt_change[1] = False
            #     else:
            #         opt_change[0] = False
            # else:
            #     opt_change[0] = True
            #     opt_change[1] = True
            k = lam * k
            # if i == 1:
            #     y = opt_value + 1
            #     opt_value_old = opt_value
            #
            # else:
            

            #更新最大值
            if opt_value_old < opt_value:
                y = opt_value 
                opt_value_old = opt_value
            else:
                y = opt_value_old 
                
            '''            
            #opt_value为什么小于0
            
            if opt_value_old < opt_value:
                y = opt_value_old + 1
            '''

            # if not prob_is_max:
            #     k = lam + k
            #     y = opt_value + 1
            i += 1
            total_search_number += r
            print(opt_values[2],opt_values[3])
            if(opt_value == opt_value1 and opt_values[2]+opt_values[3]>0.6):
                print("### MAXIMUM ITERATION!!! ###")
                break

            if i == max_iter:
                print("### MAXIMUM ITERATION!!! ###")


        #todo opt_value is minused by y in every iteration
        
        #每次运行都画一次线
        


        #plot_histogram(counts)
        #plt.show()
        max_value = self.loss_function(opt_choice)
        print(f"总搜索次数为{total_search_number}\t最优组合为:{opt_choice}\t最优值为{max_value}")
        #如果

        return opt_choice, opt_value, total_search_number,i*opt_values[-1],max_value
    
    def show(self):
        """画图用"""
        self.coefficients_trans()
       
        k = 1
        i = 1
        opt_change = [True, True]
        #
        opt_value_old = self.constant_int
        y = 1
        r = 2
        prob_is_max = False
        total_search_number = 0
        qc = QuantumCircuit()
        x = QuantumRegister(self.n, name='x')
        z = QuantumRegister(self.num_value, name='z')
    
        qc.add_register(x)
        qc.add_register(z)
        if self.constraint is not None:
            c = QuantumRegister(len(self.constraint) + 2, name='c')
            qc.add_register(c)
        o = QuantumRegister(1, name='o')
        qc.add_register(o)

        if self.constraint is not None:
            ay_gate = self.gas_constraint_Ay_gate(y)
        else:
            ay_gate = self.gas_no_constraint_Ay_gate(y)

        qc.append(ay_gate, range(qc.num_qubits - 1))

        for _ in range(r):
            qc.append(self.gas_Gy_gate(y), range(qc.num_qubits))
        meas = ClassicalRegister(self.num_value + self.n)
        qc.add_register(meas)
        for qubit in range(self.n):
            qc.measure(x[qubit], meas[qubit])
        qc.decompose().draw('mpl')
        plt.show()
#优化问题这么写没错，约束的可能因为有多个约束所以需要进行维度的统一
#没有constrait项就输入None 有的话即使常数项为0都要输入0

#gas1 = GAS(4,None,np.array([[0,2,1,1],[1,3,0,1],[1,0,3,0],[3,0,0,0]]),np.array([1,0,0,0]),np.array([[1]]),np.array([[[1,2,0,3],[1,2,0,3],[1,2,0,3],[1,2,0,3]]]),np.array([[1,0,0,0]]))
#print("111111111111111111111111")
#gas1.show()



def ten_bit_version(data,p):
    #先调用credit card 然后在 credit card 保存的结果中 GAS中loss代表fx 那么首先搞明白
    #1、先不考虑惩罚因子设置constraint 然后 设置两个constraint 一个是大于号 一个是小于号
    #2、得处理一下数据 fx 和 constraint满足输入要求
    #约束条件就是 x1+x2+....+x10 = 1
    #转换为(x1+x2+x3....+x10-1)^2
    #p就是t h就是h p-（1+p)h 最后矩阵类型转换成10*10
    index,iter1= Cc.step1(data,[1,1])
    card_id,iter2= Cc.step2(data,index,[1,1])
    card_id = card_id + 1 
    card_id2,card_threhold,Max_reward = Cc.traditional_count(data)
    max_iter = max(iter1,iter2)

    LA = 20
    B = 1
    choice = matrix_generator(data,LA,Max_reward,B,qaoa_on = p*max_iter)
    #100组数据


    data2 = data[choice.repeat(2),[x for x in range(0, 200)]]
    data2.reshape((10,-1))
    choice2 = matrix_generator(data2,LA,Max_reward,B,qaoa_on = p*max_iter)
    #最后剩十个数据
    data3 = data2[[choice2.repeat(2)],[x for x in range(0,20)]]
    data3.reshape((10,-1))


    choice3 = matrix_generator(data3,LA,Max_reward,B,qaoa_on = p*max_iter)
    #choice3就是最后的结果 是一个1X1数组
    pos_h = 2*choice3[0] #代表上个data的列 行藏在choice2里
    pos_h_r = 2*data.shape[1]*choice2[pos_h] + 2*pos_h#列 
    pos_h_c = choice[pos_h] #行
    pos_t_r = pos_h_r+1 
    print((pos_h_c,pos_t_r))
    #pos_h_c,pos_t_r的行列就是最终data选取的行列

def five_bit_solving(data,p):
    #先做上半部分 再做下半部分然后再合并
    #index = Cc.step1(data)
    #card_id = Cc.step2(data,index)+1
    #card_id2,card_threhold,Max_reward = Cc.traditional_count(data)
    #第一轮上半部分和下半部分 5x200列
    theta_tmp = np.around(np.random.rand(2*p),1)
    index,max_iter1 = Cc.step1(data,[1,1])
    card_id,max_iter2 = Cc.step2(data,index,[1,1])
    card_id = card_id + 1
    _,_,Max_reward = Cc.traditional_count(data)
    
    LA = 20
    B = 1
    max_iter = max(max_iter1,max_iter2)
    data1 = data[0:5,::]
    data1_ = data[5:10,::]

    choice = matrix_generator(data[0:5,::],LA,Max_reward,B,qaoa_on = p*max_iter)
    choice_ = matrix_generator(data[5:10,::],LA,Max_reward,B,qaoa_on = p*max_iter)

    #1x200
    data2 = data1[choice.repeat(2),[x for x in range(0, 200)]]
    data2_ = data1_[choice.repeat(2),[x for x in range(0, 200)]]
    
    #5x40
    data2.reshape((5,-1))
    data2_.reshape((5,-1))
    choice2 = matrix_generator(data2,10,Max_reward,B,qaoa_on = p*max_iter)
    choice2_ = matrix_generator(data2_,10,Max_reward,B,qaoa_on = p*max_iter)

    #1x40
    data3 = data2[choice2.repeat(2),[x for x in range(0, 40)]]
    data3_ = data2[choice2_.repeat(2),[x for x in range(0, 40)]]

    #5x8
    data3.reshape((5,-1))
    data3_.reshape((5,-1))
    choice3 = matrix_generator(data3,LA,Max_reward,B,qaoa_on = p*max_iter)
    choice3_ = matrix_generator(data3_,LA,Max_reward,B,qaoa_on = p*max_iter)

    #1x8 
    data4 = data3[choice3.repeat(2),[x for x in range(0, 8)]]
    data4_ = data3[choice3_.repeat(2),[x for x in range(0, 8)]]

    #4*2
    data4.reshape((4,-1))
    data4_.reshape((4,-1))
    choice4 = matrix_generator(data4,LA,Max_reward,B,qaoa_on = p*max_iter)
    choice4_ = matrix_generator(data4_,LA,Max_reward,B,qaoa_on = p*max_iter)

    #结束做转换
    
    #choice3就是最后的结果 是一个1X1数组
    #代表5x8的列
    pos_h_r = 2*choice4[0]
    pos_h_r_ = 2*choice4_[0] 
    #代表5x40的列 
    pos_h_r = data3.shape[1]*choice3[pos_h_r] + 2*pos_h_r#列
    pos_h_r_ = data3_.shape[1]*choice3_[pos_h_r] + 2*pos_h_r_#列 
    #代表5x200的列
    pos_h_r = data2.shape[1]*choice2[pos_h_r] + 2*pos_h_r#列
    pos_h_r_ = data2_.shape[1]*choice2_[pos_h_r_] + 2*pos_h_r_#列
    pos_h_c = choice[pos_h_r]
    pos_h_c_ = choice_[pos_h_r_]
    pos_t_r = pos_h_r+1
    pos_t_r_ = pos_h_r_+1 
    #print((pos_h_c,pos_t_r))


'''
gas1 = GAS(5,None,np.array([1,0,0,0,0]),np.array([[0,2,1,1,0],[1,3,0,1,1],[1,0,3,0,2],[3,0,0,0,3],[1,3,3,1,0]]),np.array([[1]]),np.array([[1,0,0,0,1]]),np.array([[[1,2,0,3,1],[1,2,0,3,1],[1,2,0,3,2],[1,2,0,3,1],[1,3,2,0,1]]]))
gas1.show()
gas1.gas_step(10,1,20)
#最后的答案是 最后5位倒过来 得思考我是不是在上面把顺序换了造成了这样的结果
'''
data = Cc.read_tsv(".\data_100.csv")
'''
mu = np.zeros(1000)
for i in range(data.shape[0]):
    for j in range(data.shape[1]//2):
        mu[i*100+j] = data[i][2*j] - (1+data[i][2*j])*data[i][2*j+1]
plt.plot(mu)
plt.show()
'''

#ten_bit_version(data,1)
five_bit_solving(data,1)


#对比试验
#1、对GAS初始量子态进行调整 
#2、计算结果中全为0的目标态占比很大可以在线路中增加惩罚项
#3、或者使用if语句剔除全0项
#4、QAOA改成退火机算法
