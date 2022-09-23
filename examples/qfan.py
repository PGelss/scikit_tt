# -*- coding: utf-8 -*-

"""
This is an example of tensor-based quantum simulation. See [1]_ for details.
References
----------
.. [1] P. Gelß, S. Klus, Z. Shakibaei, S. Pokutta, "Low-rank tensor decompositions of 
       quantum circuits", arXiv:2205.09882, 2022
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.providers.aer import AerSimulator
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import time
from scikit_tt.tensor_train import TT
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import scikit_tt.quantum_computation as qc

def qfa():
    """single QFA"""
    U = QuantumCircuit(4)
    U.ccx(1,2,3)
    U.cx(1,2)
    U.ccx(0,2,3)
    U.cx(2,0)
    U.cx(1,2)
    U = U.to_gate()
    U.name = "QFA"
    return U

def construct_qk(number_of_adders):
    """QFA network"""
    A = [None]*number_of_adders
    B = [None]*number_of_adders
    Z = [None]*number_of_adders
    S = [None]*number_of_adders
    C_in = QuantumRegister(1, name='c_{in}')
    for i in range(number_of_adders):
        A[i] = QuantumRegister(1, name='a_' + str(i))
        B[i] = QuantumRegister(1, name='b_' + str(i))
        Z[i] = QuantumRegister(1, name='z_' + str(i))
        S[i] = ClassicalRegister(1, name='s_' + str(i))
    C_out = ClassicalRegister(1, name='c_{out}')
    qc = QuantumCircuit(C_in)
    for i in range(number_of_adders):
        qc.add_register(A[i])
        qc.add_register(B[i])
        qc.add_register(Z[i])
    for i in range(number_of_adders):
        qc.add_register(S[i])
    qc.add_register(C_out)
    for i in range(number_of_adders):
        qc.h(A[i])
        qc.h(B[i])
    qc.barrier()    
    for i in range(number_of_adders):
        qc.append(qfa(), [j+3*i for j in range(4)])
        qc.barrier()
    qc.measure(C_in,S[0])
    for i in range(1,number_of_adders):
        qc.measure(Z[i-1],S[i])
    qc.measure(Z[-1],C_out)
        
    return qc

def sampling_qk(qc, number_of_samples, plot_tf=False):
    """sampling in qiskit"""
    
    cpu_time = time.time() 
    simulator = AerSimulator(method='statevector')
    counts = execute(qc, backend=simulator, shots=number_of_samples).result().get_counts(qc)
    cpu_time = time.time()-cpu_time
    probs=np.array(list(counts.values()))/number_of_samples

    return cpu_time

def sampling_qk_mps(qc, number_of_samples, plot_tf=False):
    """sampling in qiskit"""
    
    cpu_time = time.time() 
    simulator = AerSimulator(method='matrix_product_state')
    counts = execute(qc, backend=simulator, shots=number_of_samples).result().get_counts(qc)
    cpu_time = time.time()-cpu_time
    probs=np.array(list(counts.values()))/number_of_samples
    
    return cpu_time

def initial_state_tt(number_of_adders):
    """define initial quantum state (same as above for qiskit)"""
    
    cores = [None]*(3*number_of_adders+1)
    cores[0]=np.zeros([1,2,1,1])
    cores[0][0,:,0,0] = np.array([1,0])
    for i in range(number_of_adders):
        cores[1+3*i]=np.zeros([1,2,1,1])
        cores[1+3*i][0,:,0,0] = [np.sqrt(0.5), np.sqrt(0.5)]
        cores[2+3*i]=np.zeros([1,2,1,1])
        cores[2+3*i][0,:,0,0] = [np.sqrt(0.5), np.sqrt(0.5)]
        cores[3+3*i]=np.zeros([1,2,1,1])
        cores[3+3*i][0,:,0,0] = np.array([1,0])
    initial_state = TT(cores)
    
    return initial_state

utl.header(title='Quantum full adder network')

# parameters
number_of_samples = 1000
max_adders = 8
repeats = 10

# simulation in Qiskit and MPS format
cpu_times = np.zeros([3, max_adders])
k = 0
start_time_global = utl.progress('QFAN simulation with Qiskit and Scikit-TT', 0)
for number_of_adders in range(1, max_adders+1):
    measure = [3*i for i in range(number_of_adders+1)]
    cpu_qk_mps = 0
    cpu_qk=0
    cpu_tt=0   
    for r in range(repeats):
        start_time = time.time()
        qubits_qk = construct_qk(number_of_adders)
        sampling_qk(qubits_qk, number_of_samples)
        cpu_qk = cpu_qk + time.time() - start_time
        k += 1
    for r in range(repeats):
        start_time = time.time()
        qubits_qk = construct_qk(number_of_adders)
        sampling_qk_mps(qubits_qk, number_of_samples)
        cpu_qk_mps = cpu_qk_mps + time.time() - start_time
        k += 1
    for r in range(repeats):
        start_time = time.time()
        qfan_mpo = mdl.qfan(number_of_adders)
        initial_state = initial_state_tt(number_of_adders)
        final_state = (qfan_mpo@initial_state).ortho_right()
        measure_list = list(np.arange(0,final_state.order,3))
        qc.sampling(final_state, measure_list, number_of_samples)
        cpu_tt = cpu_tt + time.time() - start_time
        k += 1
    cpu_times[0,number_of_adders-1] = cpu_qk/repeats
    cpu_times[1,number_of_adders-1] = cpu_qk_mps/repeats
    cpu_times[2,number_of_adders-1] = cpu_tt/repeats
    utl.progress('QFAN simulation with Qiskit and Scikit-TT', 100*k/(3*repeats*max_adders), cpu_time=time.time() - start_time_global)

# plot runtimes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size'  : 20})
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(np.arange(1,max_adders+1), cpu_times.T, marker = 'o')
plt.xlabel('\\textsf{number of full adders}', fontsize=20)
plt.ylabel('\\textsf{computation time (s)}', fontsize=20)
plt.legend(['\\textsf{Qiskit (SV)}','\\textsf{Qiskit (MPS)}','\\textsf{Scikit-TT}'], fontsize=16)
plt.tight_layout()
plt.show()
