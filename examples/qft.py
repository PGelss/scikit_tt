# -*- coding: utf-8 -*-

"""
This is an example of tensor-based quantum simulation. See [1]_ for details.
References
----------
.. [1] P. Gelß, S. Klus, Z. Shakibaei, S. Pokutta, "Low-rank tensor decompositions of 
       quantum circuits", arXiv:2205.09882, 2022
"""

import numpy as np
import time
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
import scikit_tt.tensor_train as tt
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import scikit_tt.quantum_computation as qc


def qft_qk(circuit, i, n):
    """Gate group G_i of QFT on n qubits"""
    circuit.h(i)
    for qubit in range(2,n-i+1):
        circuit.cp(2*np.pi/2**qubit, i+qubit-1, i)

def sampling_qk(qc, number_of_samples):
    """sampling in qiskit"""
    simulator = AerSimulator(method='matrix_product_state')
    counts = execute(qc, backend=simulator, shots=number_of_samples).result().get_counts(qc)
    probs=np.array(list(counts.values()))/number_of_samples
    return counts

utl.header(title='Quantum Fourier transform')

# parameters
dimensions = [16, 32]
number_of_samples = [100, 10000, 1000000]
repeats = 10
print('dimensions:', dimensions)
print('samples:   ', number_of_samples)
print('repeats:   ', repeats, '\n')

# sampling
cpu_times = np.zeros([2, len(dimensions), len(number_of_samples), repeats])
t = 0
start_time_global = utl.progress('Running Qiskit simulations ...', 0)
for count, n in enumerate(dimensions):
    for count2, m in enumerate(number_of_samples): 
        for k in range(repeats):
            start_time = time.time()
            init = np.random.randint(2, size=n)
            circuit = QuantumCircuit(n)
            for i in range(n):
                if init[i]==1:
                    circuit.x(i)
            for i in range(n):
                qft_qk(circuit,i,n)
                circuit.barrier()
            circuit.measure_all()
            sampling_qk(circuit, m)
            cpu_times[0,count,count2,k] = time.time()-start_time
            t += 1
            utl.progress('Running Qiskit simulations ...', 100*t/(len(dimensions)*len(number_of_samples)*repeats), cpu_time=time.time() - start_time_global)

t = 0
start_time_global = utl.progress('Running Scikit-TT simulations ...', 0)
for count, n in enumerate(dimensions):
    for count2, m in enumerate(number_of_samples): 
        for k in range(repeats):
            start_time = time.time()
            init = np.random.randint(2, size=n)
            G = mdl.qft(n)
            psi = tt.unit([2]*n,init)
            for i in range(n):
                psi = (G[i]@psi).ortho(threshold=1e-12)
            qc.sampling(psi, np.arange(n), m)
            cpu_times[1,count,count2,k] = time.time()-start_time
            t += 1
            utl.progress('Running Qiskit simulations ...', 100*t/(len(dimensions)*len(number_of_samples)*repeats), cpu_time=time.time() - start_time_global)

# print results
print('Qiskit: \n')
for i in range(len(number_of_samples)):
    string = ' '
    for j in range(len(dimensions)):
        string = string + str("%.2f" % np.mean(np.squeeze(cpu_times[0,j,i,:]))) + ' ± ' + str("%.2f" % np.std(cpu_times[0,j,i,:])) + '   '
    print(string)
print('\nScikit-TT: \n')
for i in range(len(number_of_samples)):
    string = ' '
    for j in range(len(dimensions)):
        string = string + str("%.2f" % np.mean(np.squeeze(cpu_times[1,j,i,:]))) + ' ± ' + str("%.2f" % np.std(cpu_times[1,j,i,:])) + '   '
    print(string)
print(' ')


