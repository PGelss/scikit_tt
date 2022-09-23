# -*- coding: utf-8 -*-

"""
This is an example of tensor-based quantum simulation. See [1]_ for details.
References
----------
.. [1] P. Gelß, S. Klus, Z. Shakibaei, S. Pokutta, "Low-rank tensor decompositions of 
       quantum circuits", arXiv:2205.09882, 2022
"""

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from fractions import Fraction
from scikit_tt.tensor_train import TT
import scikit_tt.models as mdl
import scikit_tt.utils as utl

def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2,4,7,8,11,13]:
        raise ValueError("'a' must be 2,4,7,8,11 or 13")
    U = QuantumCircuit(4)        
    for iteration in range(power):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a in [4, 11]:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "iQFT"
    return qc

def shor_qk(a,n,m):
    """circuit construction for 7^x mod 15 in Qiskit"""
    
    circuit = QuantumCircuit(n + m, n)
    for q in range(n): # apply Hadamard gates on first register
        circuit.h(q)
    circuit.barrier()
    circuit.x(3+n) # apply U_f 
    for q in range(n):
        circuit.append(c_amod15(a, 2**q), [q] + [i+n for i in range(4)])
    circuit.barrier()
    circuit.append(qft_dagger(n), [i for i in range(n)]) # apply inverse QFT
    circuit.barrier()
    circuit.measure(range(n), range(n)) # measure first register
    
    return circuit

def sampling_qk(qc, number_of_samples):
    """sampling in qiskit"""
    
    simulator = AerSimulator(method='matrix_product_state')
    counts = execute(qc, backend=simulator, shots=number_of_samples).result().get_counts(qc)
    probs=np.array(list(counts.values()))/number_of_samples
    
    return counts
    
utl.header(title='Shor\'s algorithm')

# parameters
M = 15 # number to factorize
a = 7 # base
n = 8 # number of qubits in first register
m = 4 # number of qubits in second register

### Qiskit approach

# construct circuit in Qiskit
circuit = shor_qk(a,n,m)

# simulate circuit
number_of_samples = 10000
counts = sampling_qk(circuit, number_of_samples)
prob = []
for i in counts:
    prob.append(counts[i]/number_of_samples)

# find order
A = np.ones([len(counts),n], dtype=int) # convert measured bitstrings to matrix
for i, btstr in enumerate(counts):
    for j in range(n):
        A[i,j] = np.asarray(btstr[j], dtype=int)
inds = A
ints = inds@np.array([128,64,32,16,8,4,2,1])
phase = ints/2**n
candidates = np.zeros(len(phase), dtype=int)
for i in range(len(phase)):
    candidates[i] = Fraction(phase[i]).limit_denominator(15).denominator

print('Qiskit:\n')
print('bitstring -  probability - integer - phase - order')
print('--------------------------------------------------')
for i in range(len(ints)):
    print(inds[i,:], ' - probability: ', prob[i])
    print((1+2*n)*' ',' - integer    : ', ints[i])
    print((1+2*n)*' ',' - quotient   : ', phase[i])
    print((1+2*n)*' ',' - pot. period: ', candidates[i])
    if candidates[i] % 2 == 0:
        if np.mod(7**(candidates[i]/2),15) != -1:
            print((1+2*n)*' ',' - factors    : ', [np.gcd(int(7**(candidates[i]/2)-1), 15),np.gcd(int(7**(candidates[i]/2)+1), 15)])
print(' ')


### MPS/MPO approach

# initial state (after Hadamard gates)
cores = [None]*12
for i in range(8):
    cores[i] = np.ones([1,2,1,1], dtype=complex)
for i in range(8,12):
    cores[i] = np.zeros([1,2,1,1], dtype=complex)
    cores[i][0,:,0,0] = [1,0]
psi_init = (1/16)*TT(cores)
    
# compute state after application of oracle
psi = mdl.shor(a)@psi_init

# apply inverse QFT to compute final state
G = mdl.iqft(n)
id_core = np.zeros([1,2,2,1], dtype=complex)
id_core[0,:,:,0] = np.eye(2)
for i in range(n):
    cores = G[i].cores.copy()
    G[i] = TT(cores + [id_core.copy() for _ in range(m)])
for i in range(n):
    psi = (G[i]@psi).ortho(threshold=1e-12)

# compute probability tensor
psi_diag = TT.diag(psi, np.arange(n))
probabilities = psi_diag.transpose(conjugate=True)@psi
probabilities = np.squeeze(probabilities.full())

# find order
inds = np.argwhere(probabilities>1e-14)
ints = inds@np.array([1,2,4,8,16,32,64,128])
phase = ints/2**n
candidates = np.zeros(len(phase), dtype=int)
for i in range(len(phase)):
    candidates[i] = Fraction(phase[i]).limit_denominator(15).denominator

print('Scikit-TT:\n')
print('bitstring -  probability - integer - phase - order')
print('--------------------------------------------------')
for i in range(len(ints)):
    print(inds[i,::-1], ' - probability: ', np.real(probabilities[inds[i,0],inds[i,1],inds[i,2],inds[i,3],inds[i,4],inds[i,5],inds[i,6],inds[i,7]]))
    print((1+2*n)*' ',' - integer    : ', ints[i])
    print((1+2*n)*' ',' - quotient   : ', phase[i])
    print((1+2*n)*' ',' - pot. period: ', candidates[i])
    if candidates[i] % 2 == 0:
        if np.mod(a**(candidates[i]/2),M) != -1:
            print((1+2*n)*' ',' - factors    : ', [np.gcd(int(a**(candidates[i]/2)-1), M),np.gcd(int(a**(candidates[i]/2)+1), M)])
print(' ')
