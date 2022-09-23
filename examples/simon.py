# -*- coding: utf-8 -*-

"""
This is an example of tensor-based quantum simulation. See [1]_ for details.
References
----------
.. [1] P. Gelß, S. Klus, Z. Shakibaei, S. Pokutta, "Low-rank tensor decompositions of 
       quantum circuits", arXiv:2205.09882, 2022
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers.aer import AerSimulator
from scikit_tt.tensor_train import TT
import scikit_tt.models as mdl
import scikit_tt.utils as utl

def simon_qk(n):
    """Circuit construction for hidden bitstring b=1010."""

    q_reg1 = QuantumRegister (n,'reg1')
    q_reg2 = QuantumRegister (n,'reg2')
    c_reg = ClassicalRegister (n)
    circuit = QuantumCircuit (q_reg1, q_reg2, c_reg)
    circuit.h(q_reg1) # apply Hadamard gates on first register
    circuit.barrier()
    circuit.cx(q_reg1, q_reg2) # copy first register to second register
    circuit.barrier()
    circuit.cx(q_reg1[0],q_reg2[0]) # apply CNOT gates at positions where b is 1
    circuit.cx(q_reg1[0],q_reg2[2])
    circuit.barrier()
    circuit.measure(q_reg2,c_reg) # measure qubits of second register
    circuit.barrier()
    circuit.h(q_reg1) # apply Hadamard gates on first register
    circuit.barrier()
    circuit.measure(q_reg1, c_reg) # measure qubits of first register
    circuit.draw(output='latex')
    
    return circuit

def sampling_qk(qc, number_of_samples):
    """sampling in qiskit"""
    
    simulator = AerSimulator(method='matrix_product_state')
    counts = execute(qc, backend=simulator, shots=number_of_samples).result().get_counts(qc)
    probabilities=np.array(list(counts.values()))/number_of_samples
    return counts

utl.header(title='Simon\'s algorithm')

### Qiskit approach

# construct and simulate circuit in Qiskit
n = 4
circuit = simon_qk(n)
print(circuit)
counts = sampling_qk(circuit, 100000)

# convert measured bitstrings to matrix
A = np.ones([len(counts),n]) 
for i, btstr in enumerate(counts):
    for j in range(n):
        A[i,j] = np.asarray(btstr[j], dtype=int)
A = np.fliplr(A)

# find solution of z*b=0 for all measured bitstrings z
v = np.unravel_index(np.arange(2**n), [2]*n) 
v = np.array(v)
r = np.sum(np.abs(np.mod(A@v,2)),axis=0)
inds = np.where(r==0)[0]
solutions = v[:,inds].T
print(' ')
print('Result of Qiskit simulation:')
for i in range(solutions.shape[0]):
    print(str(i) + ') ' + str(solutions[i,:]))
print(' ')


### MPS/MPO approach

# construct final state in TT format
final_state = mdl.simon()

# compute probabilities
qubits_diag = TT.diag(final_state, [0,2,4,6])
probabilities = qubits_diag.transpose(conjugate=True)@final_state

# extract bitstrings
probabilities = np.squeeze(probabilities.full())
inds_bitstrings = np.argwhere(probabilities>1e-14)

# convert bitstrings with non-vanishing probability to matrix
A = np.array(inds_bitstrings) 

# find solution of z*b=0 for all measured bitstrings z
v = np.unravel_index(np.arange(2**n), [2]*n) 
v = np.array(v)
r = np.sum(np.abs(np.mod(A@v,2)),axis=0)
inds = np.where(r==0)[0]
solutions = v[:,inds].T
print('Result of MPO simulation')
for i in range(solutions.shape[0]):
    print(str(i) + ') ' + str(solutions[i,:]))
print(' ')
