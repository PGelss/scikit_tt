# -*- coding: utf-8 -*-

"""
This is an example of tensor-based quantum simulation. See [1]_ for details.
References
----------
.. [1] P. Gelß, S. Klus, Z. Shakibaei, S. Pokutta, "Low-rank tensor decompositions of 
       quantum circuits", arXiv:2205.09882, 2022
"""

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from scikit_tt.tensor_train import TT
import scikit_tt.models as mdl
import scikit_tt.utils as utl

utl.header(title='Quantum full adder')

# construct quantum full adder using Qiskit
circuit = QuantumCircuit(4,2)
circuit.h(1)
circuit.h(2)
circuit.barrier()
circuit.ccx(1,2,3)
circuit.cx(1,2)
circuit.ccx(2,0,3)
circuit.cx(2,0)
circuit.cx(1,2)
circuit.barrier()
circuit.measure(0,0) # sum S
circuit.measure(3,1) # carry out C
circuit.draw(output='latex')
print(circuit)

# simulate circuit using "qasm simulator"
number_of_simulations = 100000
simulator = Aer.get_backend('qasm_simulator')
counts = execute(circuit, backend=simulator, shots=number_of_simulations).result().get_counts(circuit)
plot_histogram(counts, title='Qiskit')

### MPS/MPO approach

# construct QFA as MPO
full_adder = mdl.qfa()

# define initial quantum state (same as above for qiskit)
cores = [None]*4
cores[0]=np.zeros([1,2,1,1])
cores[0][0,:,0,0] = np.array([1,0])
cores[1]=np.zeros([1,2,1,1])
cores[1][0,:,0,0] = np.array([1,1])
cores[2]=np.zeros([1,2,1,1])
cores[2][0,:,0,0] = np.array([1,1])
cores[3]=np.zeros([1,2,1,1])
cores[3][0,:,0,0] = np.array([1,0])
qubits_in = 0.5*TT(cores)

# apply full adder and compute probabilities
qubits_out = full_adder@qubits_in
qubits_diag = TT.diag(qubits_out, [0,3])
probabilities = np.squeeze((qubits_diag.transpose(conjugate=True)@qubits_out).full())
inds = np.array(np.where(probabilities>1e-14)).T # indices of non-zero entries

# plot results
plt.figure()
m = inds.shape[0]
p = np.real([probabilities[inds[i,0], inds[i,1]] for i in range(m)])
tick_label = [np.array2string(inds[i,:], separator='')[-2:0:-1] for i in range(m)]
plt.bar(np.arange(m), p, tick_label=tick_label, zorder=2)
plt.ylim([0,0.7])
plt.ylabel('Probabilities')
plt.title('Scikit-TT')
plt.grid(axis='y', zorder=1)
for i, v in enumerate(p):
    plt.text(i-0.18, v+0.02, str('%.3f' %v))
plt.show()
