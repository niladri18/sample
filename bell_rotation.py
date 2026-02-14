import os, sys, math
import numpy as np
import argparse
from scipy.sparse import kron, identity
import matplotlib.pyplot as plt
from scipy.linalg import expm
#from bell_rotation import *

import pdb

from scipy.linalg import expm  # optional check
#from qiskit import *
#from qiskit import Aer
import sys
import os
from tqdm import tqdm
#from qiskit.circuit.library import DraperQFTAdder
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
#from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
#from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
#rom qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.library import QFT
from qiskit.circuit.library import IntegerComparator
from qiskit.circuit.library.standard_gates import HGate, XGate, YGate, ZGate,   UGate, RZGate,      RXGate, RYGate
from qiskit.quantum_info import Statevector, SparsePauliOp, Pauli
from qiskit.quantum_info import Operator
from qiskit_aer import Aer
#from qiskit_aer import AerSimulator
#import mthree
import pdb
import numpy as np
import matplotlib.pyplot as plt
#from qiskit.primitives import StatevectorEstimator, StatevectorSampler
#from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import json
'''
Author: Niladri Gomes
This code implements
e^{-iSt} using Bell decomposition
where
S =e^{+i \lambda} |a><b| + e^{-i \lambda}|b><a|
tests for lambda = 0 and lambda = pi/2 is included
'''

def prepare_superposition(n, a, b, phase):

    '''
    Assumes a < b
    '''

    # Convert a and b to binary strings
    bin_a = format(a, f'0{n}b')
    bin_b = format(b, f'0{n}b')

    print(f"|a> = |{bin_a}> , |b> = |{bin_b}>")
    
    # get rid of the redundant qubits: significant qubits that have same value 
    idx = 0
    redundant_qubits = [] #the qubit indices
    equal_bits = [] # state of the ubit indices

    for i in range(n):
        #print(bin_a[i]=="0" and bin_b[i]=="0")
        if bin_a[i] != bin_b[i]:
            break
        if bin_a[i]== bin_b[i]:
            redundant_qubits.append(i)
            equal_bits.append(bin_a[i])
            idx = i+1


            #print(bin_a[i]=="0" and bin_b[i]=="0")

    #print(f"first index:{idx}")
    bin_a = bin_a[idx:]
    bin_b = bin_b[idx:]
    if len(bin_a) != len(bin_b):
        print("Two kets cannot be of different size!")

    print(f"|a> = |{bin_a}> , |b> = |{bin_b}>")
    n_trunc = len(bin_a)

    qc = QuantumCircuit(n_trunc)

    for qubit in range(0, n_trunc-1):
        qc.x(qubit)

    #prepare |a>
    for k,b in enumerate(bin_a):
        if b=='1':
            qc.x(n_trunc-k-1)

    # find the difference qubits
    diff_positions = [(n_trunc-i-1) for i in range(n_trunc) if bin_a[i] != bin_b[i]]
    #print("difference:")
    #print(diff_positions)

    first = diff_positions[0]
    #apply hadamard
    qc.h(first)
    qc.p(phase,first)

    #apply the CNOTS
    for k in diff_positions[1:]:
        qc.cx(first,k)


    #print(qc)
    return n_trunc, redundant_qubits, equal_bits, qc


def build_exp_iSt(n, a, b, theta, phase):
       
    circ = QuantumCircuit(n)

    n_trunc, redundant_qubits, equal_bits, b_transform = prepare_superposition(n,a,b,phase)
    qc = QuantumCircuit(n_trunc)
    #print(f"Num trunc qubits: {n_trunc}")
    #for k in range(n-1):
    #    qc.x(k)
    #print(b_transform)


    qc.append(b_transform.inverse(),[i for i in range(n_trunc)] )
    gate = RZGate(-theta).control(n_trunc-1)
    qc.append(gate, [i for i in range(n_trunc)] )
    qc.append(b_transform,[i for i in range(n_trunc)])
    
    if n_trunc == n:
        circ.append(qc, [i for i in range(n)])
    else:

        n_red = len(redundant_qubits)
        redundant_qubits = [n-i-1 for i in redundant_qubits]
        #print("redundant")
        #print(redundant_qubits)
        gate = qc.decompose().to_gate().control(num_ctrl_qubits=n_red, ctrl_state="".               join(equal_bits))
        circ.append( gate, redundant_qubits + [i for i in range(n_trunc)] )


    #print(circ.decompose())
    return circ


if __name__=="__main__":

    n = 5
    h = 1.0

    dt = 1e-12
    eps = 8.85419e-12 
    mu = 1.25664e-6
    H_coef = 1.0/(h*mu)
    E_coef = 1.0/(h*eps)
    theta = (H_coef - E_coef)*dt

    for a in range(2**n):
        for b in range(2**n):
            if a == b:
                continue
            A = np.zeros((2**n,2**n),dtype=complex)


            # test for real
            A[a,b] = theta
            A[b,a] = theta
            phase = 0

            # test for complex
            cmplx = True
            if cmplx:
                A[a,b] = 1j*theta
                A[b,a] = -1j*theta
                phase = np.pi/2

                theta*= -1


            
            #test pauli basis
            op = SparsePauliOp.from_operator(A)
            #print(op)
            evo = PauliEvolutionGate(op, time=1.0)

            U2 = expm(0.5j* A)

            circ = QuantumCircuit(n)

            if a > b:

                n_trunc, redundant_qubits, equal_bits, b_transform = prepare_superposition(n,b,a,phase =-phase)
            else:
                n_trunc, redundant_qubits, equal_bits, b_transform = prepare_superposition(n,a,b,phase = phase)

            #print("b_transform constructed")

            qc = QuantumCircuit(n_trunc)
            print(f"Num trunc qubits: {n_trunc}")
            #print(b_transform)


            circ.append(b_transform.inverse(),[i for i in range(n_trunc)] )
            #print("appended!")
            if a > b and phase != 0:
                if n_trunc > 1:
                    gate = RZGate(-theta).control(n_trunc-1)
                else:
                    gate = RZGate(-theta)
            else:
                if n_trunc > 1:
                    gate = RZGate(-theta).control(n_trunc-1)
                else:
                    gate = RZGate(-theta)


            qc.append(gate, [i for i in range(n_trunc)] )
            #qc.append(b_transform,[i for i in range(n_trunc)])
            if n_trunc == n:
                circ.append(qc, [i for i in range(n)])
            else:

                n_red = len(redundant_qubits)
                redundant_qubits = [n-i-1 for i in redundant_qubits]
            #print("redundant")
        #print(redundant_qubits)
                gate = qc.decompose().to_gate().control(num_ctrl_qubits=n_red, ctrl_state="".join(reversed(equal_bits)))
                circ.append( gate, redundant_qubits + [i for i in range(n_trunc)] )

    
            circ.append(b_transform,[i for i in range(n_trunc)])
            print(circ)

            U1 = Operator(circ).data
            U3 = Operator(evo).data
            #print(np.round(U1,2))
            #print(np.round(U2,2))
            diff = np.linalg.norm(U1 - U2)
            diff2 = np.linalg.norm(U3 - U2)
            if abs(diff) > 1e-10:
                print("Circuit operator doesn't match the real matrix")
                sys.exit(1)
            print("||U1-U2|| =", diff)
            #print("||U3-U2|| =", diff2)

    print("All test passed!")
