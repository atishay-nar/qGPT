import pennylane as qml
import numpy as np

def zz_feature_map(x, wires, reps=1):
    n = len(wires)
    for _ in range(reps):
        # (a) H + U1(2·x_i) on each wire
        for i in range(n):
            qml.Hadamard(wires=wires[i])
            qml.PhaseShift(2 * x[i], wires=wires[i])
        # (b) full “controlled-phase” entanglement
        for i in range(n):
            for j in range(i + 1, n):
                θ = 2 * (np.pi - x[i]) * (np.pi - x[j])
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.PhaseShift(θ, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])