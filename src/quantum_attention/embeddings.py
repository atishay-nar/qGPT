import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt


# zz feature map. likely will go unused
def zz_feature_map(x, wires, reps=1):
    assert len(x) == len(wires), "x and wires must have the same length"
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

def circuit_14(weights, wires, reps=1):
    assert len(weights[0]) == len(wires), "single vector of weights and wires must have the same length"
    assert len(weights) == 4*reps, "there must be 4 weights vectors per repetition"
    n = len(wires)

    for rep in range(reps):
        # RY
        for i in range(n):
            qml.RY(weights[rep][i], wires=wires[i])

        # CRX 
        for i in range(n):
            qml.CRX(weights[rep + 1][i], wires=[wires[(n - i - 1) % n], wires[(n - i) % n]])

        # RY
        for i in range(n):
            qml.RY(weights[rep + 2][i], wires=wires[i])

        # CRX
        for i in range(n):
            qml.CRX(weights[rep + 3][i], wires=[wires[(n // 2 + i + 1) % n], wires[(n // 2 + i) % n]])

        
        


if __name__ == "__main__":
    # Example usage
    wires = [0, 1, 2, 3]
    weights = np.random.rand(4, len(wires)) * np.pi  # Random weights for testing
    dev = qml.device("default.qubit", wires=wires)

    # Create a quantum circuit using the circuit_14 function
    qml.QNode(circuit_14, dev, interface="autograd")(weights, wires)
    
    # Visualize the circuit
    qml.draw_mpl(circuit_14)(weights, wires)

    plt.show()  # Show the circuit diagram

