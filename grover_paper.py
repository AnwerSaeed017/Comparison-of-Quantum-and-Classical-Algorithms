from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from math import ceil, log2, sqrt
import numpy as np
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import io



def create_oracle(n_bits: int, targets: list) -> QuantumCircuit:
    """Creates a quantum oracle for multiple target states."""
    oracle = QuantumCircuit(n_bits)
    
    # For each target state, mark it with a phase flip
    for target in targets:
        binary_repr = format(target, f"0{n_bits}b")
        # Apply X gates to bits that should be 0
        for qubit, bit in enumerate(reversed(binary_repr)):
            if bit == '0':
                oracle.x(qubit)
        
        # Multi-controlled Z gate implementation
        oracle.h(n_bits - 1)
        oracle.mcx(list(range(n_bits - 1)), n_bits - 1)
        oracle.h(n_bits - 1)
        
        # Restore the qubits
        for qubit, bit in enumerate(reversed(binary_repr)):
            if bit == '0':
                oracle.x(qubit)
    
    return oracle

def create_diffuser(n_bits: int) -> QuantumCircuit:
    """Creates the Grover diffuser operation."""
    diffuser = QuantumCircuit(n_bits)
    
    # Apply H gates to all qubits
    diffuser.h(range(n_bits))
    # Apply X gates to all qubits
    diffuser.x(range(n_bits))
    
    # Apply multi-controlled Z gate
    diffuser.h(n_bits - 1)
    diffuser.mcx(list(range(n_bits - 1)), n_bits - 1)
    diffuser.h(n_bits - 1)
    
    # Restore the states
    diffuser.x(range(n_bits))
    diffuser.h(range(n_bits))
    
    return diffuser


def visualize_circuit(circuit: QuantumCircuit) -> tuple:
    """
    Creates visualizations of the quantum circuit.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to visualize
    
    Returns:
        tuple: (text representation, matplotlib figure)
    """
    # Create text representation
    text_circuit = circuit_drawer(circuit, output='text')
    
    # Create matplotlib figure
    fig = plt.figure(figsize=(15, 8))
    circuit_drawer(circuit, output='mpl', style={'fontsize': 12})
    plt.tight_layout()
    
    return text_circuit, fig

def grovers_algorithm(N: int, targets: list, version='optimistic', shots=1024, visualize=False) -> tuple:
    """
    Implements Grover's algorithm for multiple targets with optimistic/pessimistic versions.
    
    Args:
        N (int): Size of the search space
        targets (list): List of target elements to search for
        version (str): 'optimistic' or 'pessimistic' version
        shots (int): Number of measurements to perform
        visualize (bool): Whether to generate circuit visualizations
    
    Returns:
        tuple: (counts dictionary, execution time, circuit depth, visualizations)
    """
    n_bits = ceil(log2(N))
    num_targets = len(targets)
    
    # Create main circuit
    qc = QuantumCircuit(n_bits, n_bits, name="Grover's Algorithm")
    
    # Initialize superposition
    qc.h(range(n_bits))
    
    # Calculate number of iterations
    if version == 'optimistic':
        num_iterations = int(sqrt(N / num_targets))
    elif version == 'pessimistic':
        num_iterations = int(0.58278 * sqrt(N / num_targets))
    else:
        raise ValueError("Invalid version. Use 'optimistic' or 'pessimistic'.")
    
    # Create oracle and diffuser
    oracle = create_oracle(n_bits, targets)
    diffuser = create_diffuser(n_bits)
    
    # Store visualizations if requested
    visualizations = {}
    
    # Apply Grover iterations
    for i in range(num_iterations):
        qc.append(oracle, range(n_bits))
        qc.append(diffuser, range(n_bits))
        if visualize and i == 0:  # Visualize first iteration
            visualizations['first_iteration'] = visualize_circuit(qc)
    
    # Measure all qubits
    qc.measure(range(n_bits), range(n_bits))
    
    # Generate complete circuit visualization
    if visualize:
        visualizations['complete_circuit'] = visualize_circuit(qc)
    
    # Simulate the circuit
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    result = simulator.run(transpiled_qc, shots=shots).result()
    
    counts = result.get_counts()
    execution_time = result.time_taken
    circuit_depth = transpiled_qc.depth()
    
    return counts, execution_time, circuit_depth, visualizations