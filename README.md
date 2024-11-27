# Comparison-of-Quantum-and-Classical-Algorithms

This repository compares the performance of quantum and classical algorithms. Specifically, it benchmarks **Grover's Algorithm** vs. **Classical Linear Search** for unsorted search problems, and **Shor's Algorithm** vs. **General Number Field Sieve (GNFS)** for integer factorization tasks.

The project utilizes **Qiskit** for simulating quantum algorithms and **Streamlit** for an interactive web application that visualizes and analyzes the results.

## Algorithms Compared

### 1. **Grover's Algorithm vs. Classical Linear Search**
- **Grover's Algorithm**: A quantum search algorithm that provides a quadratic speedup for searching an unsorted database. This project implements Grover's algorithm for multiple target searches in a given search space.
- **Classical Linear Search**: A straightforward classical algorithm that searches through an unsorted list by checking each element until a match is found.

### 2. **Shor's Algorithm vs. General Number Field Sieve (GNFS)**
- **Shor's Algorithm**: A quantum algorithm for integer factorization, which can efficiently factor large integers in polynomial time. This project implements Shor's algorithm using Qiskit to factorize numbers and analyze its performance.
- **GNFS**: The classical **General Number Field Sieve** is one of the most efficient algorithms for factoring large integers. It is implemented in Python for comparison with Shor's algorithm.

## Features

- **Interactive Web Interface**: Built using **Streamlit**, it allows users to input numbers and compare the performance of quantum vs classical algorithms.
- **Visualization of Quantum Circuits**: Visualizes the quantum circuits used in Shor’s algorithm, including modular exponentiation, quantum Fourier transform (QFT), and measurements.
- **Speedup Analysis**: The application calculates and plots the speedup ratio of Shor’s algorithm compared to GNFS for a range of input sizes.
- **Circuit Depth & Qubit Analysis**: Estimates and visualizes the resource requirements (qubits and circuit depth) for Shor’s algorithm across different input sizes.
- **Period Finding Simulation**: Simulates the period finding step of Shor’s algorithm and visualizes the measurement outcomes.

## Getting Started

### Prerequisites

Before you run the project, make sure you have the following installed:

- **Python 3.8+**
- **Qiskit**: For quantum computing simulations.
- **Streamlit**: For the web interface.
- **Matplotlib**: For plotting graphs and visualizations.
- **NumPy**: For numerical computations.
- **SciPy** (optional): For advanced mathematical operations.

You can install the required dependencies using the following:

pip install qiskit streamlit matplotlib numpy scipy



**Running the project**

First clone the repository

Then navigate into project directory

Then on terminal run:

streamlit run app.py -- for comparison of grover and linear search

streamlit run compare.py --for comparison of shor and gnfs 
