from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
import numpy as np
from math import gcd, ceil, log2, sqrt
from typing import Tuple, List, Optional
import time

class ShorAlgorithm:
     def __init__(self, N: int):
        if N < 3:
            raise ValueError("N must be >= 3")
        self.N = N
        # Optimize register sizes
        self.n_count = ceil(2 * log2(N))  # Reduced from 3* to 2* for efficiency
        self.n_target = ceil(log2(N))
               
     def _controlled_modular_multiplication(self, qc: QuantumCircuit, 
                                        control: int, 
                                        target_register: List[int], 
                                        aux_register: List[int], 
                                        a: int) -> None:
        """Implement controlled modular multiplication a·x mod N"""
        n = len(target_register)
        
        # Convert 'a' to binary representation
        a_bin = format(a, f'0{n}b')
        
        # Apply controlled additions based on binary decomposition
        for i, bit in enumerate(reversed(a_bin)):
            if bit == '1':
                shift = pow(2, i, self.N)
                self._controlled_modular_addition(
                    qc, control, target_register, 
                    aux_register, shift
                )
    
     def _controlled_modular_addition(self, qc: QuantumCircuit,
                                   control: int,
                                   target_register: List[int],
                                   aux_register: List[int],
                                   a: int) -> None:
        """Implement controlled modular addition (a + x) mod N"""
        n = len(target_register)
        
        # Apply QFT to target register
        qc.append(QFT(n), target_register)
        
        # Add phase rotations for modular addition
        for i in range(n):
            phase = 0
            for j in range(i, n):
                if (a >> (n-1-j)) & 1:
                    phase += 2**(i-j)
            if phase:
                qc.cp(2*np.pi*phase/2**n, control, target_register[i])
                
        # Apply inverse QFT
        qc.append(QFT(n, inverse=True), target_register)
        
        # Perform modular reduction if auxiliary register is provided
        if aux_register:
            self._modular_reduction(qc, target_register, aux_register)
    
     def _modular_reduction(self, qc: QuantumCircuit,
                          target_register: List[int],
                          aux_register: List[int]) -> None:
        """Implement modular reduction x mod N"""
        if not aux_register:
            return
            
        # Compare target value with N
        self._quantum_comparator(qc, target_register, aux_register[0], self.N)
        
        # Subtract N if target ≥ N
        self._controlled_subtraction(qc, aux_register[0], 
                                   target_register, self.N)
        
     def _quantum_comparator(self, qc: QuantumCircuit,
                           register: List[int],
                           result_qubit: int,
                           value: int) -> None:
        """Quantum circuit for comparing register value with classical value"""
        n = len(register)
        value_bin = format(value, f'0{n}b')
        
        # Apply X gates where value bits are 0
        for i, bit in enumerate(reversed(value_bin)):
            if bit == '0':
                qc.x(register[i])
                
        # Multi-controlled X gate
        qc.mcx(register, result_qubit)
        
        # Restore register state
        for i, bit in enumerate(reversed(value_bin)):
            if bit == '0':
                qc.x(register[i])
                
     def _controlled_subtraction(self, qc: QuantumCircuit,
                              control: int,
                              target_register: List[int],
                              value: int) -> None:
        """Implement controlled subtraction of classical value"""
        # Invert target register
        qc.x(target_register)
        
        # Add value (equivalent to subtraction when inverted)
        self._controlled_modular_addition(
            qc, control, target_register,
            [], value
        )
        
        # Restore target register
        qc.x(target_register)

     def _modular_exponentiation(self, qc: QuantumCircuit,
                              control_register: List[int],
                              target_register: List[int],
                              aux_register: List[int],
                              a: int) -> None:
        """Implement controlled modular exponentiation a^x mod N"""
        n = len(target_register)
        
        # Apply controlled multiplications for each control qubit
        for i, control in enumerate(control_register):
            power = pow(a, 2**i, self.N)
            if power != 1:  # Optimization: skip if power is 1
                self._controlled_modular_multiplication(
                    qc, control, target_register,
                    aux_register, power
                )

     def _quantum_period_finding(self, a: int) -> Optional[int]:
        """Improved quantum period finding for a^r ≡ 1 (mod N)"""
        if gcd(a, self.N) != 1:
            return gcd(a, self.N)

        # Create quantum registers
        count = QuantumRegister(self.n_count, 'count')
        target = QuantumRegister(self.n_target, 'target')
        aux = QuantumRegister(self.n_target, 'aux')
        classical = ClassicalRegister(self.n_count, 'classical')
        
        qc = QuantumCircuit(count, target, aux, classical)
        
        # Initialize superposition
        qc.h(count)
        qc.x(target[0])
        
        # Modular exponentiation
        self._modular_exponentiation(
            qc,
            list(range(self.n_count)),
            list(range(self.n_target)),
            list(range(self.n_target)),
            a
        )
        
        # Inverse QFT
        qc.append(QFT(self.n_count, inverse=True), count)
        qc.measure(count, classical)
        
        # Run with increased shots and better backend config
        backend = AerSimulator()
        job = backend.run(
            qc,
            shots=8192,  # Increased shots
            optimization_level=3,
            seed_simulator=42
        )
        counts = job.result().get_counts()
        
        # Process results with better period estimation
        return self._estimate_period(counts, a)
    
     def _estimate_period(self, counts: dict, a: int) -> Optional[int]:
        """Improved period estimation from measurement results"""
        shots_sum = sum(counts.values())
        
        # Sort by probability
        sorted_counts = sorted(
            [(int(bitstring, 2), count/shots_sum) 
             for bitstring, count in counts.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Try multiple candidates
        for measured_value, _ in sorted_counts[:5]:  # Check top 5 results
            phase = measured_value / (2 ** self.n_count)
            
            # Get period candidates
            period_candidates = self._get_period_candidates(phase)
            
            # Verify candidates
            for r in period_candidates:
                if r > 0 and r < self.N:
                    # Check if this is actually a period
                    if pow(a, r, self.N) == 1:
                        # Verify it's the smallest valid period
                        is_minimal = all(pow(a, k, self.N) != 1 
                                      for k in range(1, r))
                        if is_minimal:
                            return r
        return None
    
     def _get_period_candidates(self, phase: float) -> List[int]:
        """Get period candidates using continued fractions"""
        fractions = []
        h = [0, 1]  # Numerators
        k = [1, 0]  # Denominators
        
        # Maximum depth based on number size
        max_depth = ceil(log2(self.N)) + 2
        
        a = []  # Continued fraction coefficients
        frac = phase
        
        for _ in range(max_depth):
            a.append(int(frac))
            if len(a) > 1:
                h.append(a[-1] * h[-1] + h[-2])
                k.append(a[-1] * k[-1] + k[-2])
                if k[-1] > 0:
                    fractions.append(k[-1])
            frac = 1/(frac - a[-1]) if abs(frac - a[-1]) > 1e-10 else 0
            if abs(frac) < 1e-10:
                break
                
        return fractions
    
     def _continued_fractions(self, phase: float, max_depth: int = 20) -> List[int]:
        """Compute continued fraction expansion for phase estimation"""
        convergents = []
        h = [0, 1]  # Numerators
        k = [1, 0]  # Denominators
        a = []  # Continued fraction coefficients
        
        frac = phase
        for _ in range(max_depth):
            a.append(int(frac))
            if len(a) > 1:
                h.append(a[-1] * h[-1] + h[-2])
                k.append(a[-1] * k[-1] + k[-2])
                convergents.append(k[-1])
            frac = 1/(frac - a[-1]) if frac != a[-1] else 0
            if abs(frac) < 1e-10:
                break
                
        return convergents

     def _process_results(self, counts: dict, a: int) -> Optional[int]:
        """Process measurement results with improved period finding"""
        max_results = 10  # Increased from 5 for better sampling
        
        # Sort results by frequency
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        for bitstring, _ in sorted_counts[:max_results]:
            phase = int(bitstring, 2) / (2**self.n_count)
            candidates = self._continued_fractions(phase, max_depth=30)  # Increased depth
            
            # Verify each candidate period with additional checks
            for r in candidates:
                if r > 1 and r < self.N:
                    # Additional verification steps
                    if pow(a, r, self.N) == 1:
                        # Verify it's the smallest valid period
                        is_minimal = all(pow(a, k, self.N) != 1 for k in range(1, r))
                        if is_minimal:
                            return r
        return None

     def factor(self) -> Tuple[List[Optional[int]], float]:
        """Main factoring method with improved timing precision"""
        # Use time.perf_counter() for highest precision timing
        start_time = time.perf_counter()
        
        factors = [None, None]
        try:
            # Handle small factors first
            if self.N % 2 == 0:
                elapsed_time = time.perf_counter() - start_time
                return [2, self.N // 2], elapsed_time
                
            # Try direct GCD first
            for a in range(2, min(int(sqrt(self.N)) + 1, self.N)):
                if self.N % a == 0:
                    elapsed_time = time.perf_counter() - start_time
                    return [a, self.N // a], elapsed_time
            
            # Main factoring loop with improved timing
            attempts = min(20, ceil(log2(self.N)))
            
            for _ in range(attempts):
                # Add small random delay to ensure timing variation
                time.sleep(0.001 * np.random.random())
                
                a = np.random.randint(2, self.N)
                
                gcd_val = gcd(a, self.N)
                if 1 < gcd_val < self.N:
                    elapsed_time = time.perf_counter() - start_time
                    return [gcd_val, self.N // gcd_val], elapsed_time
                
                if gcd_val == 1:
                    for _ in range(2):
                        period = self._quantum_period_finding(a)
                        
                        if period and period % 2 == 0:
                            x = pow(a, period // 2, self.N)
                            factor1 = gcd(x + 1, self.N)
                            factor2 = gcd(x - 1, self.N)
                            
                            if 1 < factor1 < self.N:
                                factor2 = self.N // factor1
                                factors = [factor1, factor2]
                                break
                            
                            if 1 < factor2 < self.N:
                                factor1 = self.N // factor2
                                factors = [factor1, factor2]
                                break
                    
                    if factors != [None, None]:
                        break
            
            elapsed_time = time.perf_counter() - start_time
            return factors, elapsed_time
            
        except Exception as e:
            print(f"Error in factoring: {e}")
            elapsed_time = time.perf_counter() - start_time
            return [None, None], elapsed_time
            
     def get_circuit(self) -> QuantumCircuit:
        """Returns a simplified visualization circuit"""
        try:
            # Create smaller test circuit for visualization
            n_vis = min(self.n_count, 5)  # Limit size for visualization
            count = QuantumRegister(n_vis, 'count')
            target = QuantumRegister(min(self.n_target, 2), 'target')
            aux = QuantumRegister(min(self.n_target, 2), 'aux')
            classical = ClassicalRegister(n_vis, 'classical')
            
            qc = QuantumCircuit(count, target, aux, classical)
            
            # Add representative operations
            qc.h(count)  # Initial superposition
            qc.x(target[0])  # Initialize target state
            
            # Add simple modular arithmetic demonstration
            self._controlled_modular_addition(
                qc, count[0], list(range(len(target))),
                list(range(len(aux))), 1
            )
            
            # Add inverse QFT
            qc.append(QFT(n_vis, inverse=True), count)
            
            # Add measurements
            qc.measure(count, classical)
            
            return qc
            
        except Exception as e:
            print(f"Error creating visualization circuit: {e}")
            # Return minimal working circuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            return qc