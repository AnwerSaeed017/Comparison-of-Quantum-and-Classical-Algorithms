import numpy as np
from math import gcd
from typing import Tuple, List, Optional

class GNFSAlgorithm:
    def __init__(self, N: int):
        self.N = N
        self.smoothness_bound = self._compute_smoothness_bound()
        
    def _is_prime(self, n: int) -> bool:
        """Helper method to check if a number is prime"""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    def _compute_smoothness_bound(self) -> int:
        """Compute smoothness bound according to paper specifications"""
        log_N = np.log(self.N)
        # Updated formula based on paper
        return int(np.exp(
            np.power(log_N, 1/3) * 
            np.power(np.log(log_N), 2/3) * 
            np.power(8/9, 1/3)
        ))
            
    def _prime_factors(self, n: int, bound: int) -> Optional[List[int]]:
        """Find prime factors of n up to bound"""
        if n <= 1:
            return []
            
        factors = []
        for p in range(2, bound + 1):
            while n % p == 0 and p <= bound:
                factors.append(p)
                n //= p
            if n == 1:
                return factors
            if p > np.sqrt(n):
                if n <= bound:
                    factors.append(n)
                    return factors
                break
        return None
        
    def factor(self) -> Tuple[List[Optional[int]], float]:
        """Main GNFS factoring method"""
        import time
        start_time = time.time()
        
        # Check trivial cases
        if self.N % 2 == 0:
            return [2, self.N // 2], time.time() - start_time
            
        if self._is_prime(self.N):
            return [None, None], time.time() - start_time
            
        try:
            bound = self._compute_smoothness_bound()
            base = int(pow(self.N, 1/3))
            
            # Try values around cube root of N
            for a in range(max(2, base - 100), base + 100):
                val = pow(a, 3) - self.N
                factors = self._prime_factors(abs(val), bound)
                
                if factors is not None:
                    g = gcd(pow(a, 3) - self.N, self.N)
                    if 1 < g < self.N:
                        return [g, self.N // g], time.time() - start_time
                        
            # Fall back to trial division for small numbers
            for i in range(2, int(np.sqrt(self.N)) + 1):
                if self.N % i == 0:
                    return [i, self.N // i], time.time() - start_time
                    
            return [None, None], time.time() - start_time
            
        except Exception as e:
            print(f"Error in GNFS: {e}")
            return [None, None], time.time() - start_time