from math import ceil, log2

def analyze_results(counts: dict, targets: list, N: int) -> dict:
    """
    Analyzes the results from Grover's algorithm.
    
    Args:
        counts (dict): Measurement counts from quantum circuit
        targets (list): Target elements that were searched for
        N (int): Size of search space
    
    Returns:
        dict: Analysis results including success rate and target distribution
    """
    total_shots = sum(counts.values())
    target_counts = 0
    
    # Convert targets to binary strings for comparison
    target_bins = [format(t, f"0{ceil(log2(N))}b") for t in targets]
    
    # Count successful measurements
    for state, count in counts.items():
        if state in target_bins:
            target_counts += count
    
    success_rate = target_counts / total_shots
    
    return {
        "success_rate": success_rate,
        "total_measurements": total_shots,
        "target_hits": target_counts,
        "distribution": counts
    }