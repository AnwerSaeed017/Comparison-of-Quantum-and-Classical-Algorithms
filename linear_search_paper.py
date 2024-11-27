def classical_unsorted_search(arr: list, targets: list) -> tuple:
    """
    Implements classical linear search for multiple targets.
    
    Args:
        arr (list): Search space
        targets (list): Target elements to find
    
    Returns:
        tuple: (list of found indices, number of oracle calls)
    """
    results = []
    oracle_calls = 0
    
    for target in targets:
        found = False
        for i, value in enumerate(arr):
            oracle_calls += 1
            if value == target:
                results.append(i)
                found = True
                break
        if not found:
            results.append(None)
    
    return results, oracle_calls