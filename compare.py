import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from math import gcd,ceil,log2,sqrt
from shor import ShorAlgorithm
from gnfs_factorization import GNFSAlgorithm
import time

def plot_quantum_circuit(circuit,N):
    """Create an enhanced matplotlib visualization of Shor's algorithm quantum circuit"""
    # Calculate register sizes based on N
    n_count = ceil(2 * log2(N))    # Phase estimation register
    n_target = ceil(log2(N))       # Target register
    n_aux = n_target               # Auxiliary register for modular arithmetic
    
    total_qubits = n_count + n_target + n_aux
    
    # Create figure with better proportions
    fig_width = 16
    fig_height = max(10, total_qubits * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Enhanced color scheme
    colors = {
        'register_line': '#2C3E50',
        'hadamard': '#3498DB',
        'init': '#E74C3C',
        'modexp': '#9B59B6',
        'qft': '#2ECC71',
        'measure': '#F1C40F',
        'control': '#34495E',
        'box': '#BDC3C7'
    }
    
    # Draw register lines and labels
    def draw_register_line(y, label, register_type=None):
        ax.hlines(y=y, xmin=0, xmax=15, color=colors['register_line'], linewidth=1)
        # Enhanced label box
        bbox_props = dict(
            facecolor='white',
            edgecolor=colors['box'],
            boxstyle='round,pad=0.5',
            alpha=0.9
        )
        ax.text(-0.8, y, label, ha='right', va='center', fontsize=10,
                bbox=bbox_props)
        if register_type:
            ax.text(-1.5, y, register_type, ha='right', va='center',
                   fontsize=9, style='italic', color='#666666')

    # Draw registers with separators and clear labels
    current_y = 0
    
    # Phase Estimation Register
    for i in range(n_count):
        draw_register_line(current_y + i, f'|ψ{i}⟩', 
                          'Phase Est.' if i == n_count//2 else None)
    current_y += n_count
    
    # Separator
    ax.axhline(y=current_y-0.5, color='gray', linestyle=':', alpha=0.5)
    
    # Target Register
    for i in range(n_target):
        draw_register_line(current_y + i, f'|x{i}⟩', 
                          'Target' if i == n_target//2 else None)
    current_y += n_target
    
    # Separator
    ax.axhline(y=current_y-0.5, color='gray', linestyle=':', alpha=0.5)
    
    # Auxiliary Register
    for i in range(n_aux):
        draw_register_line(current_y + i, f'|a{i}⟩', 
                          'Auxiliary' if i == n_aux//2 else None)
    
    # Circuit elements
    x = 1  # Starting x position
    
    # Hadamard gates on phase estimation register
    for i in range(n_count):
        rect = plt.Rectangle((x-0.3, i-0.3), 0.6, 0.6, 
                           fill=True, color=colors['hadamard'])
        ax.add_patch(rect)
        ax.text(x, i, 'H', ha='center', va='center', 
                color='white', fontweight='bold')
    x += 1.5
    
    # Target register initialization
    y_target = n_count
    rect = plt.Rectangle((x-0.3, y_target-0.3), 0.6, 0.6, 
                        fill=True, color=colors['init'])
    ax.add_patch(rect)
    ax.text(x, y_target, '|1⟩', ha='center', va='center', 
            color='white', fontweight='bold')
    x += 1.5
    
    # Controlled modular exponentiation block
    block_width = 6
    block_height = n_count + n_target + n_aux
    rect = plt.Rectangle((x-0.3, -0.3), block_width, block_height-0.4,
                        fill=False, color=colors['modexp'], linewidth=2)
    ax.add_patch(rect)
    
    # Add modular exponentiation details
    ax.text(x + block_width/2, block_height/2-0.5,
            f'Controlled\nModular\nExponentiation\n$a^x$ mod {N}\n\n' + 
            'Using quantum\narithmetic circuits\nfor modular\nmultiplication',
            ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    x += block_width + 0.5
    
    # Inverse QFT on phase estimation register
    qft_width = 2
    rect = plt.Rectangle((x-0.3, -0.3), qft_width, n_count-0.4,
                        fill=False, color=colors['qft'], linewidth=2)
    ax.add_patch(rect)
    ax.text(x + qft_width/2, (n_count-1)/2,
            'QFT†\n(Inverse\nQuantum\nFourier\nTransform)',
            ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    x += qft_width + 0.5
    
    # Measurements on phase estimation register
    for i in range(n_count):
        rect = plt.Rectangle((x-0.3, i-0.3), 0.6, 0.6,
                           fill=True, color=colors['measure'])
        ax.add_patch(rect)
        ax.text(x, i, 'M', ha='center', va='center', fontweight='bold')
        # Classical measurement line
        ax.plot([x, x+1], [i, i], 'k--', alpha=0.5)
    
    # Add classical register at the end
    for i in range(n_count):
        rect = plt.Rectangle((x+1-0.3, i-0.3), 0.6, 0.6,
                           fill=True, color=colors['box'], alpha=0.3)
        ax.add_patch(rect)
        ax.text(x+1, i, 'c', ha='center', va='center', fontsize=10)
    
    # Adjust plot limits and style
    ax.set_xlim(-2, x+2)
    ax.set_ylim(-1, total_qubits)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Title with quantum resource requirements
    title = f"Shor's Algorithm Quantum Circuit for Factoring N={N}\n"
    title += f"Total Qubits: {total_qubits} (Phase Est.: {n_count}, Target: {n_target}, Aux: {n_aux})"
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
    
    # Add legend with detailed explanations
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors['hadamard'], label='Hadamard Gates\n(Superposition)'),
        plt.Rectangle((0,0),1,1, facecolor=colors['init'], label='State Initialization'),
        plt.Rectangle((0,0),1,1, facecolor=colors['modexp'], alpha=0.3, label='Modular Exponentiation'),
        plt.Rectangle((0,0),1,1, facecolor=colors['qft'], alpha=0.3, label='Inverse QFT'),
        plt.Rectangle((0,0),1,1, facecolor=colors['measure'], label='Measurement')
    ]
    ax.legend(handles=legend_elements, loc='center left',
             bbox_to_anchor=(1.05, 0.5), fontsize=10)
    
    plt.tight_layout()
    return fig

def analyze_circuit_depth(N):
    # More accurate resource estimation
    max_n = min(10000, max(2*N, 1000))
    n_values = np.logspace(np.log10(max(4, N//2)), np.log10(max_n), 50)
    depths = []
    qubits = []
    
    for n in n_values:
        # Precise register sizes
        count_qubits = ceil(2 * log2(n))
        target_qubits = ceil(log2(n))
        aux_qubits = ceil(log2(n))
        total_qubits = count_qubits + target_qubits + aux_qubits
        
        # Detailed depth calculation
        modexp_depth = count_qubits * target_qubits * log2(n)  # Modular exponentiation
        qft_depth = count_qubits * (count_qubits - 1) / 2      # Inverse QFT
        mod_arith_depth = target_qubits * log2(n)              # Modular arithmetic
        
        total_depth = modexp_depth + qft_depth + mod_arith_depth
        
        depths.append(total_depth)
        qubits.append(total_qubits)
    
    return n_values, np.array(depths), np.array(qubits)

def plot_circuit_analysis(N, n_values, depths, qubits):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Enhanced depth analysis plot
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.plot(n_values, depths, 'b-', label='Total Circuit Depth', linewidth=2)
    ax1.axvline(x=N, color='r', linestyle='--', label=f'N={N}')
    ax1.set_xlabel('Input Size (N)', fontsize=10)
    ax1.set_ylabel('Circuit Depth', fontsize=10)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize=10)
    ax1.set_title('Circuit Depth Analysis', fontsize=12)
    
    # Enhanced qubit requirements plot
    ax2.set_xscale('log')
    ax2.plot(n_values, qubits, 'g-', label='Total Qubits', linewidth=2)
    ax2.axvline(x=N, color='r', linestyle='--', label=f'N={N}')
    ax2.set_xlabel('Input Size (N)', fontsize=10)
    ax2.set_ylabel('Number of Qubits', fontsize=10)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(fontsize=10)
    ax2.set_title('Qubit Requirements Analysis', fontsize=12)
    
    plt.tight_layout()
    return fig

def simulate_period_finding(N, shots=2000):
    # Enhanced simulation for larger N
    if N % 2 == 0:
        r = 2
    else:
        # Find the first coprime base and its period
        for a in range(2, min(N, 100)):  # Limit search for large N
            if gcd(a, N) == 1:
                r = 1
                while r < min(N, 1000):  # Limit period search for large N
                    if pow(a, r, N) == 1:
                        break
                    r += 1
                break
        else:
            r = N  # Fallback if no period found
    
    # Calculate required precision and qubits
    n_qubits = int(np.ceil(2 * np.log2(N)))
    precision = 1.0 / (2 * r) if r != 0 else 0
    
    # Generate measurement outcomes with noise
    phases = np.random.normal(loc=precision, scale=precision/10, size=shots)
    measurements = np.mod(np.round(phases * (2**n_qubits)), 2**n_qubits)
    
    # Create histogram with adaptive binning
    n_bins = min(100, 2**n_qubits)  # Limit bins for large N
    counts, _ = np.histogram(measurements, bins=n_bins)
    
    return counts, r

def plot_period_finding_results(N, counts, period):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram with adaptive bin width
    n_bins = len(counts)
    bin_edges = np.linspace(0, 2**int(np.ceil(np.log2(N))), n_bins + 1)
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges)[0], alpha=0.7)
    
    # Highlight peaks
    peak_threshold = np.max(counts) * 0.5
    peaks = np.where(counts > peak_threshold)[0]
    for peak in peaks:
        ax.bar([bin_edges[peak]], [counts[peak]], 
               width=np.diff(bin_edges)[0], color='red', alpha=0.7)
    
    # Add period markers if applicable
    if period and period < n_bins:
        expected_spacing = n_bins / period
        for i in range(period):
            ax.axvline(x=i * expected_spacing, color='green', 
                      linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Measurement Outcome')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Period Finding Simulation (N={N}, Period={period})')
    
    plt.tight_layout()
    return fig
def calculate_speedup_ratios(N_values):
    """Calculate speedup ratios for Shor's Algorithm vs GNFS."""
    shor_times = []
    gnfs_times = []
    speedup_ratios = []

    for N in N_values:
        # Initialize algorithms
        shor = ShorAlgorithm(N)
        gnfs = GNFSAlgorithm(N)
        
        # Measure Shor's Algorithm time
        start_time = time.perf_counter()
        shor.factor()
        shor_time = time.perf_counter() - start_time
        shor_times.append(shor_time)
        
        # Measure GNFS time
        start_time = time.perf_counter()
        gnfs.factor()
        gnfs_time = time.perf_counter() - start_time
        gnfs_times.append(gnfs_time)
        
        # Calculate speedup ratio
        speedup = gnfs_time / shor_time if shor_time > 0 else float('inf')
        speedup_ratios.append(speedup)
    
    return shor_times, gnfs_times, speedup_ratios

def plot_speedup_ratios(N_values, speedup_ratios):
    """Plot speedup ratios for Shor's Algorithm vs GNFS."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(N_values, speedup_ratios, marker='o', linestyle='-', color='blue', label='Speedup Ratio')
    ax.set_xscale('log')
    ax.set_xlabel('Input Size (N)', fontsize=12)
    ax.set_ylabel('Speedup (GNFS Time / Shor Time)', fontsize=12)
    ax.set_title('Comparative Speedup Ratios for Shor\'s Algorithm vs GNFS', fontsize=14)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10)
    
    # Highlight specific points
    for i, N in enumerate(N_values):
        ax.annotate(f'N={N}', (N, speedup_ratios[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Quantum vs Classical Factoring Comparison")
    
    st.markdown("""
    This application compares Shor's quantum factoring algorithm with the General Number Field Sieve (GNFS).
    
    - **Shor's Algorithm**: Quantum complexity O((log N)² · log(log N) · log(log(log N)))
    - **GNFS**: Classical complexity O(exp((log N)^(1/3) · (log log N)^(2/3)))
    """)
    
    
    
    N = st.number_input(
            "Enter number to factor",
            min_value=4,
            value=15,
            help="Enter a composite number to factor"
        )
    
    N_values = [15, 77, 221, 437, 2047, 10055]
    if st.button("Run Comparison", type="primary"):
        with st.spinner("Running algorithms..."):
                # Initialize algorithms
                shor = ShorAlgorithm(N)
                gnfs = GNFSAlgorithm(N)
                
                # Run factorizations
                shor_factors, shor_time = shor.factor()
                gnfs_factors, gnfs_time = gnfs.factor()
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Results", "Circuit Analysis", "Period Finding", "Speedup Analysis"])
                
                with tab1:
                    # Display results side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Shor's Algorithm")
                        st.write(f"Factors: {shor_factors}")
                        st.write(f"Time: {shor_time:.6f} seconds")
                        
                        try:
                            circuit = shor.get_circuit()
                            fig = plot_quantum_circuit(circuit,N)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Circuit visualization error: {str(e)}")
                    
                    with col2:
                        st.subheader("GNFS Algorithm")
                        st.write(f"Factors: {gnfs_factors}")
                        st.write(f"Time: {gnfs_time:.6f} seconds")
                        
                        
                
                with tab2:
                    n_values, depths, qubits = analyze_circuit_depth(N)
                    fig = plot_circuit_analysis(N, n_values, depths, qubits)
                    st.pyplot(fig)
                    
                    # Pre-calculate circuit depth to avoid embedding logic in f-string
                    if len(np.where(n_values >= N)[0]) > 0:
                        circuit_depth = int(depths[np.where(n_values >= N)[0][0]])
                    else:
                        circuit_depth = int(depths[-1])  # Use the last depth as a fallback

                    # Resource requirements display
                    st.info(f"""
                    **Resource Requirements for N={N}:**
                    - Total Qubits: {int(np.ceil(4 * np.log2(N)))}
                    - Circuit Depth: {circuit_depth}
                    """)

                
                with tab3:
                    counts, period = simulate_period_finding(N)
                    fig = plot_period_finding_results(N, counts, period)
                    st.pyplot(fig)
                    
                    st.info(f"""
                    **Period Finding Results:**
                    - Found Period: {period}
                    - Number of Measurements: {len(counts)}
                    - Peak Count: {max(counts)}
                    """)
                with tab4:
                    st.subheader("Speedup Analysis")
                    shor_times, gnfs_times, speedup_ratios = calculate_speedup_ratios(N_values)
                    fig = plot_speedup_ratios(N_values, speedup_ratios)
                    st.pyplot(fig)
                
if __name__ == "__main__":
    main()