import streamlit as st
import random
import matplotlib.pyplot as plt
from math import ceil, log2, sqrt
import numpy as np
import pandas as pd
from grover_paper import grovers_algorithm
from linear_search_paper import classical_unsorted_search
from analysis import analyze_results

def main():
    st.set_page_config(page_title="Grover's Algorithm Benchmark", layout="wide")
    
    st.title("Grover's Algorithm vs Classical Linear Search")
    st.markdown("""
    This application compares the performance of Grover's quantum search algorithm with classical linear search.
    The comparison is made across different problem sizes and number of target elements.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Advanced configuration options
    with st.sidebar.expander("Advanced Settings"):
        shots = st.number_input("Number of shots:", min_value=100, max_value=10000, value=1024)
        plot_type = st.selectbox(
            "Plot Type:", 
            ["Oracle Calls", "Success Rate", "Circuit Depth"],
            help="Select the metric to visualize"
        )
        show_circuit = st.checkbox(
            "Show Circuit Diagram", 
            value=True,
            help="Display quantum circuit visualization"
        )
    
    # Main parameters
    col1, col2 = st.columns(2)
    with col1:
        start_size = st.number_input(
            "Start Size (N = 2^n):", 
            min_value=4, 
            value=16,
            help="Minimum problem size to test"
        )
        end_size = st.number_input(
            "End Size (N = 2^n):", 
            min_value=start_size + 1, 
            value=64,
            help="Maximum problem size to test"
        )
    
    with col2:
        num_targets = st.number_input(
            "Number of targets (k):", 
            min_value=1, 
            value=3,
            help="Number of elements to search for"
        )
        version = st.selectbox(
            "Grover Version:", 
            ["optimistic", "pessimistic"],
            help="Optimistic uses more iterations, pessimistic uses fewer"
        )
    
    if st.button("Run Benchmark", help="Start the comparison"):
        sizes = [2 ** i for i in range(int(log2(start_size)), int(log2(end_size)) + 1)]
        results = {
            'sizes': sizes,
            'classical_calls': [],
            'grover_calls': [],
            'success_rates': [],
            'circuit_depths': []
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Circuit visualization
        if show_circuit:
            status_text.text("Generating circuit visualization...")
            example_size = sizes[0]
            example_targets = random.sample(range(example_size), min(num_targets, example_size))
            _, _, _, visualizations = grovers_algorithm(
                example_size, 
                example_targets, 
                version=version, 
                shots=shots, 
                visualize=True
            )
            
            try:
                st.subheader("Quantum Circuit Visualization")
                col1, col2 = st.columns(2)
                
                with col1:
                    text_complete, fig_complete = visualizations['complete_circuit']
                    st.pyplot(fig_complete)
                    st.caption("Complete Grover's Circuit")
                
                with col2:
                    text_first, fig_first = visualizations['first_iteration']
                    st.pyplot(fig_first)
                    st.caption("First Grover Iteration")
                
                with st.expander("Circuit Text Representation"):
                    st.code(text_complete, language="text")
                
                plt.close('all')
            
            except Exception as e:
                st.error(f"Error displaying circuit: {str(e)}")
        
        # Run benchmarks
        for i, size in enumerate(sizes):
            status_text.text(f"Processing size {size}...")
            search_space = list(range(size))
            targets = random.sample(search_space, min(num_targets, size))
            
            # Classical search
            _, classical_calls = classical_unsorted_search(search_space, targets)
            results['classical_calls'].append(classical_calls)
            
            # Grover's algorithm
            counts, exec_time, depth, _ = grovers_algorithm(
                size, 
                targets, 
                version=version, 
                shots=shots, 
                visualize=False
            )
            analysis = analyze_results(counts, targets, size)
            
            results['grover_calls'].append(int(sqrt(size / len(targets))) * len(targets))
            results['success_rates'].append(analysis['success_rate'])
            results['circuit_depths'].append(depth)
            
            progress_bar.progress((i + 1) / len(sizes))
        
        # Display results
        st.subheader("Benchmark Results")
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Oracle Calls":
            ax.plot(sizes, results['classical_calls'], 'b-', label='Classical Linear Search')
            ax.plot(sizes, results['grover_calls'], 'r-', label=f"Grover's Algorithm ({version})")
            ax.set_ylabel("Number of Oracle Calls")
            ax.set_yscale('log', base=2)
        elif plot_type == "Success Rate":
            ax.plot(sizes, results['success_rates'], 'g-', label='Grover Success Rate')
            ax.set_ylabel("Success Rate")
        else:  # Circuit Depth
            ax.plot(sizes, results['circuit_depths'], 'm-', label='Circuit Depth')
            ax.set_ylabel("Circuit Depth")
        
        ax.set_xlabel("Problem Size (N)")
        ax.set_title(f"{plot_type} vs Problem Size")
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log', base=2)
        
        st.pyplot(fig)
        plt.close()
        
        # Display numerical results
        st.subheader("Detailed Results")
        results_df = pd.DataFrame({
            'Size': sizes,
            'Classical Calls': results['classical_calls'],
            'Grover Calls': results['grover_calls'],
            'Success Rate': results['success_rates'],
            'Circuit Depth': results['circuit_depths']
        })
        st.dataframe(results_df)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Success Rate", 
                f"{np.mean(results['success_rates']):.2%}"
            )
        with col2:
            classical_speedup = np.mean(np.array(results['classical_calls']) / 
                                     np.array(results['grover_calls']))
            st.metric(
                "Average Quantum Speedup", 
                f"{classical_speedup:.2f}x"
            )
        with col3:
            st.metric(
                "Average Circuit Depth", 
                f"{np.mean(results['circuit_depths']):.1f}"
            )
        
        status_text.text("Benchmark complete!")

if __name__ == "__main__":
    main()