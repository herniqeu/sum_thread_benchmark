import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
from pathlib import Path

class BenchmarkRunner:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.results = []
        
    def compile_cpp(self):
        """Compile C++ program"""
        cpp_path = self.root_dir / "cpp" / "sum_thread.cpp"
        output_path = self.root_dir / "cpp" / "sum_thread"
        subprocess.run(["g++", "-O3", "-pthread", str(cpp_path), "-o", str(output_path)])
        return output_path

    def compile_go(self):
        """Compile Go program"""
        go_path = self.root_dir / "go" / "_sum_thread.go"
        output_path = self.root_dir / "go" / "sum_thread"
        subprocess.run(["go", "build", "-o", str(output_path), str(go_path)])
        return output_path

    def compile_haskell(self):
        """Compile Haskell program"""
        hs_path = self.root_dir / "haskell" / "sum_thread.hs"
        output_path = self.root_dir / "haskell" / "sum_thread"
        subprocess.run(["ghc", "-O2", "-threaded", str(hs_path), "-o", str(output_path)])
        return output_path

    def run_benchmark(self, size, num_threads):
        """Run benchmark for all languages with given parameters"""
        # C++
        cpp_exec = self.compile_cpp()
        start_time = time.time()
        subprocess.run([str(cpp_exec), str(size), str(num_threads)], capture_output=True)
        cpp_time = time.time() - start_time

        # Go
        go_exec = self.compile_go()
        start_time = time.time()
        subprocess.run([str(go_exec), str(size), str(num_threads)], capture_output=True)
        go_time = time.time() - start_time

        # Haskell
        hs_exec = self.compile_haskell()
        start_time = time.time()
        subprocess.run([str(hs_exec), str(size), str(num_threads)], capture_output=True)
        hs_time = time.time() - start_time

        # Python
        py_path = self.root_dir / "python" / "sum_thread.py"
        start_time = time.time()
        subprocess.run(["python", str(py_path), str(size), str(num_threads)], capture_output=True)
        py_time = time.time() - start_time

        return {
            'size': size,
            'threads': num_threads,
            'cpp': cpp_time,
            'go': go_time,
            'haskell': hs_time,
            'python': py_time
        }

    def run_all_benchmarks(self):
        """Run benchmarks with various parameters"""
        sizes = [100000, 1000000, 10000000]
        thread_counts = [2, 4, 8]

        for size in sizes:
            for threads in thread_counts:
                result = self.run_benchmark(size, threads)
                self.results.append(result)

    def plot_results(self):
        """Generate plots from benchmark results"""
        df = pd.DataFrame(self.results)
        
        # Plot by size
        plt.figure(figsize=(12, 6))
        for lang in ['cpp', 'go', 'haskell', 'python']:
            plt.plot(df['size'], df[lang], marker='o', label=lang.upper())
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Input Size')
        plt.ylabel('Execution Time (s)')
        plt.title('Performance Comparison by Input Size')
        plt.legend()
        plt.grid(True)
        plt.savefig('benchmark_by_size.png')
        plt.close()

        # Plot by threads
        plt.figure(figsize=(12, 6))
        for lang in ['cpp', 'go', 'haskell', 'python']:
            plt.plot(df['threads'], df[lang], marker='o', label=lang.upper())
        
        plt.xlabel('Number of Threads')
        plt.ylabel('Execution Time (s)')
        plt.title('Performance Comparison by Thread Count')
        plt.legend()
        plt.grid(True)
        plt.savefig('benchmark_by_threads.png')
        plt.close()

def main():
    runner = BenchmarkRunner()
    runner.run_all_benchmarks()
    runner.plot_results()

if __name__ == "__main__":
    main() 