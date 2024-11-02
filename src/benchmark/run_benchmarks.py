import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
import json
import psutil
import platform
from datetime import datetime
from pathlib import Path

class BenchmarkRunner:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.results = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self):
        """Collect system information"""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "python_version": platform.python_version(),
        }

    def _measure_process_metrics(self, process):
        """Measure process metrics"""
        try:
            with psutil.Process(process.pid).oneshot():
                return {
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "memory_rss": process.memory_info().rss,
                    "memory_vms": process.memory_info().vms,
                    "threads": process.num_threads(),
                }
        except:
            return {}

    def run_single_test(self, executable, size, num_threads, lang):
        """Run a single test and collect metrics"""
        start_time = time.time()
        process = psutil.Popen(
            [str(executable), str(size), str(num_threads)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Collect metrics during execution
        metrics_samples = []
        while process.poll() is None:
            metrics_samples.append(self._measure_process_metrics(process))
            time.sleep(0.1)
            
        stdout, stderr = process.communicate()
        end_time = time.time()
        
        # Calculate average metrics
        avg_metrics = {
            k: np.mean([d[k] for d in metrics_samples if k in d])
            for k in ["cpu_percent", "memory_percent", "memory_rss", "memory_vms", "threads"]
        }
        
        return {
            "language": lang,
            "execution_time_ms": (end_time - start_time) * 1000,
            "metrics": avg_metrics,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "exit_code": process.returncode,
        }

    def run_benchmark(self, size, num_threads):
        """Run benchmark for all languages with given parameters"""
        results = {
            "size": size,
            "threads": num_threads,
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        # Run tests for each language
        test_configs = [
            (self.compile_cpp(), "cpp"),
            (self.compile_go(), "go"),
            (self.compile_haskell(), "haskell"),
            (self.root_dir / "python" / "sum_thread.py", "python")
        ]
        
        for executable, lang in test_configs:
            test_result = self.run_single_test(executable, size, num_threads, lang)
            results["tests"].append(test_result)
            
        return results

    def run_all_benchmarks(self):
        """Run benchmarks with various parameters"""
        sizes = [
            1000,          
            10_000,        # 10K
            100_000,       # 100K
            500_000,       # 500K
            1_000_000,     # 1M
            2_500_000,     # 2.5M
            5_000_000,     # 5M
            7_500_000,     # 7.5M
            10_000_000,    # 10M
            25_000_000     # 25M 
        ]
        thread_counts = [1, 2, 4, 8, 16]
        iterations = 3  # Run each test multiple times
        
        full_results = {
            "system_info": self.system_info,
            "benchmark_runs": []
        }
        
        for size in sizes:
            for threads in thread_counts:
                for i in range(iterations):
                    result = self.run_benchmark(size, threads)
                    full_results["benchmark_runs"].append(result)
                    self.results.append(self._extract_plot_data(result))
        
        # Save full results to JSON
        with open('benchmark_results.json', 'w') as f:
            json.dump(full_results, f, indent=2)

    def _extract_plot_data(self, result):
        """Extract data for plotting from benchmark result"""
        plot_data = {
            'size': result['size'],
            'threads': result['threads']
        }
        
        for test in result['tests']:
            plot_data[test['language']] = test['execution_time_ms']
        
        return plot_data

    def compile_cpp(self):
        """Compile C++ program"""
        cpp_path = self.root_dir / "cpp" / "sum_thread.cpp"
        output_path = self.root_dir / "cpp" / "sum_thread"
        subprocess.run(["g++", "-std=c++17", "-O3", "-pthread", str(cpp_path), "-o", str(output_path)])
        return output_path

    def compile_go(self):
        """Compile Go program"""
        go_path = self.root_dir / "go" / "sum_thread.go"
        output_path = self.root_dir / "go" / "sum_thread"
        subprocess.run(["go", "build", "-o", str(output_path), str(go_path)])
        return output_path

    def compile_haskell(self):
        """Compile Haskell program"""
        hs_path = self.root_dir / "haskell" / "sum_thread.hs"
        output_path = self.root_dir / "haskell" / "sum_thread"
        subprocess.run(["ghc", "-O2", "-threaded", str(hs_path), "-o", str(output_path)])
        return output_path

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