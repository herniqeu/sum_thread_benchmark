# Multi-Language Parallel Sum Benchmark Suite

## Technical Overview

This repository implements a parallel sum algorithm across multiple programming languages (C++, Go, Haskell, and Python), demonstrating various parallel programming paradigms and their performance characteristics in a controlled environment.

### System Configuration
- Platform: Linux (WSL2) x86_64
- Processor: 6 physical cores, 12 logical cores
- Memory: 7.61 GB
- Compiler/Runtime Versions:
  - GCC (C++17)
  - Go 1.11+
  - GHC 8.0+
  - Python 3.8.10

## Implementation Architecture

### 1. C++ Implementation (`sum_thread.cpp`)
- Uses RAII principles with `std::thread` and modern C++ features
- Leverages `std::accumulate` for vectorized operations
- Key optimizations:
  ```cpp
  partial_sum = std::accumulate(numbers.begin() + start_index, 
                               numbers.begin() + end_index, 0LL);
  ```
- Thread-safe design through data partitioning
- Performance: Best mean execution time (167.33ms ± 121.23ms)

### 2. Go Implementation (`sum_thread.go`)
- Utilizes goroutines and channels for communication
- Efficient memory management through Go's runtime
- Performance: Second-best mean execution time (203.35ms ± 179.97ms)

### 3. Haskell Implementation (`sum_thread.hs`)
- Employs Software Transactional Memory (STM)
- Pure functional approach with MVars for synchronization
- Performance: Third place (745.29ms ± 1129.14ms)

### 4. Python Implementation (`sum_thread.py`)
- GIL-constrained threading model
- Numpy-optimized operations where possible
- Performance: Baseline reference (2446.33ms ± 3862.51ms)

## Statistical Analysis

### Performance Metrics

#### 1. Execution Time Distribution [BoxPlot_Reference]
- C++: Lowest variability (CV: 72.45%)
- Go: Moderate variability (CV: 88.50%)
- Haskell: High variability (CV: 151.51%)
- Python: Highest variability (CV: 157.89%)

#### 2. Statistical Significance
- ANOVA results: F(3, 596) = 42.04, p < 1.18e-24
- Tukey HSD Analysis:
  - C++ vs Python: Significant (p < 0.001)
  - Go vs Python: Significant (p < 0.001)
  - Haskell vs Python: Significant (p < 0.001)
  - C++ vs Go: Non-significant (p = 0.9987)

### Scaling Analysis

#### 1. Small-Scale Performance (n ≤ 100,000)
- All languages perform similarly
- Mean execution times within 5% range
- Low coefficient of variation (<2%)

#### 2. Medium-Scale Performance (100,000 < n ≤ 1,000,000)
- C++ and Go maintain consistent performance
- Python shows linear degradation
- Haskell begins showing increased variance

#### 3. Large-Scale Performance (n > 1,000,000)
- C++: Best scaling (493.23ms at n=25M)
- Go: Linear scaling (701.25ms at n=25M)
- Haskell: Exponential growth (3885.12ms at n=25M)
- Python: Significant degradation (12866.13ms at n=25M)

## Performance Characteristics

### 1. Memory Efficiency
- C++: Lowest memory footprint (0.49% mean usage)
- Go: Efficient garbage collection (0.67% mean usage)
- Haskell: Higher memory overhead (3.28% mean usage)
- Python: Significant memory usage (2.84% mean usage)

### 2. Thread Scaling
- Linear scaling up to physical core count (6)
- Diminishing returns beyond logical core count (12)
- False sharing effects visible in large datasets

### 3. Cache Effects
- Visible in performance jumps at:
  - L1 cache boundary (~32KB)
  - L2 cache boundary (~256KB)
  - L3 cache boundary (~12MB)

## Technical Insights

1. **Algorithmic Complexity**
   - Theoretical: O(n/p), where p = thread count
   - Practical: Limited by memory bandwidth
   - Cache coherency overhead significant at thread boundaries

2. **Memory Access Patterns**
   - Sequential access benefits from hardware prefetching
   - Thread-local summation minimizes false sharing
   - NUMA effects visible in large datasets

3. **Synchronization Overhead**
   - Minimal in C++ (final reduction only)
   - Channel-based in Go (negligible impact)
   - STM overhead in Haskell (significant at scale)
   - GIL contention in Python (major bottleneck)

## Benchmark Runner Architecture

### Core Components (`run_benchmarks.py`)

1. **Execution Pipeline**
```python
def run_single_test(self, executable, size, num_threads, lang):
    process = psutil.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    metrics_samples.append(self._measure_process_metrics(process))
```
- Real-time metrics collection
- Process isolation per test
- Resource monitoring (CPU, memory, threads)

2. **Compilation Strategy**
- C++: `-O3 -pthread` optimizations
- Go: Native build with race detector
- Haskell: `-O2 -threaded` RTS options
- Python: JIT compilation through CPython

### Statistical Methodology

1. **Sampling Framework**
- Sample size: 150 runs per language
- Confidence level: 95%
- Power analysis: β > 0.95

2. **Variance Analysis**
```text
Language   Mean (ms)    Std Dev    CV (%)
----------------------------------------
C++        167.33      121.64     72.45
Go         203.35      180.57     88.50
Haskell    745.29     1132.93    151.51
Python    2446.33     3875.45    157.89
```

3. **Distribution Characteristics**
- Non-normal distributions (Shapiro-Wilk test)
  - C++: W = 0.5886, p < 8.92e-19
  - Go: W = 0.6002, p < 1.59e-18
  - Haskell: W = 0.6073, p < 2.27e-18
  - Python: W = 0.6411, p < 1.33e-17

## Detailed Performance Analysis

### 1. Small-Scale Efficiency (n ≤ 100,000)
```text
Size: 100,000
Language   Mean (ms)    CV (%)
----------------------------
C++        103.77      1.29
Go         104.60      1.12
Haskell    104.08      1.15
Python     104.62      1.25
```
- Negligible performance differences
- Cache-resident data sets
- Linear scaling with threads

### 2. Medium-Scale Behavior (n = 1,000,000)
```text
Language   Mean (ms)    CV (%)
----------------------------
C++        103.40      0.48
Go         105.70      1.11
Haskell    145.27     34.25
Python     406.15      0.49
```
- Memory hierarchy effects become visible
- Thread synchronization overhead emerges
- GC pressure in managed languages

### 3. Large-Scale Performance (n = 25,000,000)
```text
Language   Mean (ms)    CV (%)
----------------------------
C++        493.23     16.59
Go         701.25     12.38
Haskell   3885.12     17.08
Python   12866.13     22.14
```
- Memory bandwidth saturation
- NUMA effects dominant
- Garbage collection overhead significant

## Technical Optimizations

### 1. Memory Management
- Thread-local accumulation
- Cache-line alignment
- NUMA-aware thread pinning
- False sharing mitigation

### 2. Synchronization Strategies
```cpp
// C++: Lock-free accumulation
std::vector<std::thread> threads;
std::vector<SumThread> sum_threads;
```

```haskell
-- Haskell: MVar-based synchronization
results <- replicateM numWorkers newEmptyMVar
```

```python
# Python: Thread pooling
thread = SumThread(numbers, start_idx, end_idx)
threads.append(thread)
```

### 3. Compiler Optimizations
- Loop unrolling
- Vectorization
- Constant folding
- Dead code elimination

## Performance Bottlenecks

1. **Language-Specific Limitations**
- Python: GIL contention
- Haskell: Garbage collection pauses
- Go: Channel communication overhead
- C++: Cache coherency protocol

2. **Hardware Constraints**
- Memory bandwidth saturation
- Cache line bouncing
- NUMA access patterns
- Thread scheduling overhead

## Future Optimizations

1. **Implementation Improvements**
- SIMD vectorization
- Cache-oblivious algorithms
- Work-stealing schedulers
- Dynamic thread pooling

2. **Measurement Enhancements**
- Hardware performance counters
- Cache miss profiling
- Branch prediction statistics
- Memory bandwidth utilization

## Conclusions

1. **Performance Hierarchy**
- C++ provides best raw performance
- Go offers good balance of performance and safety
- Haskell shows competitive small-scale performance
- Python suitable for prototype development

2. **Scaling Characteristics**
- Linear scaling up to physical core count
- Memory bandwidth becomes bottleneck at scale
- Thread synchronization overhead increases with dataset size

3. **Statistical Significance**
- ANOVA confirms significant differences (p < 1.18e-24)
- Tukey HSD shows clear language performance tiers
- Non-normal distribution suggests complex performance factors

![Log Execution Time vs Input Size](/img/log_execution_time_vs_input_size.png)
![CPU Usage](/img/cpu_usage.png)
![Execution Time](/img/execution_time.png)
![Execution Time vs Input Size](/img/execution_time_vs_input_size.png)
![Memory Usage by Performance](/img/memory_usage_by_perfomance.png)
![Performance Consistency](/img/perfomance_consistency.png)
![Speedup Number of Threads](/img/speedup_number_of_threads.png)
![Statistical Confidence](/img/statistical_confidence.png)
![Thread Scaling Language](/img/thread_scaling_language.png)