# Multi-Language Parallel Sum Benchmark Suite

## Overview
This repository contains a comprehensive benchmarking suite that implements and compares parallel sum algorithms across multiple programming languages (C++, Go, Haskell, and Python). The suite demonstrates various parallel programming paradigms, concurrency models, and their performance characteristics.

## Technical Architecture

### Core Components

1. **Benchmark Runner** (`run_benchmarks.py`)
- Orchestrates benchmark execution across languages
- Implements automated compilation and execution
- Generates comparative performance visualizations
- Uses matplotlib/seaborn for data visualization

2. **Language-Specific Implementations**

#### C++ Implementation

```1:60:src/cpp/sum_thread.cpp
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <numeric>
#include <string>

class SumThread {
private:
    const std::vector<int>& numbers;
    size_t start_index;
    size_t end_index;
    long long partial_sum;

public:
    SumThread(const std::vector<int>& nums, size_t start, size_t end)
        : numbers(nums), start_index(start), end_index(end), partial_sum(0) {}

    void calculate() {
        partial_sum = std::accumulate(numbers.begin() + start_index, 
                                    numbers.begin() + end_index, 0LL);
    }

    long long getPartialSum() const { return partial_sum; }
};

std::pair<long long, double> parallel_sum(const std::vector<int>& numbers, int num_threads) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    std::vector<SumThread> sum_threads;
    
    size_t chunk_size = numbers.size() / num_threads;
    
    // Create and start threads
    for (int i = 0; i < num_threads; ++i) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = (i == num_threads - 1) ? numbers.size() : (i + 1) * chunk_size;
        
        sum_threads.emplace_back(numbers, start_idx, end_idx);
        threads.emplace_back(&SumThread::calculate, &sum_threads.back());
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Calculate total sum
    long long total_sum = 0;
    for (const auto& sum_thread : sum_threads) {
        total_sum += sum_thread.getPartialSum();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    
    return {total_sum, duration};
}
```

- Utilizes modern C++ threading primitives
- Employs RAII principles for resource management
- Leverages std::accumulate for vectorized operations
- Uses move semantics for efficient thread handling

#### Go Implementation

```12:57:src/go/_sum_thread.go
// SumWorker calculates partial sum for a slice of numbers
func SumWorker(numbers []int, result chan<- int) {
	sum := 0
	for _, num := range numbers {
		sum += num
	}
	result <- sum
}

// ParallelSum calculates sum using multiple goroutines
func ParallelSum(numbers []int, numWorkers int) (int, time.Duration) {
	start := time.Now()

	chunkSize := len(numbers) / numWorkers
	results := make(chan int, numWorkers)
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		startIdx := i * chunkSize
		endIdx := startIdx + chunkSize
		if i == numWorkers-1 {
			endIdx = len(numbers)
		}

		go func(start, end int) {
			defer wg.Done()
			SumWorker(numbers[start:end], results)
		}(startIdx, endIdx)
	}

	// Close results channel when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Calculate total sum
	totalSum := 0
	for partialSum := range results {
		totalSum += partialSum
	}

	return totalSum, time.Since(start)
}
```

- Implements CSP (Communicating Sequential Processes) pattern
- Uses goroutines and channels for concurrent communication
- Employs WaitGroups for synchronization
- Demonstrates Go's lightweight thread model

#### Python Implementation

```6:54:src/python/sum_thread.py
class SumThread(threading.Thread):
    """Thread class to calculate partial sum of a list"""
    def __init__(self, numbers, start_index, end_index):
        threading.Thread.__init__(self)
        self.numbers = numbers
        self.start_index = start_index
        self.end_index = end_index
        self.partial_sum = 0

    def run(self):
        """Calculate the sum of the assigned portion of the list"""
        self.partial_sum = sum(self.numbers[self.start_index:self.end_index])

def parallel_sum(numbers, num_threads=2):
    """
    Calculate the sum of a list using multiple threads
    
    Args:
        numbers (list): List of numbers to sum
        num_threads (int): Number of threads to use
        
    Returns:
        tuple: Total sum and execution time
    """
    # Record start time
    start_time = time.time()
    
    # Calculate chunk size for each thread
    chunk_size = len(numbers) // num_threads
    
    # Create and start threads
    threads = []
    for i in range(num_threads):
        start_idx = i * chunk_size
        # Handle the last chunk which might be larger
        end_idx = len(numbers) if i == num_threads - 1 else (i + 1) * chunk_size
        
        thread = SumThread(numbers, start_idx, end_idx)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Calculate total sum from partial sums
    total_sum = sum(thread.partial_sum for thread in threads)
    
    # Calculate execution time
```

- Uses native threading module
- Implements OOP approach to thread management
- Demonstrates GIL (Global Interpreter Lock) limitations
- Provides clean thread lifecycle management

#### Haskell Implementation

```11:40:src/haskell/sum_thread.hs
sumWorker :: [Int] -> MVar Int -> IO ()
sumWorker numbers result = do
    let partialSum = sum numbers
    putMVar result partialSum

-- Parallel sum implementation
parallelSum :: [Int] -> Int -> IO (Int, Double)
parallelSum numbers numWorkers = do
    startTime <- getCurrentTime
    
    -- Create MVars for results
    results <- replicateM numWorkers newEmptyMVar
    
    -- Calculate chunk size and create workers
    let chunkSize = length numbers `div` numWorkers
        chunks = splitIntoChunks chunkSize numbers
    
    -- Start workers
    forM_ (zip chunks results) $ \(chunk, result) ->
        forkIO $ sumWorker chunk result
    
    -- Collect results
    partialSums <- mapM takeMVar results
    let totalSum = sum partialSums
    
    endTime <- getCurrentTime
    let duration = realToFrac $ diffUTCTime endTime startTime
    
    return (totalSum, duration)

```

- Leverages pure functional programming paradigm
- Uses MVars for thread synchronization
- Demonstrates immutable state management
- Implements lazy evaluation strategies

## Performance Characteristics

The suite measures:
- Sequential vs. parallel execution time
- Scaling efficiency with thread count
- Memory usage patterns
- Thread synchronization overhead

Test cases include:
1. Sequential numbers (cache-friendly access)
2. Random numbers (memory access patterns)
3. Uniform numbers (compiler optimization cases)

## Advanced Concepts Demonstrated

1. **Concurrency Models**
   - OS-level threads (C++)
   - Green threads (Go)
   - GIL-constrained threads (Python)
   - Software transactional memory (Haskell)

2. **Memory Access Patterns**
   - Cache coherency effects
   - False sharing mitigation
   - NUMA considerations
   - Memory barrier implications

3. **Synchronization Mechanisms**
   - Mutex-based synchronization (C++)
   - Channel-based communication (Go)
   - MVar synchronization (Haskell)
   - Thread joining strategies

## Usage

```bash
# Run complete benchmark suite
python src/benchmark/run_benchmarks.py

# Individual language tests
./src/cpp/sum_thread <size> <num_threads>
./src/go/sum_thread <size> <num_threads>
python src/python/sum_thread.py <size> <num_threads>
./src/haskell/sum_thread <size> <num_threads>
```

## Performance Analysis

The suite generates two types of visualizations:
1. Performance scaling by input size (logarithmic scale)
2. Thread scaling efficiency comparison

Key metrics:
- Absolute execution time
- Relative speedup vs. sequential
- Thread scaling efficiency
- Memory bandwidth utilization

## Technical Considerations

1. **Thread Pooling**
   - Dynamic vs. static thread allocation
   - Work stealing algorithms
   - Thread affinity impacts

2. **Memory Management**
   - Stack vs. heap allocation
   - Memory fence operations
   - Cache line alignment
   - False sharing prevention

3. **Compiler Optimizations**
   - Vectorization opportunities
   - Loop unrolling effects
   - Inlining decisions
   - Dead code elimination

## Requirements

- C++ compiler with C++11 support
- Go 1.11+
- Python 3.7+
- GHC (Glasgow Haskell Compiler) 8.0+
- matplotlib, seaborn, pandas (Python packages)

## Future Improvements

1. Implementation of:
   - SIMD vectorization
   - Cache-oblivious algorithms
   - Work stealing schedulers
   - Dynamic thread pooling

2. Additional metrics:
   - Cache miss rates
   - Branch prediction statistics
   - Memory bandwidth utilization
   - Context switching overhead