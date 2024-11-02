import threading
import time
import random
import sys

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
    execution_time = time.time() - start_time
    
    return total_sum, execution_time

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <size> <num_threads>")
        sys.exit(1)

    size = int(sys.argv[1])
    num_threads = int(sys.argv[2])

    # Test cases
    test_cases = [
        list(range(size)),  # Sequential numbers
        [random.randint(1, 100) for _ in range(size)],  # Random numbers
        [1] * size  # Uniform numbers
    ]
    
    for i, numbers in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"List size: {len(numbers)}")
        
        # Calculate sum using regular method for comparison
        start_time = time.time()
        regular_sum = sum(numbers)
        regular_time = time.time() - start_time
        print(f"Regular sum: {regular_sum}")
        print(f"Regular execution time: {regular_time:.4f} seconds")
        
        # Calculate sum using parallel method
        parallel_result, parallel_time = parallel_sum(numbers)
        print(f"Parallel sum: {parallel_result}")
        print(f"Parallel execution time: {parallel_time:.4f} seconds")
        print(f"Speed improvement: {(regular_time/parallel_time):.2f}x")

if __name__ == "__main__":
    main()