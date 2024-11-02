package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

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

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <size> <num_threads>\n", os.Args[0])
		os.Exit(1)
	}

	size, _ := strconv.Atoi(os.Args[1])
	numThreads, _ := strconv.Atoi(os.Args[2])

	// Test cases
	testCases := [][]int{
		make([]int, size), // Sequential numbers
		make([]int, size), // Random numbers
		make([]int, size), // Uniform numbers
	}

	// Initialize test cases
	for i := range testCases[0] {
		testCases[0][i] = i
	}

	for i := range testCases[1] {
		testCases[1][i] = rand.Intn(100) + 1
	}

	for i := range testCases[2] {
		testCases[2][i] = 1
	}

	// Run tests
	for i, numbers := range testCases {
		fmt.Printf("\nTest Case %d:\n", i+1)
		fmt.Printf("List size: %d\n", len(numbers))

		// Regular sum
		start := time.Now()
		regularSum := 0
		for _, num := range numbers {
			regularSum += num
		}
		regularTime := time.Since(start)

		fmt.Printf("Regular sum: %d\n", regularSum)
		fmt.Printf("Regular time: %v\n", regularTime)

		// Parallel sum
		parallelSum, parallelTime := ParallelSum(numbers, numThreads)
		fmt.Printf("Parallel sum: %d\n", parallelSum)
		fmt.Printf("Parallel time: %v\n", parallelTime)
		fmt.Printf("Speed improvement: %.2fx\n", float64(regularTime)/float64(parallelTime))
	}
}
