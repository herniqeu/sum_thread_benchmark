module Main where

import Control.Concurrent
import Control.Monad
import Data.Time.Clock
import System.Environment
import System.Exit

-- Worker function to calculate partial sum
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

-- Helper function to split list into chunks
splitIntoChunks :: Int -> [a] -> [[a]]
splitIntoChunks _ [] = []
splitIntoChunks n xs = take n xs : splitIntoChunks n (drop n xs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [size, threads] -> do
            let size' = read size :: Int
                threads' = read threads :: Int
            numbers <- return $ [1..size']
            (sum', time) <- parallelSum numbers threads'
            putStrLn $ "Parallel sum: " ++ show sum'
            putStrLn $ "Time: " ++ show time
        _ -> do
            putStrLn "Usage: sum_thread <size> <threads>"
            exitFailure
