import Control.Concurrent
import Control.Concurrent.MVar
import Control.Monad
import Data.Time.Clock
import System.Random
import Text.Printf
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

-- Regular sum with timing
regularSum :: [Int] -> IO (Int, Double)
regularSum numbers = do
    startTime <- getCurrentTime
    let result = sum numbers
    endTime <- getCurrentTime
    let duration = realToFrac $ diffUTCTime endTime startTime
    return (result, duration)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [size, threads] -> do
            let size' = read size :: Int
                threads' = read threads :: Int
