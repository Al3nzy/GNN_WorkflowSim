@echo off
REM Run script for WorkflowSim Benchmark

echo Running LIWSA Benchmark...
echo.

java -cp "bin;lib\*" org.workflowsim.examples.planning.LIWSABenchmarkExample

echo.
echo Benchmark complete. Results written to results/benchmark_results.csv
