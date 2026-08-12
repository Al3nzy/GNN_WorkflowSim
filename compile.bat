@echo off
REM Build script for WorkflowSim Benchmark - Proper Dependency Order
REM Compiles: CloudSim core -> CloudSim -> WorkflowSim -> Examples

setlocal enabledelayedexpansion

if not exist bin mkdir bin

echo.
echo ============================================================
echo Stage 1: Compiling CloudSim core (org.cloudbus.cloudsim.core)
echo ============================================================

javac -encoding UTF-8 -d bin sources\org\cloudbus\cloudsim\core\*.java 2>&1 | findstr "error" | findstr /V "^$"

echo.
echo ============================================================
echo Stage 2: Compiling CloudSim base (org.cloudbus.cloudsim)
echo ============================================================

javac -encoding UTF-8 -d bin sources\org\cloudbus\cloudsim\*.java 2>&1 | findstr "error" | findstr /V "^$"

echo.
echo ============================================================
echo Stage 3: Compiling CloudSim provisioners
echo ============================================================

javac -encoding UTF-8 -cp bin -d bin sources\org\cloudbus\cloudsim\provisioners\*.java 2>&1 | findstr "error" | findstr /V "^$"

echo.
echo ============================================================
echo Stage 4: Compiling WorkflowSim utils
echo ============================================================

javac -encoding UTF-8 -cp "bin;lib\*" -d bin sources\org\workflowsim\utils\*.java 2>&1 | findstr "error" | findstr /V "^$"

echo.
echo ============================================================
echo Stage 5: Compiling WorkflowSim core
echo ============================================================

javac -encoding UTF-8 -cp "bin;lib\*" -d bin sources\org\workflowsim\*.java 2>&1 | findstr "error" | findstr /V "^$"

echo.
echo ============================================================
echo Stage 6: Compiling WorkflowSim planning algorithms
echo ============================================================

javac -encoding UTF-8 -cp "bin;lib\*" -d bin sources\org\workflowsim\planning\*.java 2>&1 | findstr "error" | findstr /V "^$"

echo.
echo ============================================================
echo Stage 7: Compiling Examples
echo ============================================================

javac -encoding UTF-8 -cp "bin;lib\*" -d bin examples\org\workflowsim\examples\planning\*.java 2>&1 | findstr "error" | findstr /V "^$"

echo.
echo ============================================================
echo Compilation complete!
echo ============================================================
echo.
echo Compiled classes: %NUMBER_OF_NODES% class files
echo.
echo To run the benchmark, execute:
echo   java -cp "bin;lib\*" org.workflowsim.examples.planning.LIWSABenchmarkExample
echo.
