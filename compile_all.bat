@echo off
REM Comprehensive build script - compiles all sources as one batch
REM This avoids circular dependency issues by compiling everything together

setlocal enabledelayedexpansion

if not exist bin mkdir bin

echo.
echo ============================================================
echo Compiling ALL sources in dependency order
echo ============================================================
echo.

REM Collect all Java files
setlocal enabledelayedexpansion
set SOURCES=

REM Core classes first (alphabetically helps with order)
for /r sources\org\cloudbus\cloudsim\core %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\cloudbus\cloudsim\lists %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\cloudbus\cloudsim\provisioners %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\cloudbus\cloudsim\distributions %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\cloudbus\cloudsim\network %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\cloudbus\cloudsim %%f in (*.java) do set SOURCES=!SOURCES! "%%f"

REM WorkflowSim sources
for /r sources\org\workflowsim\utils %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\workflowsim\clustering %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\workflowsim\failure %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\workflowsim\reclustering %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\workflowsim\scheduling %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\workflowsim %%f in (*.java) do set SOURCES=!SOURCES! "%%f"
for /r sources\org\workflowsim\planning %%f in (*.java) do set SOURCES=!SOURCES! "%%f"

REM Example utility classes (these have no external dependencies)
set SOURCES=!SOURCES! "examples\org\workflowsim\examples\planning\ParetoMetrics.java"
set SOURCES=!SOURCES! "examples\org\workflowsim\examples\planning\ResultsCsvWriter.java"
set SOURCES=!SOURCES! "examples\org\workflowsim\examples\planning\RunMetricsCalculator.java"

REM Main benchmark example
set SOURCES=!SOURCES! "examples\org\workflowsim\examples\planning\LIWSABenchmarkExample.java"

echo Compiling %SOURCES:~0,100%...
javac -encoding UTF-8 -cp "lib\*" -d bin %SOURCES% 2>&1 | findstr "error" | findstr /V "^$"

if exist bin\org\workflowsim\examples\planning\LIWSABenchmarkExample.class (
  echo.
  echo ============================================================
  echo SUCCESS! Compilation complete.
  echo ============================================================
  echo.
  echo To run the benchmark, execute:
  echo   java -cp "bin;lib\*" org.workflowsim.examples.planning.LIWSABenchmarkExample
  echo.
) else (
  echo.
  echo ============================================================
  echo ERROR: Compilation failed - LIWSABenchmarkExample.class not found
  echo ============================================================
)
