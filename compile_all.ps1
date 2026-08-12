#!/usr/bin/env pwsh
# Comprehensive build script - compiles all sources in one batch

$ErrorActionPreference = "Continue"

if (-not (Test-Path "bin")) { New-Item -ItemType Directory -Path "bin" | Out-Null }

Write-Host ""
Write-Host "============================================================"
Write-Host "Compiling ALL WorkflowSim and CloudSim sources"
Write-Host "============================================================"
Write-Host ""

# Get all Java files from sources (everything that CloudSim and WorkflowSim need)
$sourceFiles = @()
$sourceFiles += Get-ChildItem -Path "sources" -Recurse -Include "*.java" | ForEach-Object { $_.FullName }
$sourceFiles += @(
    "examples/org/workflowsim/examples/planning/ParetoMetrics.java",
    "examples/org/workflowsim/examples/planning/ResultsCsvWriter.java",
    "examples/org/workflowsim/examples/planning/RunMetricsCalculator.java",
    "examples/org/workflowsim/examples/planning/LIWSABenchmarkExample.java"
)

Write-Host "Found $($sourceFiles.Count) Java files to compile"
Write-Host ""

# Compile all at once
$jarFiles = Get-ChildItem -Path "lib" -Include "*.jar" | ForEach-Object { $_.FullName }
$cp = ($jarFiles -join ";")

Write-Host "Compiling with classpath: $($cp.Substring(0, [Math]::Min(100, $cp.Length)))..."
Write-Host ""

$compileCmd = @(
    "javac"
    "-encoding", "UTF-8"
    "-cp", $cp
    "-d", "bin"
    "-XDignore.symbol.file"
) + $sourceFiles

& $compileCmd 2>&1 | Where-Object { $_ -match "error" } | Select-Object -First 50

if (Test-Path "bin/org/workflowsim/examples/planning/LIWSABenchmarkExample.class") {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "SUCCESS! Compilation complete."
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "To run the benchmark, execute:"
    Write-Host "  java -cp `"bin;lib\*`" org.workflowsim.examples.planning.LIWSABenchmarkExample"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Compilation Status:"
    Write-Host "============================================================"
    (Get-ChildItem -Path "bin" -Recurse -Include "*.class" | Measure-Object).Count | Write-Host "Compiled class files: "
}
