param(
    # all: run the entire unittest suite
    # data: run only dataset loading / label conversion tests
    # metrics: run only evaluation metric tests
    # export: run only submission-format validation tests
    [ValidateSet("all", "data", "metrics", "export")]
    [string]$Target = "all"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $ProjectRoot

try {
    switch ($Target) {
        "all" {
            # Full regression check for the current project.
            python -m unittest
        }
        "data" {
            # Verifies dataset loading and binary label conversion.
            python -m unittest tests.test_data
        }
        "metrics" {
            # Verifies precision / recall / F1 metric calculations.
            python -m unittest tests.test_metrics
        }
        "export" {
            # Verifies dev.txt / test.txt submission-format checks.
            python -m unittest tests.test_export
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Test command failed."
    }
}
finally {
    Pop-Location
}
