#!/usr/bin/env python3
"""
Test runner script for Dream Pixel Forge
"""
import sys
import pytest

def main():
    """Run the test suite"""
    # Add coverage and other options
    args = [
        "--cov=dream_pixel_forge",
        "--cov-report=term",
        "--cov-report=html:coverage_report",
        "-v",
    ]
    
    # Allow passing additional arguments from command line
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # Run the tests
    sys.exit(pytest.main(args))

if __name__ == "__main__":
    main() 