#!/bin/bash

# This script continuously checks your Slurm jobs using 'squeue --me'
# and displays the output every second.

echo "--- Starting Slurm job monitoring for user $(whoami) ---"
echo "Press Ctrl+C to stop."
echo ""

while true; do
    clear # Clears the terminal screen for a fresh display
    echo "Last checked: $(date)"
    echo "-------------------------------------"
    squeue --me
    echo "-------------------------------------"
    sleep 3 # Wait for 1 second before the next check
done
