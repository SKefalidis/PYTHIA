# AI Generated File

#!/bin/bash

# Usage: ./split_triples.sh input_file output_directory num_subfiles

# Check for required arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 input_file output_directory num_subfiles"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="$2"
NUM_SPLITS="$3"

# Ensure the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 2
fi

# Ensure the output directory exists (create it if not)
mkdir -p "$OUTPUT_DIR"

# Count total lines in the input file
TOTAL_LINES=$(wc -l < "$INPUT_FILE")

# Calculate lines per split (minimum 1)
if [ "$NUM_SPLITS" -le 0 ]; then
    echo "Error: num_subfiles must be greater than 0."
    exit 3
fi

LINES_PER_SPLIT=$((TOTAL_LINES / NUM_SPLITS))
[ "$LINES_PER_SPLIT" -eq 0 ] && LINES_PER_SPLIT=1

echo "Total lines: $TOTAL_LINES"
echo "Lines per subfile: $LINES_PER_SPLIT"
echo "Splitting into approximately $NUM_SPLITS subfiles..."

# Run the split command
split -d -l "$LINES_PER_SPLIT" "$INPUT_FILE" "$OUTPUT_DIR/subfile_"

echo "Done. Files saved to: $OUTPUT_DIR"
