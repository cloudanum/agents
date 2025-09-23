# Copilot Instructions for AI Coding Agents

## Project Overview
This project demonstrates CSV file handling in Python using both the built-in `csv` module and `pandas`. The main script `test.py` implements three approaches to CSV processing:
1. Reading with `csv.reader` for basic row-by-row processing
2. Reading with `pandas` for data frame operations
3. Writing CSV files with different formatting options using `pandas`

## Key Patterns and Conventions
- **Dual CSV Processing**: Uses both `csv.reader` (for simple row iteration) and `pandas` (for data frame operations)
- **File Naming**: Input file is expected as `read.txt`
- **Output Files**: 
  - `write.txt`: CSV output with headers
  - `write_no_header.txt`: CSV output without headers
- **Error Handling**: Currently relies on Python's default file handling exceptions

## Developer Workflows
- **Setup**: Install required dependencies:
  ```bash
  pip install pandas
  ```
- **Run the script**: Execute with `python test.py` from the project root
- **Input/Output**:
  1. Place input CSV as `read.txt` in project root
  2. Run script to generate outputs:
     - Console output shows both raw rows and DataFrame
     - Two CSV files created with different header options

## Data Flow
1. Input: `read.txt` (CSV data source)
2. Processing:
   - Direct CSV reading via `csv.reader`
   - Pandas DataFrame creation and manipulation
3. Output:
   - Terminal: Raw CSV rows and DataFrame display
   - Files: Two formatted CSV files (with/without headers)

## File Reference
- `test.py`: Main script demonstrating multiple CSV processing patterns
- Input:
  - `read.txt`: Required CSV input file
- Output:
  - `write.txt`: CSV with headers
  - `write_no_header.txt`: CSV without headers

---

*Update this file as the project grows to document new conventions, workflows, and architectural decisions.*
