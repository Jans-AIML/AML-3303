## Step 1: Raw / Messy Code (Before Principles)

# Messy code – not modular, not reusable, hard to maintain 
import random

numbers = [random.randint(1, 100) for _ in range(10)] # Generate 10 random numbers 
print("Generated numbers:", numbers) # Print numbers

# Calculate average
total = 0 # Sum variable
for n in numbers: 
    total += n
average = total / len(numbers)
print("Average:", average)

# Find max
max_num = numbers[0]
for n in numbers:
    if n > max_num:
        max_num = n
print("Max:", max_num)


"""
🔴 Problems:

No functions (not modular).

Can’t reuse logic elsewhere.

Hard to extend (e.g., adding min/median).

Not scalable (works only for small lists).

No error handling (reliability issue).

No comments/documentation.
"""

## Step 2: Refactored Code (With Principles)

import random # Importing random module
from typing import List # Importing List type for type hints

def generate_numbers(count: int, lower: int = 1, upper: int = 100) -> List[int]:
    """Generate a list of random integers."""
    return [random.randint(lower, upper) for _ in range(count)]

def calculate_average(numbers: List[int]) -> float:
    """Return the average of a list of numbers."""
    if not numbers:
        raise ValueError("List of numbers cannot be empty")
    return sum(numbers) / len(numbers)

def find_max(numbers: List[int]) -> int:
    """Return the maximum number from a list."""
    if not numbers:
        raise ValueError("List of numbers cannot be empty")
    return max(numbers)

if __name__ == "__main__": 
    # Example workflow (can be reused in other projects)
    nums = generate_numbers(10)
    print("Generated numbers:", nums)
    print("Average:", calculate_average(nums))
    print("Max:", find_max(nums))

"""
✅ Improvements:

Modularity: Code broken into functions.

Reusability: Functions can be used in any project.

Maintainability: Easy to add min/median later.

Scalability: Can handle larger datasets (just change count).

Reliability & Quality: Error handling included.

Security & Trust: Checks against empty input.

Collaboration: Docstrings/comments make it understandable for teams.
"""

### Classroom Activity 

## Step 1: Raw / Messy Pandas Code

import pandas as pd

# Load CSV
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Print average sepal length
avg = df['sepal_length'].mean()
print("Average sepal length:", avg)

# Print max petal width
mx = df['petal_width'].max()
print("Max petal width:", mx)

# Filter rows where species is setosa
print(df[df['species'] == 'setosa'].head())

"""
🔴 Problems:

All logic in one block → not modular.

Hard to reuse functions for other datasets.

No error handling → breaks if column names change.

Not scalable (imagine working on multiple CSVs).

No documentation → not good for collaboration.
"""

## Step 2: Refactored Pandas Code (With SDLC Principles)

import pandas as pd
from typing import Optional

def load_data(url: str) -> pd.DataFrame:
    """
    Load CSV data from a URL with error handling.
    
    Args:
        url: The URL of the CSV file
        
    Returns:
        A pandas DataFrame containing the data
        
    Raises:
        ValueError: If URL is invalid or data cannot be loaded
    """
    try:
        df = pd.read_csv(url)
        if df.empty:
            raise ValueError("Loaded data is empty")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load data from {url}: {str(e)}")

def calculate_column_mean(df: pd.DataFrame, column: str) -> float:
    """
    Calculate the mean of a specified column with validation.
    
    Args:
        df: The pandas DataFrame
        column: The column name to calculate mean for
        
    Returns:
        The mean value of the column
        
    Raises:
        ValueError: If column doesn't exist or contains no numeric data
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if df[column].empty:
        raise ValueError(f"Column '{column}' contains no data")
    return df[column].mean()

def find_column_max(df: pd.DataFrame, column: str) -> float:
    """
    Find the maximum value in a specified column with validation.
    
    Args:
        df: The pandas DataFrame
        column: The column name to find max for
        
    Returns:
        The maximum value in the column
        
    Raises:
        ValueError: If column doesn't exist or contains no numeric data
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if df[column].empty:
        raise ValueError(f"Column '{column}' contains no data")
    return df[column].max()

def filter_by_category(df: pd.DataFrame, column: str, value: str, limit: int = 10) -> pd.DataFrame:
    """
    Filter DataFrame rows by a categorical value.
    
    Args:
        df: The pandas DataFrame
        column: The column name to filter by
        value: The value to match
        limit: Number of rows to return (default: 5)
        
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If column doesn't exist
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    return df[df[column] == value].head(limit)

if __name__ == "__main__":
    # URL for the iris dataset
    IRIS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    
    # Load data
    df = load_data(IRIS_URL)
    
    # Calculate and display average sepal length
    avg_sepal = calculate_column_mean(df, 'sepal_length')
    print(f"Average sepal length: {avg_sepal}")
    
    # Calculate and display max petal width
    max_petal = find_column_max(df, 'petal_width')
    print(f"Max petal width: {max_petal}")
    
    # Filter and display setosa species
    setosa_data = filter_by_category(df, 'species', 'setosa')
    print("\nSetosa species (first 10 rows):")
    print(setosa_data)

"""
Improvements:

Modularity: Code broken into reusable functions (load_data, calculate_column_mean, find_column_max, filter_by_category).

Reusability: Functions work with any DataFrame and column names – easily adapt for other datasets.

Error Handling: Validates column existence, non-empty data, and handles URL loading failures.

Scalability: Can handle multiple CSVs, different column names, and larger datasets without modification.

Documentation: Comprehensive docstrings with Args, Returns, and Raises for each function.

Maintainability: Easy to add new operations (min, median, standard deviation) as separate functions.

Collaboration: Clear structure and documentation make it easy for team members to understand and extend.

Security & Trust: Input validation prevents crashes from unexpected data or missing columns.
"""
