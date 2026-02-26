# Week 1: SDLC Principles Applied to Machine Learning

## Overview

This week's activity demonstrates the application of **Software Development Life Cycle (SDLC) principles** to machine learning and data science workflows. The focus is on understanding how clean code, modularity, and best practices improve code quality, maintainability, and scalability from the very beginning of a project.

## Activity Description

### `SDLC_Principles.py`

This script is an **in-class learning activity** that illustrates the transformation of messy, monolithic code into clean, modular, production-ready code by applying fundamental SDLC principles.

#### What the Script Demonstrates

The script contains two main parts:

1. **General Programming Example**: A simple numerical analysis task (generating random numbers, calculating statistics)
   - Shows "Bad" code: All logic in a single block, no functions, no documentation
   - Shows "Good" code: Modular functions, type hints, docstrings, error handling

2. **Classroom Activity - Pandas Data Analysis**: A real-world ML preprocessing example
   - Shows "Bad" approach: Loading and processing data inline without structure
   - Shows "Good" approach: Reusable, well-documented functions with validation

#### Key SDLC Principles Illustrated

| Principle | What It Means | Why It Matters for ML |
|-----------|--------------|----------------------|
| **Modularity** | Code is broken into small, focused functions | Easier to test, debug, and reuse in different ML pipelines |
| **Reusability** | Functions can be used across multiple projects | Saves time in model development and reduces bugs |
| **Readability & Documentation** | Clear code with docstrings and comments | Essential for collaboration in data science teams |
| **Error Handling** | Gracefully manages edge cases and invalid inputs | Prevents silent failures in production ML systems |
| **Type Hints** | Explicit parameter and return types | Improves code clarity and enables better IDE support |
| **Scalability** | Code can handle growing datasets and complexity | Critical for transitioning from experiments to production ML |

## Connection to ML Lifecycle

In machine learning projects, the SDLC principles ensure:

- **Data Preparation Phase**: Well-structured data loading and preprocessing functions that can be reused.
- **Model Development**: Clean separation between data handling, feature engineering, and model training.
- **Experiment Tracking**: Modular code makes it easier to run multiple experiments and compare results.
- **Production Deployment**: Production-ready ML code requires the same quality standards as any software project.
- **Maintenance & Updates**: When models need retraining or feature updates, clean code makes these changes manageable.

## Learning Outcomes

By studying this activity, you will:

- Understand how to transform exploratory code into production-quality code.
- Learn to apply SDLC principles within ML/data science workflows.  
- Write Python functions with proper documentation and error handling.  
- Recognize the importance of code quality in ML projects.  
- Appreciate how clean code reduces technical debt in ML systems.  

## How to Use This File

1. **Read through the script** to understand the progression from messy to clean code.
2. **Study the comments and docstrings** that explain each principle.
3. **Try refactoring the "messy" examples** in your own code.
4. **Apply these patterns** to your own ML projects going forward.

---

**Note**: This is an educational resource demonstrating best practices. The examples use public datasets and are designed for learning purposes.
