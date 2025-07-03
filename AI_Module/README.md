# AI Module for Healthcare Fraud Detection

This module provides a complete workflow for generating synthetic healthcare claims data and training a machine learning model to detect fraudulent claims.

## Getting Started

This guide will walk you through setting up the environment and running the module from start to finish.

### Step 1: Install `uv` (if you don't have it)

This project uses `uv` for Python package management. It's a fast and modern alternative to `pip` and `venv`. If you don't have it installed, you can install it with `pip`:

```bash
pip install uv
```

### Step 2: Create the Virtual Environment

Once `uv` is installed, navigate to the `AI_Module` directory in your terminal and run the following command to create a virtual environment and install all the required packages:

```bash
uv venv && uv pip install -r requirements.txt
```

This command does two things:
1.  `uv venv`: Creates a new virtual environment in a `.venv` folder.
2.  `uv pip install -r requirements.txt`: Installs all the necessary libraries listed in the `requirements.txt` file into this new environment.

### Step 3: Generate the Synthetic Data

Before you can train the model, you need to generate the dataset. Run the following command in your terminal:

```bash
python generate_data.py
```

This will create a file named `healthcare_claims_complete.csv` in the current directory. This file contains 10,000 synthetic healthcare claims, including a mix of legitimate and fraudulent cases.

### Step 4: Train the Model and Generate the Report

Now that you have the data, you can train the fraud detection model. Run the following command:

```bash
python train_model.py
```

This script will perform the following actions:
1.  **Load the Data:** It loads the `healthcare_claims_complete.csv` file.
2.  **Preprocess the Data:** It cleans the data and creates new, meaningful features for the model.
3.  **Train the Model:** It trains a powerful XGBoost classifier on 80% of the data.
4.  **Evaluate the Model:** It tests the model on the remaining 20% of the data and prints a detailed evaluation report, including accuracy, F1 score, and a confusion matrix.
5.  **Generate Red Flag Report:** It creates a `red_flag_report.csv` file. This file contains all the claims from the test set that the model predicted as fraudulent, sorted by the likelihood of fraud in descending order. This report is the primary output for an analyst to review.
6.  **Save the Model:** It saves the trained model to a `.joblib` file for future use.
7.  **Log Everything:** All the output from the script is saved to a `fraud_detection.log` file for debugging and record-keeping.

## Understanding the Output

After running the `train_model.py` script, you will have two key outputs:

*   **`red_flag_report.csv`**: This is the main result. It's a CSV file containing the claims that are most likely to be fraudulent. The `FraudProbability` column gives you the model's confidence in its prediction, allowing you to prioritize the most suspicious cases first.
*   **`fraud_detection.log`**: This file contains a complete log of the entire process, including timings for each step. If you encounter any errors, this file will be the first place to look for clues.
*   **`fraud_detection_model_*.joblib`**: This file contains the trained model. You can use this file to make predictions on new data without having to retrain the model every time.
