# Financial Distress Dataset Samples

## Overview

This repository contains sample data from a temporal dataset designed for the early prediction of financial distress among Chinese listed companies. The dataset captures dynamic financial indicators over time, enabling the modeling of temporal relationships and trends inherent in financial metrics. **The complete dataset will be made publicly available at a later stage.**

## Data Files

### `step_samples.csv`

- **Description**: Contains 50 rows of single time-step sample data, selected at intervals of 3, 6, 9, 12, and 15 months prior to the final prediction point. Each interval includes 10 samples, with an equal split between financial distress samples (label `1`) and non-financial distress samples (label `0`).

### `sequence_samples.pkl`

- **Description**: Contains sequences composed of 12 or more time steps of financial ratio data. A balanced set of sequences is included, with equal numbers of labels `0` (non-distress) and `1` (financial distress).

## Data Description

The dataset includes 35 financial ratios calculated from financial statements. These ratios cover various aspects of a company's financial health and are essential features for machine learning models aimed at predicting financial distress.

The financial ratios (columns starting with F0...) are defined as follows:

| **Column**       | **Description**                                     | **Formula**                                                                                         |
|-------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| F030101A         | Current Assets Ratio                                | Total Current Assets ÷ Total Assets                                                               |
| F030201A         | Cash Assets Ratio                                   | Ending Balance of Cash and Cash Equivalents ÷ Total Assets                                         |
| F030301A         | Receivables Assets Ratio                            | (Net Notes Receivable + Net Accounts Receivable) ÷ Total Assets                                    |
| F030401A         | Working Capital to Current Assets Ratio             | (Total Current Assets - Total Current Liabilities) ÷ Total Current Assets                          |
| F030501A         | Working Capital Ratio                               | (Current Assets - Current Liabilities) ÷ Total Assets                                              |
| F030601A         | Working Capital to Net Assets Ratio                 | Working Capital ÷ Net Assets                                                                       |
| F030701A         | Non-Current Assets Ratio                            | Non-Current Assets ÷ Total Assets                                                                 |
| F030801A         | Fixed Assets Ratio                                  | Net Fixed Assets ÷ Total Assets                                                                    |
| F030901A         | Intangible Assets Ratio                             | Net Intangible Assets ÷ Total Assets                                                              |
| F031001A         | Tangible Assets Ratio                               | Total Tangible Assets ÷ Total Assets                                                              |
| F031101A         | Shareholders' Equity Ratio                          | Total Shareholders' Equity ÷ Total Assets                                                         |
| F031201A         | Retained Earnings to Assets Ratio                   | (Surplus Reserve + Undistributed Profits) ÷ Total Assets                                           |
| F031301A         | Long-term Assets Fit Ratio                          | (Total Equity + Non-Current Liabilities) ÷ (Net Fixed Assets + Net Available-for-Sale Financial Assets + Net Held-to-Maturity Investments + Net Long-term Equity Investments) |
| F031401A         | Shareholders' Equity to Fixed Assets Ratio          | Shareholders' Equity ÷ Net Fixed Assets                                                           |
| F031501A         | Current Liabilities Ratio                           | Total Current Liabilities ÷ Total Liabilities                                                     |
| F031601A         | Operating Liabilities Ratio                         | (Total Current Liabilities - Short-term Borrowings - Non-current Liabilities due within One Year - Trading Financial Liabilities - Derivative Financial Liabilities) ÷ Total Liabilities |
| F031701A         | Financial Liabilities Ratio                         | (Non-current Liabilities + Short-term Borrowings + Non-current Liabilities due within One Year + Trading Financial Liabilities + Derivative Financial Liabilities) ÷ Total Liabilities |
| F031801A         | Non-Current Liabilities Ratio                       | Non-Current Liabilities ÷ Total Liabilities                                                       |
| F031901A         | Parent Company Shareholders' Equity Ratio           | Total Equity Attributable to Parent Company ÷ Total Shareholders' Equity                          |
| F032001A         | Minority Shareholders' Equity Ratio                 | Minority Shareholders' Equity ÷ Total Shareholders' Equity                                         |
| F032101B         | Main Business Profit Ratio                          | (Operating Income - Operating Costs) ÷ Total Profit                                               |
| F032201B         | Financial Activities Profit Ratio                   | (Investment Income + Fair Value Change Income + Exchange Income) ÷ Total Profit                   |
| F032301B         | Operating Profit Ratio                              | Operating Profit ÷ Total Profit                                                                   |
| F032401B         | Non-operating Income Ratio                          | (Non-operating Income - Non-operating Expenses) ÷ Total Profit                                    |
| F032501B         | Turnover Tax Rate                                   | Business Taxes and Surcharges ÷ Total Operating Income                                             |
| F032601B         | Comprehensive Tax Rate A                            | (Business Taxes and Surcharges + Income Tax Expense) ÷ Total Operating Income                     |
| F032701B         | Comprehensive Tax Rate B                            | (Business Taxes and Surcharges + Income Tax Expense) ÷ Total Profit                               |
| F032801B         | Income Tax Rate                                     | Income Tax Expense ÷ Total Profit                                                                 |
| F032901B         | Net Profit Attributable to Parent Company Ratio     | Net Profit Attributable to Parent Company ÷ Net Profit                                             |
| F033001B         | Net Profit Attributable to Minority Shareholders Ratio | Net Profit Attributable to Minority Shareholders ÷ Net Profit                                    |
| F033101B         | Net Profit to Comprehensive Income Ratio            | Net Profit ÷ Total Comprehensive Income                                                           |
| F033201B         | Other Comprehensive Income Ratio                    | Other Comprehensive Income ÷ Total Comprehensive Income                                           |
| F033301B         | Comprehensive Income Attributable to Parent Company Ratio | Total Comprehensive Income Attributable to Parent Company ÷ Total Comprehensive Income        |
| F033401B         | Comprehensive Income Attributable to Minority Shareholders Ratio | Total Comprehensive Income Attributable to Minority Shareholders ÷ Total Comprehensive Income |
| F033501A         | Parent Company Equity to Invested Capital Ratio     | (Total Equity Attributable to Parent Company) ÷ (Total Assets - Total Current Liabilities + Notes Payable + Short-term Borrowings + Non-current Liabilities due within One Year) |



## Usage

### Code to Read and Display `step_samples.csv`

```python
import pandas as pd

# Read the step_samples.csv file
step_samples = pd.read_csv('step_samples.csv')

# Display the first few rows of the data
print("Step Samples:")
print(step_samples.head())

# If you want to display all rows, use:
# print(step_samples)
```

### Code to Read and Display `sequence_samples.pkl`

```python
import pickle
import numpy as np

# Load the sequence_samples.pkl file
with open('sequence_samples.pkl', 'rb') as f:
    sequence_samples = pickle.load(f)

# Display the first few samples
print("Sequence Samples:")
for i, (sequence_data, label) in enumerate(sequence_samples[:5]):
    print(f"Sample {i + 1}:")
    print(f"Sequence data shape: {sequence_data.shape}")
    print("Sequence data:")
    print(sequence_data)
    print(f"Label: {label}")
    print("-" * 50)
```

## Notes

- The data provided is for research and educational purposes.
- Labels:
  - `0`: Non-financial distress.
  - `1`: Financial distress.
- Ensure you have the necessary packages (`pandas`, `numpy`, `pickle`) installed to run the code snippets.

## Citation

If you use this dataset in your research, please cite accordingly.

## License

This project is licensed under the MIT License.
