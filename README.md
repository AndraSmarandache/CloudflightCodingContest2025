# CloudLight Coding Contest 2025 - Data/AI Solutions

**Team Darkside: Smarandache Andra + Smarandache Alexandru**

## Overview

Solutions for the CloudLight Coding Contest 2025 Data/AI section. The competition consisted of 6 progressive levels, each requiring different data processing techniques, algorithms, and machine learning approaches.

## Results

- **Local Ranking**: 1st place
- **Global Ranking**: 6th place
- **Total Levels**: 6/6 completed
  
## Competition Structure

### Level 1: BOP Sorting
- **Task**: Sort Bird Observation Points by temperature, humidity, and ID
- **Challenge**: Parse numbers in text format ("twenty", "thirty five")
- **Solution**: Dictionary mapping + multi-criteria sorting

### Level 2: Bird Love Score Prediction
- **Task**: Predict missing Bird Love Scores using ML
- **Approach**: Feature engineering (18 features) + HistGradientBoostingRegressor
- **Techniques**: Feature engineering, gradient boosting, validation
- **Metrics**: RMSE

### Level 3: Bird Species Classification
- **Task**: Classify bird flocks into 6 species based on movement patterns
- **Approach**: Heuristic rules based on structural features
- **Features**: Palindrome detection, LCP, temperature analysis, node patterns

### Level 4: Species Classification with ML
- **Task**: Classify species with partially labeled data
- **Approach**: Hybrid strategy (ML when ≥3 labeled species, heuristics otherwise)
- **Features**: 30+ features per path, flock-level aggregation
- **Metrics**: F1 Score (macro)

### Level 5: Bird Arrivals Forecasting
- **Task**: Predict top 50 BOPs with most arrivals for days 731-760
- **Approach**: Persistence model with occupancy correction
- **Model**: `predicted = arrivals[day-1] × (occupancy[day]/occupancy[day-1])^α`
- **Optimization**: Grid search for optimal α parameter

### Level 6: Temporal Pattern Recognition
- **Task**: Predict top 50 BOPs for day 791
- **Approach**: Identified temporal reversal pattern (days 761-790 are reverse of 760-731)
- **Solution**: Day 791 = Day 730 (pattern continuation)

## Technologies

- **Python 3.13+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models
  - HistGradientBoostingRegressor
  - HistGradientBoostingClassifier
- **Standard Library**: csv, argparse, pathlib, collections

## Project Structure

```
.
├── Documentation.docx
├── level_1/
│   ├── level1.pdf
│   ├── level1.py
│   └── level_1_*.in/out
├── level_2/
│   ├── level2.pdf
│   ├── level2.py
│   └── level_2_*.in/out
├── level_3/
│   ├── level3.pdf
│   ├── level3.py
│   └── level_3_*.in/out
├── level_4/
│   ├── level4.pdf
│   ├── level4.py
│   └── level_4.in/out
├── level_5/
│   ├── level5.pdf
│   ├── level5.py
│   └── level_5.in/out
└── level_6/
    ├── level6.pdf
    ├── level6.py
    └── level_6.in/out
```

## Usage

Each level can be run independently:

```bash
# Level 1
python level_1/level1.py

# Level 2
python level_2/level2.py

# Level 3
python level_3/level3.py

# Level 4
python level_4/level4.py

# Level 5
python level_5/level5.py

# Level 6
python level_6/level6.py
```
