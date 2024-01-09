# Waterpoint Condition EDA Analysis

## Overview
This project involves an exploratory data analysis (EDA) on a dataset related to waterpoints. The dataset contains information about various features related to waterpoints, and the goal is to gain insights into their conditions.

## Dataset
Whole dataset is present inside the data folder in the src.
The dataset used for this analysis includes the following columns:
- `id`: Identification number of the water point.
- `amount_tsh`: The amount of total static head of the water point, TSH is a measure of water available to the water pump.
- ...


## Goals
1. Understand the distribution of key features in the dataset.
2. Identify patterns or trends related to waterpoint conditions.
3. Explore relationships between different features and the target variable.
4. Visualize key insights to facilitate better understanding.

## Tools and Libraries
- Python
- PyCharm
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Steps
1. **Data Loading:** Load the dataset into a Pandas DataFrame.
2. **Data Exploration:** Understand the structure of the dataset, check for missing values, and perform basic summary statistics.
3. **Data Cleaning:** Handle missing values, outliers, and any inconsistencies in the data.
4. **Visualization:** Create visualizations (bar plots, histograms, heatmaps, etc.) to explore feature distributions and relationships.
5. **Feature Engineering:** If necessary, create new features or transform existing ones to extract meaningful information.
6. **Statistical Analysis:** Conduct statistical tests or analyses to validate hypotheses or observations.
7. **Conclusion:** Summarize key findings and insights from the analysis.

## Pipeline
In the pipeline folder, the ***Updated_data function*** is called to store the final preprocessed data.

## Visualization
For relationship between variables, run the ***Plots.py*** file in 
the module folder.

## Test Cases
All the test cases for the used functions are written in the ***Test_cases.py*** file.

## Preprocessing
For data cleaning and preprocessing open the ***data_cleaning.py*** file.

## Working
The project is working on the principle of cleaning the training dataset to make it perfect for performing accurate prediction on the condition of waterpoints.
It first analyses the relationship between each feature and then we filter out the unnecessary feature points in the dataset based on our EDA. At the end, clean dataset is obtained to perform any analysis.

On fitting the data in SVM model, the accuracy came out to be 75%.

## Docker_Support
Dockerfile is added for making an image of whole Assignment-5:

1. Navigate to the directory containing docker file :
   ```bash
   cd pythonProject1/ASSIGNMENT-5
   ```
2. Build the Docker image:
   ```bash
   docker build -t waterpump
   ```
3. Run the Docker container:
   ```bash
   docker run waterpump python main.py

## How to Run
1. Ensure you have Python installed on your machine.
2. Clone this repository: `git clone https://github.com/PrinceA2/ASSIGNMENT_5.git
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the main.py file on your PyCharm.


## Output
The final output with preprocessed data will be visible in the console window.


## Contributors

Prince Tiwari
