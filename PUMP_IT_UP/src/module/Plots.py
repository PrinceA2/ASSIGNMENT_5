import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from src.utils.Constant import x_path_independent, y_path_dependent
from src.preprocessing.data_cleaning import label_encode_column


x_train = pd.read_csv(x_path_independent)
y_data = pd.read_csv(y_path_dependent)
x_train = pd.merge(x_train,y_data)

# Heat_map for all numerical features.
def plot_correlation_heatmap(dataframe):

    # Calculate the correlation matrix
    correlation_matrix = dataframe.corr()

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()

def filter_numerical_features(dataframe):
    numerical_features = dataframe.select_dtypes(include=['number'])
    return numerical_features


# Filter out numerical features using the function
numerical_features_df = filter_numerical_features(x_train)

# Display the resulting DataFrame
print(numerical_features_df)



# Count plot for each feature in the training dataset.
def count_plot_between_features(dataframe, feature1, feature2):

    sns.countplot(x=feature1, hue=feature2, data=dataframe)
    plt.show()

count_plot_between_features(x_train, 'permit', 'status_group')
count_plot_between_features(x_train, 'public_meeting', 'status_group')
count_plot_between_features(x_train, 'scheme_management', 'status_group')
count_plot_between_features(x_train, 'waterpoint_type', 'status_group')
count_plot_between_features(x_train, 'water_quality', 'status_group')
count_plot_between_features(x_train, 'quantity', 'status_group')
count_plot_between_features(x_train, 'payment_type', 'status_group')
count_plot_between_features(x_train, 'extraction_type', 'status_group')
count_plot_between_features(x_train, 'source_type', 'status_group')
count_plot_between_features(x_train, 'source_type', 'status_group')

