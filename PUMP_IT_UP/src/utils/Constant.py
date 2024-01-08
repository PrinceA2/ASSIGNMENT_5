from pathlib import Path

curr_path = Path(__file__).parents[1]
data_folder_var = curr_path / 'data'
x_path_independent = data_folder_var / 'Training_set_values.csv'
y_path_dependent = data_folder_var / 'Training_set_labels.csv'
x_predicted_file_path = data_folder_var / 'Test_set_values.csv'


