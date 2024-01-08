import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils.Constant import x_path_independent, y_path_dependent, x_predicted_file_path


# Loading the dataset
x_train = pd.read_csv(x_path_independent)
y_data = pd.read_csv(y_path_dependent)
x_test = pd.read_csv(x_predicted_file_path)


# Function to filter out unnecessary columns
def drop_columns(dataframe, columns_to_drop):

    return dataframe.drop(columns=columns_to_drop, errors='ignore')


# Function to drop the null values in a data_frame column
def drop_null_values(dataframe, column_name):
    return dataframe.dropna(subset=[column_name])


# Function to label_encode a column values
def label_encode_column(dataset, column_name):
    # Make a copy of the dataset to avoid modifying the original
    encoded_dataset = dataset.copy()
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    # Fit and transform the specified column in the copied dataset
    encoded_dataset[column_name] = label_encoder.fit_transform(encoded_dataset[column_name])
    return encoded_dataset


# Function for creating the location feature
def creating_location_feature(dataframe):

    return np.sqrt(x_train["latitude"] ** 2 + x_train["longitude"] ** 2)



# Function to calculate the age of water pump by combining date_recorded and construction_year.
def calculate_age_in_years(df, date_column, construction_year_column, new_column_name):

    df[date_column] = pd.to_datetime(df[date_column])
    # Extract the year from date_column
    df[date_column] = pd.DatetimeIndex(df[date_column]).year
    # Impute null values in construction_year_column with the median value
    df[construction_year_column].mask(df[construction_year_column] == 0, df[construction_year_column].median(),inplace=True)
    # Calculate 'age_in_years'
    df[new_column_name] = df[date_column] - df[construction_year_column]
    return df


# Function to replace 0 with mean in a column
def replace_zeros_with_mean(df, column_name):
    mean_value = df[column_name].replace(0, pd.NA).mean(skipna=True)
    df[column_name] = df[column_name].mask(df[column_name] == 0, mean_value).astype(int)
    return df



# Function to count zeros in a data_frame column
def count_zeros_in_column(df, column_name):

    return (df[column_name] == 0).sum()


# Determining unnecessary columns

refine_columns = ['funder','wpt_name','gps_height','amount_tsh','id','num_private','scheme_name',
                  'recorded_by' ,'management_group','quantity_group','subvillage',
                  'payment_type','source_type','lga','basin',
                  'extraction_type_group','extraction_type_class','ward',
                  'scheme_management','region','installer', 'wpt_name',
                  'district_code', 'quality_group','waterpoint_type_group']



# Merging the training_unlabelled data with training_labelled data.
x_train = pd.merge(x_train,y_data)


# Dropping columns in both training and test data.
x_train = drop_columns(x_train, refine_columns)
x_test = drop_columns(x_test, refine_columns)


# Latitude and longitude are combined into a single feature named location
x_train['location'] = creating_location_feature(x_train)
x_test['location'] = creating_location_feature(x_test)


# Dropping the latitude and longitude feature from x_train and x_test
dropping_step = ['latitude','longitude']
x_train = drop_columns(x_train, dropping_step)
x_test = drop_columns(x_test, dropping_step)


# Calculating the age of water_point.
x_train = calculate_age_in_years(x_train,'date_recorded', 'construction_year', 'age_in_years')
x_test = calculate_age_in_years(x_test,'date_recorded', 'construction_year', 'age_in_years')



# Dropping the date_recorded and construction_year column
dropping_step = ['date_recorded','construction_year']
x_train = drop_columns(x_train, dropping_step)
x_test = drop_columns(x_test, dropping_step)


# Replacing each 0's value in population with mean ....
x_train = replace_zeros_with_mean(x_train, 'population')
x_test = replace_zeros_with_mean(x_test, 'population')


# Dropping the n/a values in public and permit
x_train= drop_null_values(x_train, 'public_meeting')
x_test = drop_null_values(x_test, 'public_meeting')
x_train = drop_null_values(x_train, 'permit')
x_test = drop_null_values(x_test, 'permit')

# Label encoding the public meeting and permit
x_train = label_encode_column(x_train, 'public_meeting')
x_test = label_encode_column(x_test, 'public_meeting')
x_train = label_encode_column(x_train, 'permit')
x_test = label_encode_column(x_test, 'permit')


# Label_encoding the quantity feature values and storing in a new_column
x_train['quantity_enc'] = x_train['quantity'].apply(lambda x:["unknown","dry","insufficient","seasonal","enough"].index(x))
x_test['quantity_enc'] = x_test['quantity'].apply(lambda x:["unknown","dry","insufficient","seasonal","enough"].index(x))


# Dropping the quantity column
x_train = drop_columns(x_train, 'quantity')
x_test = drop_columns(x_test, 'quantity')


# Label_encoding status_group or predicting column.
x_train = label_encode_column(x_train, 'status_group')


# Total columns remaining that can be label encoded
to_be_categorized = ['extraction_type','management','water_quality','source','source_class','payment','waterpoint_type']

# iterating over every column to be encoded
for i in to_be_categorized:
    x_train = label_encode_column(x_train, i)
    x_test = label_encode_column(x_test, i)

def Updated_data():
    return x_train,x_test