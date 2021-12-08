import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_file_path='disaster_messages.csv', categories_file_path='disaster_categories.csv'):
    """
    - Reads in 2 csv files and convert into pandas dataframe.
    - Merges them into one dataframe
    
    -INPUT
        messages_file_path - path to disaster_messages.csv
        categories_file_path - path to disaster_categories.csv
    
    -OUTPUT
        merged_df - pandas dataframe merged from the two input data
    """

    # load messages dataset
    messages = pd.read_csv(messages_file_path, encoding='latin-1')
    # load categories dataset
    categories = pd.read_csv(categories_file_path, encoding='latin-1')
    
    # merge datasets based on their common columns: id
    merged_df = messages.merge(categories, how='outer', on=['id'])
    
    return merged_df

def clean_data(df):
    """
    - Cleans the dataframe for ML pipeline
    
    -INPUT
        df - Merged dataframe returned from load_data()
    -OUTPUT
        df - Cleaned data to be used by ML pipeline
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0].values
    
    # use this row to extract a list of new column names for categories
    category_colnames = [r[:-2] for r in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # filter for columns contains value other than 0 and 1    
    columns_non_binary = [col for col in categories if not np.isin(categories[col].unique(), [0, 1]).all()]
    
    # mapping extra values to `1`, in this case column 'related' contains three distinct values
    for col in columns_non_binary:
        categories[col] = categories[col].map(lambda x: 1 if (x!=0 and x!=1) else x)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df[categories.columns] = categories
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_file_name='DisasterResponse.db'):
    """
    - Save dataframe into a database table and persist the database on file system
    
    -INPUT
        df - Merged dataframe returned from load_data()
        database_file_name - Path to the database file to be saved
    -OUTPUT
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_file_name))
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    elif len(sys.argv) == 1:
        print('Loading data using default param values')
        df = load_data()
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format('DisasterResponse.db'))
        save_data(df)
        
        print('Cleaned data saved to database!')
        
    else:   
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'messages.csv categories.csv '\
              'DisasterResponse.db')


# run
if __name__ == '__main__':
    main()    
    
    