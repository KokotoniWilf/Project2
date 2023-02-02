"""


Disaster Reponses - ETL Pipeline script

This script receives 2 csv files containing messages and categories, cleans
the data and returns a SQLite database containing one table of cleaned message
data.
 

"""

# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """ 
 
    Loads 2 datasets and merges them    
    
    Parameters:
        messages_filepath - contains the filepath to the messages.csv file
        categories_filepath - contains the filepath to the categories.csv file
        
    Returns:
        df - dataframe containing the merged datasets
        
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on="id")
    
    return df
 

def clean_data(df):
    
    """
    
    Cleans the merged data by carrying out the following steps:
        Replaces category column with 36 new columns, one for each category
        Creates column headers
        Strips new column names from rows
        Drops duplicates
        
    Parameters:
        df - the merged dataframe created in load_data function
        
    Returns:
        df - cleaned dataframe ready to be loaded into SQLite database
        
    """
    
    # create a dataframe of the individual category columns
    categories = pd.Series(df["categories"]).str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.head(1)
    
    # extract a list of new column names for categories
    category_colnames = row.loc[0, :].values.tolist()
    category_colnames = [x[:-2] for x in category_colnames]
        
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        
        # convert column from string to numeric
        categories[column] =  categories[column].astype(int)
    
    # drop the original categories column from `df`
    
    df.drop(columns='categories',inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
 

def save_data(df, database_filepath):
    
    """
    
    Loads merged and cleaned dataframe into SQLite database for storage
    
    Parameters:
        df - cleaned dataframe produced by clean_data function
        database_filepath - contains the filepath to the SQLite database    
    Returns:
        None
    """
    
    
    # load dataframe to sqlite database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('CategorizedMessages', engine, index=False)
 
    
def main():

    """
    
    Checks system input for file locations and runs the load_data, clean_data 
    and save_data functions if all are present
    
    Parameters:
        None
    Returns:
        None
        
    """
    
    if len(sys.argv) == 4:
        
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'\
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
    else:
        print('Incorrect parameters provided.'\
              '  Please provide input in the following format:\n'\
              'process_data.py messages.csv categories.csv database.db')
            


if __name__ == '__main__':
    main()




