"""


Disaster Reponses - Machine Learning Pipeline Script

This script takes data from a provided database and trains a classifier model 
to be able to classify messages according to a number of different categories.
Tis model is then packaged as a Pickle file for use in a web app.


"""

import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import pandas as pd
import re
import pickle
import sys
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")



def load_data(database_filepath):
    
    """
    
    Loads data from SQLite database
    
    Parameters:
        database_filepath - contains the filepath to the SQLite database
      
    Returns:
        X - DataFrame containing only the messages
        y - DataFrame containing all other columns except message, and other
            columns dropped as not relevant
        
    """
    
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('CategorizedMessages', engine)
    
    # define feature and target variables X and y
    X = df['message']
    y = df.drop(['message','original','genre','id'], axis=1)
    
    return X,y



def tokenize(text):
    
    """
    
    Tokenizes and lemmatizes text provided as input, returning a list of tokens
    URLs are also replaced with placeholder text
    
    Parameters:
        text - text to be tokenized
      
    Returns:
        clean_tokens - list of tokens that have been lemmatized, 
            moved to lower case and stripped of whitespace.  
            URLs have been replaced with placeholder text
        
    """
    
    # Define url pattern
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # find urls and replace them with placeholder text
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenise the words
    tokens = word_tokenize(text)

    # lemmatize words, convert all to lower case and strip whitespace
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    # return list of cleaned tokens
    return clean_tokens



def build_model():
    
    """
    
    Builds the NPL pipeline and carries out a grid search to improve the model
    
    Parameters:
        None
    
    Returns:
        cv - Classifier object
    
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    # specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    # carry out grid search
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv



def train(X,y,model):
  
    """
    
    Creates the split between training and test data
    Fits the model using the training data
    Returns the trained model along with the test datasets
    
    Parameters:
        X - message DataFrame created in load_data function
        y - DataFrame of other columns created in load_data function
        model - the ML model built in the build_model function
        
    Returns:
        model - model trained on the training datasets
        X_test - DataFrame containing test messages
        y_test - DataFrame containing test variables
    
    """
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
      
    return model, X_test, y_test

 
  
def evaluate_model(model, X_test, y_test):

    """
    
    Evaluates the model and produces results for each column in a dataframe
    Results include an f1 score, precision score and a recall score for each
    category in the dataset
    
    Parameters:
        model - the model to be evaluated
        X_test - DataFrame containing test messages
        y_test - DataFrame containing test variables
        
    Returns:
        results_df - DataFrame containing results of model testing
    
    """
    
    y_pred = model.predict(X_test)

    results = []
    for index, column in enumerate(y_test):
        precision, recall, f1, _ = precision_recall_fscore_support(y_test[column], y_pred[:, index], 
                                                              average='micro')
        results.append([f1, precision, recall])

    results_df = pd.DataFrame(results, columns=['f1 score', 'precision', 'recall'],
                                    index = y_test.columns)
    
    return results_df

   

def export_model(model, model_filepath):
    
    """
    
    Exports the final model as a pickle file
    
    Parameters:
        model - the final model to be exported
        
    Returns:
        None
    
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))
    
    

def main():

    """

    Loads data from database, builds and trains the model, evaluates the model 
    and produces results data, then exports the final model as a pickle file

    Parameters:
        None
    
    Returns:
        None

    """    
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Beginning process\n')
        print('Loading data from Database: {}...\n'.format(database_filepath))
        X, y = load_data(database_filepath) 
        
        print('Building model...\n')
        model = build_model() 
        
        print('Training model (note that this step may take up to 1 hour to complete)...')
        model, X_test, y_test = train(X, y, model)  
        
        print('Evaluating model...\n')
        results_df = evaluate_model(model, X_test, y_test)
        print('Evaluation completed.  Test results are as follows:\n')
        print(results_df)
        print('\n')
        
        print('Exporting model to: {}...\n'.format(model_filepath))
        export_model(model, model_filepath)  
        
        print('Process Complete')

    else:
        print('Incorrect parameters provided.'\
              '  Please provide input in the following format:\n'\
              'train_classifier.py database.db classifier.pkl')



if __name__ == '__main__':
    main()  