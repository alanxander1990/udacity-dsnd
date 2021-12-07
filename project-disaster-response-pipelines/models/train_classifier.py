# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

# import statements
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import sys


def load_data(database_filepath='..\data\DisasterResponse.db'):
    """
    - Read dataframe from database file
    
    -INPUT
        database_filepath - path to database file
    
    -OUTPUT
        df - pandas dataframe for ML pipeline
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_query('select * from disaster_response', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    
    # mapping extra values to `1` since Y['related'] contains three distinct values
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    
    # get category columns names
    category_names = Y.columns
    
    # convert all category data value into number
#    for cat in category_names:
#        Y[cat] = pd.to_numeric(Y[cat])

    return X, Y, category_names

def tokenize(text):
    """
    - Tokenizes text data
    
    -INPUT
        text - Messages as text data
    
    -OUTPUT
        list - Processed text after normalizing, tokenizing and lemmatizing
    """
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # part of speech tagging
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words


def build_model():
    """
    Build model with GridSearchCV
    
    -OUTPUT
        Trained model after performing grid search
    """
    # model pipeline
#    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
#                         ('tfidf', TfidfTransformer()),
#                         ('clf', MultiOutputClassifier(
#                            OneVsRestClassifier(SVC())))]
#                         )
#
#    # specify parameters for grid search
#    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
#                  'vect__max_df': (0.75, 1.0)
#                  }
#
#    # create grid search object
#    # create model
#    model = GridSearchCV(
#            estimator = pipeline,
#            param_grid = parameters,
#            verbose = 3,
#            cv = 3,
#            n_jobs = -1
#    )
    
     # Create a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # Create Grid search parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 60, 70]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv = 3, n_jobs = -1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Shows model's performance on test data
    
    -INPUT
        model - trained model
        X_test - Test features
        Y_test - Test targets
        category_names - Target labels
    """
    # predict on test data
    Y_pred = model.predict(X_test)
    
    # print classification report
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], Y_pred[:, i]))
        i = i + 1
    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == Y_pred)))


def save_model(model, model_filepath='classifier.pkl'):
    """
    Saves the model to a Python pickle file    
    
    -INPUT
        model - Trained model
        model_filepath - Filepath to save the model
    """

    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    print('len(sys.argv): {}'.format(len(sys.argv)))
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
    elif len(sys.argv) == 1:
        print('Loading data with default path...')
        X, Y, category_names = load_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print('Building model...DONE')
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model with default path...')
        save_model(model)

        print('Trained model saved!')
        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# code for testing
#from sklearn.externals import joblib
#
#    engine = create_engine('sqlite:///{}'.format('..\data\DisasterResponse.db'))
#    df = pd.read_sql_query('select * from disaster_response', engine)
#    
#    model = joblib.load("classifier.pkl")
#    query = 'BMA, moreover, in cooperation with the Department of Highway (DOH) to increase the height of a 35-kilometer-long section of the embarkment along Rom Klao Road by 3 meters.'
#
#    # use model to predict classification for query
#    print('use model to predict classification for query: {}'.format(query))
#    classification_labels = model.predict([query])[0]
#    classification_results = dict(zip(df.columns[4:], classification_labels))    