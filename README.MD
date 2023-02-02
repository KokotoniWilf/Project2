# UDACITY Data Scientist Nanodegree

## Project 2 - Disaster Response Pipeline


## Installations

This project requires Python 3.x and the following additional python libraries:

For ETL Pipeline script (process_data.py):
```python
import pandas as pd
import sys
from sqlalchemy import create_engine
```

For Machine Learning Pipeline script (train_classifier.py):
```python
import nltk
nltk.download(['punkt', 'wordnet'])
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
```

For Web App script (run.py):
```python
import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
```


## Project Motivation

The aim of this project is to create a tool that can classify messages in the event of a disaster, to help disaster relief workers more easily identify important messages.

The process used is to receive two csv files containing messages and categories related to disasters, clean the messages and then use a machine learning algorith to train a classification model.  
This model will then classify the messages and feed a web app that will allow the end user to enter their own messages and have them classified accordingly.  
The app includes visualisations showing the distribution of genres and categories of the messages stored in the message database.


## File Descriptions

The files related to this project are as follows:

This README.md file, which contains information relating to the project aims and system requirements.

- app (folder)
	- templates (folder)
		- master.html  			# main page of web app
		- go.html  				# classification result page of web app
	- run.py  					# Flask file that runs app

- data (folder)
	- disaster_categories.csv	# data to process 
	- disaster_messages.csv		# data to process
	- process_data.py			# data cleaning pipeline script
	- DisasterResponse.db   	# database to save clean data to

- models (folder)
	- train_classifier.py		# machine learning pipeline script
	- classifier.pkl  			# saved model (too large to upload to GitHub)


## How to interact with the project

Execute the following commands (the first 2 steps should be run in the project root directory):

1. To run ETL pipeline: ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
2. To run ML pipeline: ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
3. Move to the app directory and then execute the following command to run the web app: ```python app.py```
4. Go to http://0.0.0.0:3000/
5. Messages can be entered into the web app, use the "Classify Message" button to show how the message is classified.


## Licensing, Authors, Acknowledgements, etc

The template code for the web app was provided by Udacity.
