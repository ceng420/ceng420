"""
The Titanic Challenge

"""


"""

The Problem
===========

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden
voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational
tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the
passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people
were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular,
we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
"""

"""
The dataset
===========

The dataset is divided into training data and  test data. We use the training data to build or model (in this case we
are building a binary classifier (which passengers survived and which did not)


"""

# modules required for data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd


"""
    PART ONE: BASIC DATA ANALYSIS
    =============================
"""
# load the data from a csv file using pandas, the data will be stored in a pandas data frame
train_data_frame = pd.read_csv('train.csv')
test_data_frame = pd.read_csv('test.csv')

# Check number of samples in the data:
samples_count_training = len(train_data_frame)
samples_count_test = len(test_data_frame)

print('The number of samples in the training data is {0} samples'.format(samples_count_training))
print('The number of samples in the test data is {0} samples'.format(samples_count_test))


# list the features names using pandas
list_of_features = train_data_frame.columns.values
print('These are the available features \n {0}'.format(list_of_features))

# to check the feature data types
print('The feature data types \n {0}'.format(train_data_frame.dtypes))

# to see a descriptive statistical summary of your (numeric columns) using .describe():
print('The overall data summary \n {0}'.format(train_data_frame.describe()))

# to list the top n samples in your data use the head(n) method and pass to it the number of samples you want to see
print('Here are the first 5 samples {0}'.format(train_data_frame.head(5)))

# to list the last n samples in your data use the tail(n) method and pass to it the number of samples you want to see
print('Here are the last 5 samples {0}'.format(train_data_frame.tail(5)))


# to select a subset from the samples let us say you want to copy the first 20 samples into a new data frame
data_frame_20_samples = train_data_frame[0:20]
print('Here are the first 20 samples {0} in a new data frame'.format(data_frame_20_samples))

# to select a subset of the features (e.g. use less features to describe the samples) note this will be manual feature
# selection. Let say we want to select the gender (sex) and Survived
sub_feature_set = train_data_frame[['Sex', 'Survived']]
print('Here we select only the sex and survived to represent the samples {0}'.format(sub_feature_set))

# to select samples based on specific feature value. Let say we want to sample with Fare price greater than 50.0
samples_fare_price = train_data_frame[train_data_frame['Fare'] > 50.0]
print('Here we select samples where the fare price > 50.0 {0}'.format(samples_fare_price))


# to select a subset of the features (e.g. use less features to describe the samples) and samples.
# Let say we want to select the gender (sex) and Survived but only the samples from index 10 to 20
sub_feature_set = train_data_frame.loc[10:20, ['Sex', 'Survived']]
print('Here we select only the sex and survived to represent the samples {0}'.format(sub_feature_set))


"""
    PART TWO: FEATURE REPRESENTATION
    =============================
"""

# features need to be represented as quantitative values (preferably numeric).
# But how we can handle Categorical features

# load the tennis data set and indicate that the csv delimiter is tab not ','
tennis_data = pd.read_csv('tennis_data.csv', sep='\t')
print(tennis_data)

# use describe method to explore the data, not the data are categorical
print(tennis_data.describe())

# For ordinal features, map the order as increasing integers in a single numeric feature.
# Any entries not found in your designated categories list will be mapped to -1:

# describe the order of an ordinal feature e.g. temperature
ordered_temp = ['cool', 'mild', 'hot']

# use pandas to convert it  to numerical feature
tennis_data['temperature'] = tennis_data['temperature'].astype("category", ordered=True,
                                                               categories=ordered_temp).cat.codes
print(tennis_data)

# for categorical nominal feature we can either encode it using numerical value where the order is not important
# the outlook feature
tennis_data['outlook'] = tennis_data['outlook'].astype("category").cat.codes
print(tennis_data)

# reload the data to overwrite the last feature representation operations
tennis_data = pd.read_csv('tennis_data.csv', sep='\t')

# the other options for categorical nominal feature is to encode them using boolean encoding
tennis_data = pd.get_dummies(tennis_data, columns=['outlook'])
print(tennis_data)



"""
    PART THREE: DATA WRANGLING
    =============================
"""

# It is very common that the collected data contains some noise, abnormal values, or even some missing values.
# Pandas provide some basic functions for data pre-processing and wrangling

# Let go back to the titanic dataset, if we run the describe method we can see that the Age feature is NAN unknown
# for almost 75% of the passengers. How can we deal with that

print(train_data_frame.describe())

# There are several options

# Any time a nan is encountered, replace it with a scalar value:
train_data_frame['Age'].fillna(train_data_frame['Age'].mean())

# or fill it with zero
train_data_frame['Age'].fillna(0)

# or replace it with the immediate, previous, non-nan value form the previous sample
train_data_frame['Age'].fillna(method='ffill', limit=1)

# or replace it with the immediate, next, non-nan value form the previous sample
train_data_frame['Age'].fillna(method='bfill', limit=1)

# or drop all the samples with non value
train_data_frame.dropna(axis=0)

# or drop all the features with non value
train_data_frame.dropna(axis=1)

# Drop any row that has at least 4 NON-NaNs within it:
train_data_frame = train_data_frame.dropna(axis=0, thresh=4)

