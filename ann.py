# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Pilot_Pred.csv')
X = dataset.iloc[:, 3:12].values #importing from credit score(independent vaiable),
y = dataset.iloc[:, 12].values

# Encoding categorical data #in this we encode the text to nubers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#creating dummy varibales for removing ambiguity
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #in this we will remove one dummy variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler#variables scalaing in the same scale, by standardization or normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #fitting in x_train
X_test = sc.transform(X_test) #X_test ke liye sirf transform karna hai kyunki training mei alerady call hua hai

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras #neural netwok build krne ke liye
from keras.models import Sequential #initialising our ann 
from keras.layers import Dense  #to create the layers

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer for our 10 nodes
#the Stochastic Gradient Descent has started here
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer #for dependent variable 
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test) #predictiond of probabilities on x_test set
y_pred = (y_pred > 0.5) #converting the probabilities in the binary if y_pred is larger than 0.5 then true if smaller than false

# Making the Confusion Matrix and making the predicted results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)