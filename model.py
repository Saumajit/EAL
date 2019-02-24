#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import sys
import csv

csv.field_size_limit(sys.maxsize)



import random

import numpy as np
import nltk
import codecs
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold

from keras.layers import Bidirectional
from keras.layers import LSTM

from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


'''
model = KeyedVectors.load_word2vec_format('cc.hi.300.vec')
'''


#creating position embedding matrix
pos=np.random.rand(1000,50)


model=np.load('total_dataset_fasttext_7jan.npy').item()			#loading the dictionary consisting of all the mentioned words in the training dataset along with its word embedding vector.

#sentence embedding
ml=100			#maximum length of a sentence
rd=open("hindi_pos_train_9jan.csv","r")			#reading the training dataset
read=csv.reader(rd,delimiter='\t')
inp=[]
or_sent=[]
pair=[]

label=[]
for row in read:
  line=row[0]
  ev = row[2]
  arg = row[3]
  
  ev=ev.strip('[')
  ev=ev.strip(']')
  ev=ev.split(',')
  
  arg=arg.strip('[')
  arg=arg.strip(']')
  arg=arg.split(',')

  c=row[4]
  c=int(c)
  label.append(c)		#storing the label for each instance
  or_sent.append(line)		#storing the instance
  pair.append(row[1])		#storing the event-argument pair mentioned in the instance
  line=line.rstrip()
  l=line.split(' ')		#tokenizing the instance
  s_v=[]		#embedding for each sentence
  i=0
  if(len(l)< ml):						#if length of the instance is less than the maximum length
  	while(i<len(l)):
    		temp=[]
    		ev[i]=ev[i].strip()
    		arg[i]=arg[i].strip()
   	 	if(ev[i]==' None' or ev[i]=='None'):		#if there exists event in the form of a phrase
      			pos_e=0
    		else:						#if there exists event in the form of a single word
    			a=int(ev[i])
      			pos_e=a
      
      
    		if(arg[i]==' None' or arg[i]=='None'):		#if there exists argument in the form of a phrase
      			pos_a=0
    		else:						#if there exists argument in the form of a single word
      			b=int(arg[i])  
      			pos_a=b
      
      
    		if l[i] in model:				#if the word lies in the dictionary
      			temp.extend(model[l[i]])
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		elif l[i].lower() in model:			#if the lower case of the word lies in the dictionary
      			temp.extend(model[l[i].lower()])
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		else:						#if the word is not present in the dictionary(Out of Vocabulary word)
      			temp.extend([0]*300)
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		i=i+1
    		s_v.append(temp)
  
  
  	while(i<ml):             				#padding shorter sentences
   		s_v.append([0]*400)
    		i=i+1
    
  else:								#if the length of the instance is greater than the maximum length
	
	while(i<ml):						#truncating longer instances till the maximum length						
    		temp=[]
    		ev[i]=ev[i].strip()
    		arg[i]=arg[i].strip()
   	 	if(ev[i]==' None' or ev[i]=='None'):
      			pos_e=0
    		else:
    			a=int(ev[i])
      			pos_e=a
      
      
    		if(arg[i]==' None' or arg[i]=='None'):
      			pos_a=0
    		else:
      			b=int(arg[i])  
      			pos_a=b
      
      
    		if l[i] in model:
      			temp.extend(model[l[i]])
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		elif l[i].lower() in model:
      			temp.extend(model[l[i].lower()])
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		else:
      			temp.extend([0]*300)
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		i=i+1
    		s_v.append(temp)
  
  
  inp.append(s_v)

rd.close()
print "Total number of training instances ",len(or_sent)			#printing the total number of instances present in the training dataset


rd=open("hindi_pos_test_9jan.csv","r")				#loading the test dataset
read=csv.reader(rd,delimiter='\t')
inp_test=[]
or_sent_test=[]
pair=[]
ev_type=[]
arg_type=[]
trend=[]
test_label=[]
for row in read:
  line=row[0]			#storing the instance
  ev = row[2]			#storing the relative postion of each word from event
  arg = row[3]			#storing the relative position of each word from argument
  
  ev=ev.strip('[')
  ev=ev.strip(']')
  ev=ev.split(',')
  
  arg=arg.strip('[')
  arg=arg.strip(']')
  arg=arg.split(',')

  c=row[4]
  c=int(c)
  test_label.append(c)		#storing the label

  or_sent_test.append(line)
  pair.append(row[1])		#storing the event-argument pair
  ev_type.append(row[5])
  arg_type.append(row[6])
  trend.append(row[7])
  line=line.rstrip()
  l=line.split(' ')
  s_v=[]				#embedding for each sentence
  i=0
  if(len(l)< ml):						#if length of the instance is less than the maximum length	
  	while(i<len(l)):
    		temp=[]
    		ev[i]=ev[i].strip()
    		arg[i]=arg[i].strip()
   	 	if(ev[i]==' None' or ev[i]=='None'):		#if there exists event in the form of a phrase
      			pos_e=0
    		else:						#if there exists event in the form of a single word
    			a=int(ev[i])
      			pos_e=a
      
      
    		if(arg[i]==' None' or arg[i]=='None'):		#if there exists argument in the form of a phrase
      			pos_a=0
    		else:						#if there exists event in the form of a single word
      			b=int(arg[i])  
      			pos_a=b
      
      
    		if l[i] in model:				#if the word lies in the dictionary
      			temp.extend(model[l[i]])
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		elif l[i].lower() in model:			#if the lower case of the word lies in the dictionary
      			temp.extend(model[l[i].lower()])
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		else:						#if the word does not lie in the dictionary(Out of Vocabulary words)
      			temp.extend([0]*300)
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		i=i+1
    		s_v.append(temp)
  
  
  	while(i<ml):             					#padding shorter sentences
   		s_v.append([0]*400)
    		i=i+1
    
  else:								#if the length of the instance is greater than the maximum length
	
	while(i<ml):						#truncating the instance to maximum length
    		temp=[]
    		ev[i]=ev[i].strip()
    		arg[i]=arg[i].strip()
   	 	if(ev[i]==' None' or ev[i]=='None'):
      			pos_e=0
    		else:
    			a=int(ev[i])
      			pos_e=a
      
      
    		if(arg[i]==' None' or arg[i]=='None'):
      			pos_a=0
    		else:
      			b=int(arg[i])  
      			pos_a=b
      
      
    		if l[i] in model:
      			temp.extend(model[l[i]])
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		elif l[i].lower() in model:
      			temp.extend(model[l[i].lower()])
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		else:
      			temp.extend([0]*300)
      			temp.extend(pos[ml+pos_e])
      			temp.extend(pos[ml+pos_a])
    		i=i+1
    		s_v.append(temp)
  
  
  	
  
  inp_test.append(s_v)

print "Total number of test instances ",len(or_sent_test)






inp=np.reshape(inp, (len(or_sent),ml,400))				#shaping the training data
inp_test=np.reshape(inp_test, (len(or_sent_test),ml,400))		#shaping the test data
label=np.array(label)							#shaping the label of the training data
test_label=np.array(test_label)						#shaping the label of the test data


X_train=inp
y_train=label
X_test=inp_test
y_test=test_label





filter_sizes = [2,3,4]			#Filter size
num_filters = 64			#Number of filters
drop = 0.50				#dropout
sequence_length=ml			#length of each instance
embedding_dim=400			#dimension of the representation of each word
epochs =  100				#Number of epochs
batch_size = 64				#Batch size


model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(ml,400)))
model.add(Conv1D(num_filters, 2, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='SAME'))
model.add(Flatten())
model.add(Dropout(drop))
model.add(Dense(100,activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
print(model.summary())






filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=callbacks_list, epochs=epochs, verbose=2, validation_split=0.3)  # starts training
scores=model.evaluate(X_test,y_test,verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))




# serialize model to JSON
model_json = model.to_json()
with open("model_bal_bilstm+cnn.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_bal_bilstm+cnn.h5")
print("Saved model to disk")


'''

# later when loading the saved model
 
# load json and create model
json_file = open('model_bal_bilstm+cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_bal_bilstm+cnn.h5")
print("Loaded model from disk")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
scores=model.evaluate(X_test,y_test,verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))

'''




p=model.predict_classes(X_test)			#prediction of the test data
count=0
count_p=0
count_n=0
i=0

ori_p=0
ori_n=0


#calculating accuracy manually

while(i<len(y_test)):
    if(y_test[i]==1):
	ori_p+=1
    elif(y_test[i]==0):
	ori_n+=1
    if(p[i]==y_test[i]):
        count+=1
	if(y_test[i]==1):
		count_p+=1
	elif(y_test[i]==0):
		count_n+=1
    #print str(y_test[i])+'\t \t'+str(p[i])+'\n'
    i+=1

print "Number of YES correctly predicted = ",count_p, " out of ",ori_p," YES sentences"
print "Number of NO correctly predicted = ",count_n, " out of ",ori_n," NO sentences"

res=100*count/len(y_test)    

wf=open("output_bi-lstm+cnnjan.csv",'w')		#storing the output to a file for performing error analysis
q=0


while(q<len(y_test)):
	wf.write(or_sent_test[q]+'\t'+pair[q]+'\t'+ev_type[q]+'\t'+arg_type[q]+'\t'+trend[q]+'\t'+str(y_test[q])+'\t'+str(p[q])+'\n')
	q+=1

print 'Calculated Accuracy is ',res




m = precision_recall_fscore_support(y_test, p, average=None,labels=['0', '1']) 			#finding Precision, Recall and F-Score for each class
l=['precision = ','recall = ','fbeta-score = ']
q=0
while(q<len(m)):
	print m[q]
	q=q+1


a=confusion_matrix(y_test,p)						#confusion matrix
print a


