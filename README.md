# EVENT-ARGUMENT LINKING(EAL)
Given a Hindi sentence comprising of the sequence,w1,w2,e1,e2,ei,w3...wk,a1,a2,aj,wk+1...wn, where ei is known as an event trigger and aj is known as a candidate argument, the task is to predict whether there exists a relationship between an event trigger ei and an argument trigger aj or not.

# System architecture
![architecture](https://user-images.githubusercontent.com/37550911/53303462-80459280-3890-11e9-9917-6090782f58d8.png)


# Code
The source code of the paper titled **Event-Argument Linking in Hindi for Information Extraction in Disaster Domain**, accepted in the *20th International Conference on Computational Linguistics and Intelligent Text Processing (CICLing 2019)* is put in the file *model.py*

# Dataset
We cannot provide the entire dataset used in our experiments as it is not a publicly available dataset. We have manually created the dataset for our own research purpose. Our research with this dataset is still ongoing. Here we have shown a sample train and test file which consists of the pattern of the training and test dataset used in our experiments.

# Dependency
* Python : 2.7
* scikit-learn: 0.19.2
* tensorflow: 1.9.0
* tensorflow-gpu: 1.9.0
* keras: 2.2.2
* numpy: 1.15.1

# Compile
*python model.py*

# Output
.json and .h5 files will be stored in the path of the file model.py and the model performance will be displayed on the terminal. Precision, Recall, F-Score for both the classes will be printed on the terminal. We have also tried to print the confusion matrix resulting from the classification. How the model predicts the test data in terms of the number of YES and NO cases actually present in the test dataset has also been evaluated and printed.
In order to understand where the system has gone wrong, we have made elaborate error analysis from the *output_bi-lstm+cnnjan.csv* which will be stored in the same path as *model.py* 
