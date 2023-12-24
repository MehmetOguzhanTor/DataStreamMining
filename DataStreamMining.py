
import numpy as np
import matplotlib.pyplot as plt
from skmultiflow.data import SEAGenerator
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from skmultiflow.meta import BatchIncrementalClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.neural_networks import PerceptronMask
import pandas as pd


# ## 1. Dataset Generation

# In[183]:


#Creating the first data file by using SEAGenerator function from scikit Multiflow
#In this first generation the noise percentage is set to be 0 and number of samples is 20000
created_data = SEAGenerator(classification_function = 2, balance_classes = False, noise_percentage = 0)
#created_data.prepare_for_use()
created_data, created_data_label = created_data.next_sample(20000)

#Combining the data and label set into one dataset in order to insert it into a file
combined = np.c_[created_data, created_data_label]
combined.shape

#Creating file named "SEA Dataset.txt" and writing the information into it
with open("SEA Dataset.txt", 'w') as first_file:
    first_file.writelines(','.join(str(j) for j in i)  + '\n' for i in combined)
    
#Converting the first data into Data Frame in order to show it
combined_df = pd.DataFrame(combined)
combined_df.head()


# In[184]:


#Creating the first data file by using SEAGenerator function from scikit Multiflow
#In this first generation the noise percentage is set to be 0.1 and number of samples is 20000
created_data_2 = SEAGenerator(classification_function = 2, balance_classes = False, noise_percentage = 0.1)
#created_data_2.prepare_for_use()
created_data_2, created_data_2_label = created_data_2.next_sample(20000)

#Combining the data and label set into one dataset in order to insert it into a file
combined_2 = np.c_[created_data_2, created_data_2_label]
combined_2.shape

#Creating file named "SEA Dataset 10.txt" and writing the information into it
with open("SEA Dataset 10.txt", 'w') as second_file:
    second_file.writelines(','.join(str(j) for j in i)  + '\n' for i in combined_2)


#Converting the first data into Data Frame in order to show it
combined_2_df = pd.DataFrame(combined_2)
combined_2_df.head()


# In[185]:


#Creating the first data file by using SEAGenerator function from scikit Multiflow
#In this first generation the noise percentage is set to be 0.7 and number of samples is 20000
created_data_3 = SEAGenerator(classification_function = 2, balance_classes = False, noise_percentage = 0.7)
#created_data_3.prepare_for_use()
created_data_3, created_data_3_label = created_data_3.next_sample(20000)

#Combining the data and label set into one dataset in order to insert it into a file
combined_3 = np.c_[created_data_3, created_data_3_label]
combined_3.shape

#Creating file named "SEA Dataset 70.txt" and writing the information into it
with open("SEA Dataset 70.txt", 'w') as third_file:
    third_file.writelines(','.join(str(j) for j in i)  + '\n' for i in combined_3)

#Converting the first data into Data Frame in order to show it
combined_3_df = pd.DataFrame(combined_3)
combined_3_df.head()


# ## 2. Data Stream Classification with Three Separate Online Single Classifiers: HT, KNN, MLP

# # HF 

# In[128]:


#Using Interleaved-Test Then-Train method in Hoeffding tree classifier
#First finding the accuracy without training
#creating a np array for writing the accuracies
accuracy = np.zeros((11))

#Testing the accuracy without training
hoeffding_tree = HoeffdingTreeClassifier()
#print('The accuracy of HF classifier without training is:',"{:.2f}".format(hoeffding_tree.score(created_data, created_data_label)*100,'%'))
accuracy[0] = hoeffding_tree.score(created_data, created_data_label)*100

#There will be 10 epochs
epochs = [0,]

#Training and testing the algorithm 9 times
for i in range (1, 11):

    #Then training the algorthm and finding the accuracy now
    hf = hoeffding_tree.partial_fit(created_data,created_data_label)
    #print('The accuracy of HF classifier is:',"{:.2f}".format(hf.score(created_data, created_data_label)*100,'%'))
    accuracy[i] = hf.score(created_data, created_data_label)*100
    epochs.append(i)

#Creating the second part for data that has 0.1 noise
##Testing the accuracy without training
accuracy_2 = np.zeros((11))
hoeffding_tree_2 = HoeffdingTreeClassifier()
#print('The accuracy of HF classifier for 0.1 noisy data without training is:',"{:.2f}".format(hoeffding_tree_2.score(created_data_2, created_data_2_label)*100,'%'))
accuracy_2[0] = hoeffding_tree_2.score(created_data_2, created_data_2_label)*100

#There will be 10 epochs
epochs_2 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,11):
    #Then training the algorthm and finding the accuracy now
    hf_2 = hoeffding_tree.partial_fit(created_data_2,created_data_2_label)
    #print('The accuracy of HF classifier for 0.1 noisy data is:',"{:.2f}".format(hf_2.score(created_data_2, created_data_2_label)*100,'%'))
    accuracy_2[i] = hf_2.score(created_data_2, created_data_2_label)*100
    epochs_2.append(i)

#Creating the second part for data that has 0.7 noise
##Testing the accuracy without training
accuracy_3 = np.zeros((11))
hoeffding_tree_3 = HoeffdingTreeClassifier()
#print('The accuracy of HF classifier for 0.7 noisy data without training is:',"{:.2f}".format(hoeffding_tree_3.score(created_data_3, created_data_3_label)*100,'%'))
accuracy_3[0] = hoeffding_tree_3.score(created_data_3, created_data_3_label)*100

#There will be 10 epochs
epochs_3 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,11):
    #Then training the algorthm and finding the accuracy now
    hf_3 = hoeffding_tree.partial_fit(created_data_3,created_data_3_label)
    #print('The accuracy of HF classifier for 0.7 noisy data is:',"{:.2f}".format(hf_3.score(created_data_3, created_data_3_label)*100,'%'))
    accuracy_3[i] = hf_3.score(created_data_3, created_data_3_label)*100
    epochs_3.append(i)
    


# In[129]:


#Plotting the accuracy graph for Hoeffding Tree Classifier with 
# labels and titles of the plotting
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Interleaved-Test Then-Train method in Hoeffding Tree Classifier')
plt.plot(epochs,accuracy,'g', label= 'SEA Dataset')    
plt.plot(epochs_2,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(epochs_3,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# # KNN

# In[170]:


#Using Interleaved-Test Then-Train method in KNN classifier
#First finding the accuracy without training
#creating a np array for writing the accuracies
accuracy = np.zeros((5))

#Testing the accuracy without training
neigh = SAMKNNClassifier(n_neighbors=5, weighting='distance', max_window_size=50,
                              stm_size_option='maxACCApprox', use_ltm=False)

print('The accuracy of KNN classifier without training is:',"{:.2f}".format(neigh.score(created_data, created_data_label)*100,'%'))
accuracy[0] = neigh.score(created_data, created_data_label)*100

#There will be 10 epochs
epochs = [0,]

#Training and testing the algorithm 9 times
for i in range (1, 5):

    #Then training the algorthm and finding the accuracy now
    neigh.fit(created_data, created_data_label)
    print('The accuracy of KNN classifier is:',"{:.2f}".format(neigh.score(created_data, created_data_label)*100,'%'))
    accuracy[i] = neigh.score(created_data, created_data_label)*100
    epochs.append(i)

#Creating the second part for data that has 0.1 noise
##Testing the accuracy without training
accuracy_2 = np.zeros((5))
neigh = SAMKNNClassifier(n_neighbors=5, weighting='distance', max_window_size=50,
                              stm_size_option='maxACCApprox', use_ltm=False)
print('The accuracy of KNN classifier without training is:',"{:.2f}".format(neigh.score(created_data_2, created_data_2_label)*100,'%'))
accuracy_2[0] = neigh.score(created_data_2, created_data_2_label)*100

#There will be 10 epochs
epochs_2 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,5):
    #Then training the algorthm and finding the accuracy now
    neigh.fit(created_data_2, created_data_2_label)
    print('The accuracy of KNN classifier is:',"{:.2f}".format(neigh.score(created_data_2, created_data_2_label)*100,'%'))
    accuracy_2[i] = neigh.score(created_data_2, created_data_2_label)*100
    epochs_2.append(i)

#Creating the second part for data that has 0.7 noise
##Testing the accuracy without training
accuracy_3 = np.zeros((5))
neigh = SAMKNNClassifier(n_neighbors=5, weighting='distance', max_window_size=50,
                              stm_size_option='maxACCApprox', use_ltm=False)
print('The accuracy of KNN classifier without training is:',"{:.2f}".format(neigh.score(created_data_3, created_data_3_label)*100,'%'))
accuracy_3[0] = neigh.score(created_data_3, created_data_3_label)*100

#There will be 10 epochs
epochs_3 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,5):
    #Then training the algorthm and finding the accuracy now
    neigh.fit(created_data_3, created_data_3_label)
    print('The accuracy of KNN is:',"{:.2f}".format(neigh.score(created_data_3, created_data_3_label)*100,'%'))
    accuracy_3[i] = neigh.score(created_data_3, created_data_3_label)*100
    epochs_3.append(i)
    


# In[171]:


#Plotting the accuracy graph for KNN Classifier  
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Interleaved-Test Then-Train method in KNN Classifier')
plt.plot(epochs,accuracy,'g', label= 'SEA Dataset')    
plt.plot(epochs_2,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(epochs_3,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# # MLP

# In[132]:


#classify_by_mlp = MLPClassifier(random_state=4, max_iter=4000).fit(train_data, train_label.ravel())
#print("The accuracy of MLP classifier for the validation set is:","{:.2f}".format((classify_by_mlp.score(validation_data, validation_label))*100),'%')


# In[133]:


#mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,4), random_state=2)
#PerceptronMask()
#print('The accuracy of KNN classifier for the validation set is:',"{:.2f}".format(mlp.score(created_data_2, created_data_2_label)*100,'%'))

#mlp = PerceptronMask()
#mlp.fit(created_data,created_data_label)
#mlp.score(created_data, created_data_label)*100


# In[189]:


#Using Interleaved-Test Then-Train method in MLP classifier
#First finding the accuracy without training
#creating a np array for writing the accuracies
accuracy = np.zeros((5))

#Testing the accuracy without training
mlp = PerceptronMask()
mlp.fit(created_data, created_data_label)
print('The accuracy of MLP classifier without training is:',"{:.2f}".format(mlp.score(created_data, created_data_label)*100,'%'))
accuracy[0] = mlp.score(created_data, created_data_label)*100

#There will be 10 epochs
epochs = [0,]

#Training and testing the algorithm 9 times
for i in range (1, 5):

    #Then training the algorthm and finding the accuracy now
    mlp.fit(created_data, created_data_label) 
    print('The accuracy of MLP classifier is:',"{:.2f}".format(mlp.score(created_data, created_data_label)*100,'%'))
    accuracy[i] = mlp.score(created_data, created_data_label)*100
    epochs.append(i)

#Creating the second part for data that has 0.1 noise
##Testing the accuracy without training
accuracy_2 = np.zeros((5))
mlp = PerceptronMask()
mlp.fit(created_data_2, created_data_2_label)
print('The accuracy of MLP classifier without training is:',"{:.2f}".format(mlp.score(created_data_2, created_data_2_label)*100,'%'))
accuracy_2[0] = mlp.score(created_data_2, created_data_2_label)*100

#There will be 10 epochs
epochs_2 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,5):
    #Then training the algorthm and finding the accuracy now
    mlp.fit(created_data, created_data_label)
    print('The accuracy of MLP classifier is:',"{:.2f}".format(mlp.score(created_data_2, created_data_2_label)*100,'%'))
    accuracy_2[i] = mlp.score(created_data_2, created_data_2_label)*100
    epochs_2.append(i)

#Creating the second part for data that has 0.7 noise
##Testing the accuracy without training
accuracy_3 = np.zeros((5))
mlp = PerceptronMask()
mlp.fit(created_data_3, created_data_label)
print('The accuracy of MLP classifier without training is:',"{:.2f}".format(mlp.score(created_data_3, created_data_3_label)*100,'%'))
accuracy_3[0] = mlp.score(created_data_3, created_data_3_label)*100

#There will be 10 epochs
epochs_3 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,5):
    #Then training the algorthm and finding the accuracy now
    mlp.fit(created_data, created_data_label)
    print('The accuracy of MLP is:',"{:.2f}".format(mlp.score(created_data_3, created_data_3_label)*100,'%'))
    accuracy_3[i] = mlp.score(created_data_3, created_data_3_label)*100
    epochs_3.append(i)


# In[187]:


#Plotting the accuracy graph for KNN Classifier  
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Interleaved-Test Then-Train method in KNN Classifier')
plt.plot(epochs,accuracy,'g', label= 'SEA Dataset')    
plt.plot(epochs_2,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(epochs_3,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# ## 3 Data Stream Classification with Two Online Ensemble Classifiers: MV, WMV

# # MV

# In[192]:


#Using Interleaved-Test Then-Train method in VotingClassifier classifier
#First finding the accuracy without training
#creating a np array for writing the accuracies
neigh = KNeighborsClassifier(n_neighbors=3)
accuracy = np.zeros((5))

#Testing the accuracy without training
eclf1 = VotingClassifier(estimators=[('ht', ht), ('knn', neigh), ('mlp', mlp)], voting='hard')
eclf1 = eclf1.fit(created_data, created_data_label)
print('The accuracy of Voting Classifier classifier without training is:',"{:.2f}".format(eclf1.score(created_data, created_data_label)*100,'%'))
accuracy[0] = eclf1.score(created_data, created_data_label)*100

#There will be 10 epochs
epochs = [0,]

#Training and testing the algorithm 9 times
for i in range (1, 5):

    #Then training the algorthm and finding the accuracy now
    eclf1 = eclf1.fit(created_data, created_data_label)
    print('The accuracy of Voting Classifier classifier is:',"{:.2f}".format(eclf1.score(created_data, created_data_label)*100,'%'))
    accuracy[i] = eclf1.score(created_data, created_data_label)*100
    epochs.append(i)

#Creating the second part for data that has 0.1 noise
##Testing the accuracy without training
accuracy_2 = np.zeros((5))
eclf1 = VotingClassifier(estimators=[('ht', ht), ('knn', neigh), ('mlp', mlp)], voting='soft')
eclf1 = eclf1.fit(created_data, created_data_label)
print('The accuracy of Voting Classifier classifier without training is:',"{:.2f}".format(eclf1.score(created_data_2, created_data_2_label)*100,'%'))
accuracy_2[0] = eclf1.score(created_data_2, created_data_2_label)*100

#There will be 10 epochs
epochs_2 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,5):
    #Then training the algorthm and finding the accuracy now
    eclf1 = eclf1.fit(created_data_2, created_data_2_label)
    print('The accuracy of Voting Classifier classifier is:',"{:.2f}".format(eclf1.score(created_data_2, created_data_2_label)*100,'%'))
    accuracy_2[i] = eclf1.score(created_data_2, created_data_2_label)*100
    epochs_2.append(i)

#Creating the second part for data that has 0.7 noise
##Testing the accuracy without training
accuracy_3 = np.zeros((5))
eclf1 = VotingClassifier(estimators=[('ht', ht), ('knn', neigh), ('mlp', mlp)], voting='soft')
eclf1 = eclf1.fit(created_data, created_data_label)
print('The accuracy of Voting Classifier classifier without training is:',"{:.2f}".format(eclf1.score(created_data_3, created_data_3_label)*100,'%'))
accuracy_3[0] = eclf1.score(created_data_3, created_data_3_label)*100

#There will be 10 epochs
epochs_3 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,5):
    #Then training the algorthm and finding the accuracy now
    eclf1 = eclf1.fit(created_data_3, created_data_3_label)
    print('The accuracy of Voting Classifier is:',"{:.2f}".format(eclf1.score(created_data_3, created_data_3_label)*100,'%'))
    accuracy_3[i] = eclf1.score(created_data_3, created_data_3_label)*100
    epochs_3.append(i)


# In[193]:


#Plotting the accuracy graph for Voting Classifier Classifier  
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Interleaved-Test Then-Train method in Voting Classifier Classifier')
plt.plot(epochs,accuracy,'g', label= 'SEA Dataset')    
plt.plot(epochs_2,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(epochs_3,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# # WMV

# eclf3 = VotingClassifier(estimators=[('lr', ht), ('rf', neigh), ('gnb', mlp)], voting='soft', weights=[1,1,1], 
#                          flatten_transform=True)
# eclf3 = eclf3.fit(created_data, created_data_label)
# 
# print('The accuracy of KNN classifier for the validation set is:',"{:.2f}".format(eclf3.score(created_data_2, created_data_2_label)*100,'%'))

# In[196]:


#Using Interleaved-Test Then-Train method in VotingClassifier classifier
#First finding the accuracy without training
#creating a np array for writing the accuracies
accuracy = np.zeros((5))

#Testing the accuracy without training
eclf1 =  VotingClassifier(estimators=[('lr', ht), ('rf', neigh), ('gnb', mlp)], voting='soft', weights=[1,2,1], 
                         flatten_transform=True)
eclf1 = eclf1.fit(created_data, created_data_label)
print('The accuracy of Weighted Voting Classifier classifier without training is:',"{:.2f}".format(eclf1.score(created_data, created_data_label)*100,'%'))
accuracy[0] = eclf1.score(created_data, created_data_label)*100

#There will be 10 epochs
epochs = [0,]

#Training and testing the algorithm 9 times
for i in range (1, 5):

    #Then training the algorthm and finding the accuracy now
    eclf1 = eclf1.fit(created_data, created_data_label)
    print('The accuracy of Weighted Voting Classifier classifier is:',"{:.2f}".format(eclf1.score(created_data, created_data_label)*100,'%'))
    accuracy[i] = eclf1.score(created_data, created_data_label)*100
    epochs.append(i)

#Creating the second part for data that has 0.1 noise
##Testing the accuracy without training
accuracy_2 = np.zeros((5))
eclf1 =  VotingClassifier(estimators=[('lr', ht), ('rf', neigh), ('gnb', mlp)], voting='soft', weights=[1,2,1], 
                         flatten_transform=True)
eclf1 = eclf1.fit(created_data_2, created_data_2_label)
print('The accuracy of Weighted Voting Classifier classifier without training is:',"{:.2f}".format(eclf1.score(created_data_2, created_data_2_label)*100,'%'))
accuracy_2[0] = eclf1.score(created_data_2, created_data_2_label)*100

#There will be 10 epochs
epochs_2 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,5):
    #Then training the algorthm and finding the accuracy now
    eclf1 = eclf1.fit(created_data_2, created_data_2_label)
    print('The accuracy of Weighted Voting Classifier classifier is:',"{:.2f}".format(eclf1.score(created_data_2, created_data_2_label)*100,'%'))
    accuracy_2[i] = eclf1.score(created_data_2, created_data_2_label)*100
    epochs_2.append(i)

#Creating the second part for data that has 0.7 noise
##Testing the accuracy without training
accuracy_3 = np.zeros((5))
eclf1 =  VotingClassifier(estimators=[('lr', ht), ('rf', neigh), ('gnb', mlp)], voting='soft', weights=[1,2,1], 
                         flatten_transform=True)
eclf1 = eclf1.fit(created_data, created_data_label)
print('The accuracy of Weighted Voting Classifier classifier without training is:',"{:.2f}".format(eclf1.score(created_data_3, created_data_3_label)*100,'%'))
accuracy_3[0] = eclf1.score(created_data_3, created_data_3_label)*100

#There will be 10 epochs
epochs_3 = [0,]

#Training and testing the algorithm 9 times
for i in range (1,5):
    #Then training the algorthm and finding the accuracy now
    eclf1 = eclf1.fit(created_data_3, created_data_3_label)
    print('The accuracy of Weighted Voting Classifier is:',"{:.2f}".format(eclf1.score(created_data_3, created_data_3_label)*100,'%'))
    accuracy_3[i] = eclf1.score(created_data_3, created_data_3_label)*100
    epochs_3.append(i)


# In[197]:


#Plotting the accuracy graph for Weighted Voting Classifier Classifier  
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Interleaved-Test Then-Train method in Weighted Voting Classifier Classifier')
plt.plot(epochs,accuracy,'g', label= 'SEA Dataset')    
plt.plot(epochs_2,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(epochs_3,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# # 4 Batch Classification with Three Separate Batch Single Classifiers: HT, KNN, MLP

# # Batch Classification for HT

# In[231]:


#Using batch classifier for every dataset that we created for HT
#The window size is determined in the for loops
accuracy_1 = np.zeros((3))
for i in range (1,4):
    window_size = i*150
    batch_hf = BatchIncrementalClassifier(base_estimator= ht, window_size=window_size, n_estimators=100)
    batch_hf.partial_fit(created_data, created_data_label)
    accuracy_1[i-1] = batch_hf.score(created_data, created_data_label)*100
    print('The accuracy of Batch classifier for the first data HT set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_1[i-1]),'%')
#Using batch classifier for every dataset that we created for HT
#The window size is determined in the for loops
accuracy_2 = np.zeros((3))
for i in range (1,4):
    window_size = i*150
    batch_hf = BatchIncrementalClassifier(base_estimator= ht, window_size=window_size, n_estimators=100)
    batch_hf.partial_fit(created_data_2, created_data_2_label)
    accuracy_2[i-1] = batch_hf.score(created_data_2, created_data_2_label)*100
    print('The accuracy of Batch classifier for the second data for HT set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_2[i-1]),'%')
    
#Using batch classifier for every dataset that we created for HT
#The window size is determined in the for loops
accuracy_3 = np.zeros((3))
for i in range (1,4):
    window_size = i*150
    batch_hf = BatchIncrementalClassifier(base_estimator= ht, window_size=window_size, n_estimators=100)
    batch_hf.partial_fit(created_data_3, created_data_3_label)
    accuracy_3[i-1] = batch_hf.score(created_data_3, created_data_3_label)*100
    print('The accuracy of Batch classifier for the third data HT set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_3[i-1]),'%')
    
#Window sizes
window_size = [150,300,450]


# In[232]:


#Plotting the accuracy graph for Batch classification for HT classifier
plt.figure()
plt.xlabel('window_size')
plt.ylabel('Accuracy')
plt.title('Batch Classification of HF with different window sizes')
plt.plot(window_size,accuracy_1,'g', label= 'SEA Dataset')    
plt.plot(window_size,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(window_size,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# # Batch Classification for KNN

# In[226]:


#Using batch classifier for every dataset that we created for KNN
#The window size is determined in the for loops
accuracy_1 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    batch_neigh = BatchIncrementalClassifier(base_estimator= neigh, window_size=window_size, n_estimators=100)
    batch_neigh.partial_fit(created_data, created_data_label)
    print('The accuracy of Batch classifier for KNN set with a window size of', window_size ,' is:',"{:.2f}".format(batch_neigh.score(created_data, created_data_label)*100,'%'))
    accuracy_1[i-1] = batch_neigh.score(created_data, created_data_label)*100
    
#Using batch classifier for every dataset that we created for KNN
#The window size is determined in the for loops    
accuracy_2 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    batch_neigh = BatchIncrementalClassifier(base_estimator= neigh, window_size=window_size, n_estimators=100)
    batch_neigh.partial_fit(created_data_2, created_data_2_label)
    print('The accuracy of Batch classifier for KNN set with a window size of', window_size ,' is:',"{:.2f}".format(batch_neigh.score(created_data_2, created_data_2_label)*100,'%'))
    accuracy_2[i-1] = batch_neigh.score(created_data_2, created_data_2_label)*100
    
#Using batch classifier for every dataset that we created for KNN
#The window size is determined in the for loops
accuracy_3 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    batch_neigh = BatchIncrementalClassifier(base_estimator= neigh, window_size=window_size, n_estimators=100)
    batch_neigh.partial_fit(created_data_3, created_data_3_label)
    print('The accuracy of Batch classifier for KNN set with a window size of', window_size ,' is:',"{:.2f}".format(batch_neigh.score(created_data_3, created_data_3_label)*100,'%'))
    accuracy_3[i-1] = batch_neigh.score(created_data_3, created_data_3_label)*100
    
#Window sizes 
window_size = [100,200,300]


# In[228]:


#Plotting the accuracy graph for Batch classification for KNN classifier
plt.figure()
plt.xlabel('window_size')
plt.ylabel('Accuracy')
plt.title('Batch Classification of KNN with different window sizes')
plt.plot(window_size,accuracy_1,'g', label= 'SEA Dataset')    
plt.plot(window_size,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(window_size,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# # Batch Classification for MLP

# In[146]:


batch_incremental_cfier_neigh = BatchIncrementalClassifier(base_estimator= eclf1, window_size=150, n_estimators=100)
batch_incremental_cfier_neigh.partial_fit(created_data, created_data_label)
#batch_incremental_cfier_neigh.score(created_data_2, created_data_2_label)*100
print('The accuracy of KNN classifier for the validation set is:',"{:.2f}".format(batch_incremental_cfier_neigh.score(created_data_2, created_data_2_label)*100,'%'))


# In[224]:


#Using batch classifier for every dataset that we created for KNN
#The window size is determined in the for loops
accuracy_1 = np.zeros((10))
for i in range (1,11):
    window_size = i*100
    batch_mlp = BatchIncrementalClassifier(base_estimator= mlp, window_size=window_size, n_estimators=100)
    batch_mlp.partial_fit(created_data, created_data_label)
    print('The accuracy of Batch classifier for MLP set with a window size of', window_size ,' is:',"{:.2f}".format(batch_mlp.score(created_data, created_data_label)*100,'%'))
    accuracy_1[i-1] = batch_mlp.score(created_data, created_data_label)*100
    
accuracy_2 = np.zeros((10))
for i in range (1,11):
    window_size = i*100
    batch_mlp = BatchIncrementalClassifier(base_estimator= mlp, window_size=window_size, n_estimators=100)
    batch_mlp.partial_fit(created_data_2, created_data_2_label)
    print('The accuracy of Batch classifier for MLP set with a window size of', window_size ,' is:',"{:.2f}".format(batch_mlp.score(created_data_2, created_data_2_label)*100,'%'))
    accuracy_2[i-1] = batch_mlp.score(created_data_2, created_data_2_label)*100

accuracy_3 = np.zeros((10))
for i in range (1,11):
    window_size = i*100
    batch_mlp = BatchIncrementalClassifier(base_estimator= mlp, window_size=window_size, n_estimators=100)
    batch_mlp.partial_fit(created_data_3, created_data_3_label)
    print('The accuracy of Batch classifier for MLP set with a window size of', window_size ,' is:',"{:.2f}".format(batch_mlp.score(created_data_3, created_data_3_label)*100,'%'))
    accuracy_3[i-1] = batch_mlp.score(created_data_3, created_data_3_label)*100
    
window_size = [100,200,300,400,500,600,700,800,900,1000]


# In[225]:


#Plotting the accuracy graph for Batch classification for MLP classifier
plt.figure()
plt.xlabel('window_size')
plt.ylabel('Accuracy')
plt.title('Batch Classification of MLP with different window sizes')
plt.plot(window_size,accuracy_1,'g', label= 'SEA Dataset')    
plt.plot(window_size,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(window_size,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# # 5 Batch Classification with Two Batch Ensemble Classifiers: MV, WMV 

# # MV

# In[233]:


#Using batch classifier for every dataset that we created for MV
#The window size is determined in the for loops
accuracy_1 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    mv = VotingClassifier(estimators=[('ht', ht), ('knn', neigh), ('mlp', mlp)], voting='hard')
    batch_MV = BatchIncrementalClassifier(base_estimator= mv, window_size=window_size, n_estimators=100)
    batch_MV.partial_fit(created_data, created_data_label)
    accuracy_1[i-1] = batch_MV.score(created_data, created_data_label)*100
    print('The accuracy of Batch classifier for MV set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_1[i-1],'%'))
    
    
#Using batch classifier for every dataset that we created for MV
#The window size is determined in the for loops    
accuracy_2 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    mv = VotingClassifier(estimators=[('ht', ht), ('knn', neigh), ('mlp', mlp)], voting='hard')
    batch_MV = BatchIncrementalClassifier(base_estimator= mv, window_size=window_size, n_estimators=100)
    batch_MV.partial_fit(created_data_2, created_data_2_label)
    accuracy_2[i-1] = batch_MV.score(created_data_2, created_data_2_label)*100
    print('The accuracy of Batch classifier for MV set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_2[i-1],'%'))
    
    
#Using batch classifier for every dataset that we created for MV
#The window size is determined in the for loops
accuracy_3 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    mv = VotingClassifier(estimators=[('ht', ht), ('knn', neigh), ('mlp', mlp)], voting='hard')
    batch_MV = BatchIncrementalClassifier(base_estimator= mv, window_size=window_size, n_estimators=100)
    batch_MV.partial_fit(created_data_3, created_data_3_label)
    accuracy_3[i-1] = batch_MV.score(created_data_3, created_data_3_label)*100
    print('The accuracy of Batch classifier for MV set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_3[i-1],'%'))
    
    
#Window sizes 
window_size = [100,200,300]


# In[234]:


#Plotting the accuracy graph for Batch classification for MV classifier
plt.figure()
plt.xlabel('window_size')
plt.ylabel('Accuracy')
plt.title('Batch Classification of MV with different window sizes')
plt.plot(window_size,accuracy_1,'g', label= 'SEA Dataset')    
plt.plot(window_size,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(window_size,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# # WMV

# In[235]:


#Using batch classifier for every dataset that we created for WMV
#The window size is determined in the for loops
accuracy_1 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    wmv = VotingClassifier(estimators=[('lr', ht), ('rf', neigh), ('gnb', mlp)], voting='soft', weights=[1,2,1], 
                         flatten_transform=True)
    batch_WMV = BatchIncrementalClassifier(base_estimator= wmv, window_size=window_size, n_estimators=100)
    batch_WMV.partial_fit(created_data, created_data_label)
    accuracy_1[i-1] = batch_WMV.score(created_data, created_data_label)*100
    print('The accuracy of Batch classifier for MV set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_1[i-1]),'%')

#Using batch classifier for every dataset that we created for MV
#The window size is determined in the for loops    
accuracy_2 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    wmv = VotingClassifier(estimators=[('lr', ht), ('rf', neigh), ('gnb', mlp)], voting='soft', weights=[1,2,1], 
                         flatten_transform=True)
    batch_WMV = BatchIncrementalClassifier(base_estimator= wmv, window_size=window_size, n_estimators=100)
    batch_WMV.partial_fit(created_data_2, created_data_2_label)
    accuracy_2[i-1] = batch_WMV.score(created_data_2, created_data_2_label)*100
    print('The accuracy of Batch classifier for MV set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_2[i-1]),'%')
    
    
#Using batch classifier for every dataset that we created for MV
#The window size is determined in the for loops
accuracy_3 = np.zeros((3))
for i in range (1,4):
    window_size = i*100
    wmv = VotingClassifier(estimators=[('lr', ht), ('rf', neigh), ('gnb', mlp)], voting='soft', weights=[1,2,1], 
                         flatten_transform=True)
    batch_WMV = BatchIncrementalClassifier(base_estimator= wmv, window_size=window_size, n_estimators=100)
    batch_WMV.partial_fit(created_data_3, created_data_3_label)
    accuracy_3[i-1] = batch_WMV.score(created_data_3, created_data_3_label)*100
    print('The accuracy of Batch classifier for MV set with a window size of', window_size ,' is:',"{:.2f}".format(accuracy_3[i-1]),'%')
    
    
#Window sizes 
window_size = [100,200,300]


# In[236]:


#Plotting the accuracy graph for Batch classification for WMV classifier
plt.figure()
plt.xlabel('window_size')
plt.ylabel('Accuracy')
plt.title('Batch Classification of WMV with different window sizes')
plt.plot(window_size,accuracy_1,'g', label= 'SEA Dataset')    
plt.plot(window_size,accuracy_2,'b',label='SEA Dataset 10')
plt.plot(window_size,accuracy_3,'r',label = 'SEA Dataset 70') 
plt.legend()


# In[ ]:




