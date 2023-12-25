# Data Stream Mining


Data Stream Mining
Introduction
The purpose of 5th project of the GE 461 Introduction to Data Science course is to work on a data stream mining project by trying different online classifiers such as Hoeffding Tree, K nearest Neighbor and Multilayer Perceptron. Along with the single online classifiers, in the project, there are different tasks that are used to increase the accuracies of the algorithms for large data assignments. As an example, for these tasks, Majority Voting rule and Weighted Majority Voting rule are implemented to ensemble the online classifiers to get more accuracy from testing the algorithm. Finally, Batch Classification method is implemented to increase the accuracy by dividing the data into window sizes and get results in more detail.
Work Done

1.	Dataset Generation
In the first part of the assignment, 3 different data is created for the purpose of using them in multiple online classification, and batch classification tasks. SEA Generator is used to create 3 new data from nothing. For the first data whose name is “SEA Dataset”, classification function which is a hyper parameter of the SEA Generator function is set to be 2 and balance classes parameter is set to be False. These two parameters are set the same with the remaining two dataset that are created. The difference for the datasets were their noise percentage. For the first data whose name is “SEA Dataset”, the noise percentage is set to be 0. For the second dataset whose name is “SEA Dataset 10”, the noise percentage is set to be 0.1 and finally, for the third dataset whose name is “SEA Dataset 70”, the noise percentage is set to be 0.7. All tree of the datasets has 2 classes which are ‘1’ and ‘0’. The number of samples for all the datasets was set to 20000. These datasets were written into files according to their names.

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/573d32c2-41dd-42d6-b217-b136029c797b)

Figure 1,2,3: A sample from “SEA Dataset”, “SEA Dataset 10” and “SEA Dataset 70”

2.	Data Stream Classification with HT, KNN, MLP
In the second part of the project, we needed to learn about Hoeffding Tree, K nearest neighbour and Multilayer Perceptron and implement them into our three datasets in order the make comparisons on online learners and noisy data and their effects on the accuracy.
a.	Hoeffding Tree
For the first online learner of the project, we needed to implement Hoeffding Tree (HT) on all the three SEA Datasets that are generated in the first part of the assignment. Hoeffding tree which is used for large data stream like in our project is an incremental decision tree algorithm. The assumption that Hoeffding Tree does is that it thinks that the distribution of the data does not change over time [1]. In the algorithm of the Hoeffding Tree, there is a Hoeffding bound which names the tree and from this bound the tree grows incrementally. This online single classifier is widely in the data stream mining projects.
In the code part of the Hoeffding Tree algorithm, scikit-multiflow library used for its function. In the process, 10 epochs are done for all three datasets to make sure that the algorithm learned better and gives better accuracies. In this part Interleaved Test Then Train approach is used in which the algorithm is testing the dataset first and then training it to get better results at the end. When the algorithm is implemented on SEA Dataset(green), SEA Dataset 10(blue) and SEA Dataset 70(red), the accuracy differs as the epoch number increases like in the following figure of plotting.

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/dc63d073-7a1b-4f44-b7f1-eeccb11ce9dc)

Figure 4: Plotting of Interleaved Test Then Train method in Hoeffding Tree Classifier

As it can clearly be observed from the graph that, when the algorithm is tested before training, the result of accuracy is extremely low for SEA Dataset and SEA Dataset 10 whose noise is either too low or there is no noise at all. However, when the training of the algorithm is done over these datasets, it can be observed that accuracy increases dramatically. On the other hand,
 
when the observation is done on the results when the high noise data is implemented on the Hoeffding Tree Classifier, the accuracy somehow higher at the begging yet, it drops when the training starts and starts increasing as the training continuous. The reason for this distribution of accuracies is the high percentage of noise in the SEA Dataset 70.
b.	K nearest neighbour
For the second online learner of the project, we needed to implement K nearest neighbour (KNN) on all the three SEA Datasets that are generated in the first part of the assignment. KNN is an algorithm that can be used for both regression and classification problems, yet, in our case it is used for classification purposes. How KNN works is that in the algorithm there is a parameter for k which represents the number of neighbor that the algorithm should check for. Then, for every sample it checks the k of the neighbors and decide the class of the sample by checking the higher number of class in the neighbors [2].
In our project, KNN is used to classify the datasets that are created in the first step with optimal parameter and with highest accuracy possible. In the process of KNN, again the Interleaved Test Then Train approach is used successfully, and the observations are done accordingly to this approach. The observation of the accuracy results can be done in the following figure of plotting.

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/c52031f6-38c6-4f4e-bd03-68eb6d0f679a)

Figure 5: Plotting of Interleaved Test Then Train method in KNN Classifier

As the observations are made on the plotting of accuracies of the three datasets, the results are similar with the Hoeffding Tree algorithm. For the first two dataset which have low or no noise in their distribution, the accuracy is extremely low in the begging but, when the training starts, it increases dramatically. On the other hand, although it is started higher than the other datasets, the accuracy of the SEA Dataset 70 does not increase that much because of the noise its data distribution has.
 
c.	Multilayer Perceptron
For the third online learner of the project, we needed to implement Multilayer Perceptron (MLP) on all the three SEA Datasets that are generated in the first part of the assignment. First, perceptron which in the name of MLP is a neuron model which was precursor for neural networks which is larger. Because of this, the other name of MLP is neural networks. MLP is used for difficult computational tasks in machine learning. Its purpose is to develop robust algorithms and structures to model and classify the difficult tasks. [3]
In the process of implementing MLP classifier in the dataset that we created in the first part of the project by using Interleaved Test Then Train approach, there was some difficulties experienced. First, when testing the algorithm of MLP is tried without training it, an error was occurred saying that Perceptron instance is not fitted yet which means we cannot use the Interleaved Test Then Train approach in the begging of the MLP algorithm. Instead, fitting function had to be incremented first on the SEA Dataset to train the algorithm and then, the training could continue. To implement the approach a little, for the remaining two data sets that are SEA Dataset 10 and SEA Dataset 70, the first SEA Dataset is used for fitting to observe how the algorithm handles the training part of the aforementioned datasets. So, the result of the MLP algorithm on the three datasets is in the following figure of plotting.

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/5f898dbc-88db-4cb0-86e7-4defa91db7aa)

Figure 6: Plotting of Interleaved Test Then Train method in MLP Classifier

As the observations are made on the results of the MLP classifier, the accuracy of the SEA Dataset is extremely high all the time and did not even need the training part. For the SEA Dataset 10 which has noise, the training seems successful, and the accuracy increased when the training is done. However, for the SEA Dataset 70 which has high percentage of noise, the algorithm failed. It started little high thanks to the training in the begging yet, when the data is implemented, the training failed because of the noise.
 
3.	Data Stream Classification with MV, WMV
In this part of the project, it is asked to implement data stream classification with two online ensemble classifiers which are the Majority Voting rule MV and Weighted Majority Voting rule WMV. These are ensemble classifiers that in our case will combine Hoeffding Tree, K Nearest Neighbor and Multilayer Perceptron single online classifiers for the three SEA Datasets that are generated in the first step of the assignment.
a.	Majority Voting Rule (MV)
For the first ensemble classifier, Voting Classifier function used from skylearn library. Inside the function, there is a parameter inside which online classifier that are desired to be used is chosen. In our case HF, KNN and MLP is chosen and inserted into the function. For the function to work without any errors, it is learned that these functions must be inserted in a different location in the code. In the process there is a error that occurred in the MLP algorithm as well. In the error, again it stated that fitting must be implemented before testing any data in the algorithm. So, Interleaved Test Then Train approach failed in the MV and WMV algorithms. However, when the results of the MV algorithm is observed, the accuracies were higher than any classifier that are used in the second part. The accuracy plot can be seen in the following figure.

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/2f4a513b-2946-4333-a9a0-b28cb107a7db)

Figure 7: Plotting of Interleaved Test Then Train method in MV Classifier

As it can be observed from the plotting that, the accuracy of the SEA Dataset is just under the 100% accuracy which is a successful result of training and testing. For the second data set of SEA Dataset 10, the result again seems good considering there is 10% noise inside the data. Finally, for the third dataset of SEA Dataset 70, the training worked better than before as an increase can be observed in the accuracy graphic considering that it started around 30%. So, it can be stated that MV online ensemble classifier gives better results than any single classifier that are used in this assignment.
 
b.	Weighted Majority Voting Rule (WMV)
When the Majority Voting Rule is used with weighted online classifier, there is a chance of getting better accuracy results from the ensembled classifier because some single classifiers have more positive affects when their weights are increased but, some of them could decrease the accuracy result since their effects could be negative when their weighs are increased. In the case of our project, after some trials, it is observed that KNN being more weighted has positive effects on the results of accuracy. So, the weight of KNN is increased as other classifiers weight stayed the same. The new results can be observed from the following figure.

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/b89b6b94-df84-4b53-845a-d1e73c2e3d4e)

Figure 8: Plotting of Interleaved Test Then Train method in WMV Classifier

As it can be observed from the plotting of accuracies, the results got even higher than they were in the MV. So, WMV could have positive effects on the accuracy when the weights are selected according to the single classifiers.
4.	Batch Classification with HT, KNN, MLP
In the fourth part of the project assignment, we are asked to implement batch classification with the online classifiers HT, KNN and MLP that are implemented in the second part of the project. The porpuse of the batch classification is to slice the data into window sizes and finds the highest accuracy among the slices. So, the important part in the batch classification is to determine the window size successfully. However, one of the worst parts of the batch classification is that the algorithm of the batch classification takes too much time to run especially when compared with the other classifications that are implemented in the other parts of the project assignment. Because it takes too much time running one batch classifier, only 3 window sizes are tried for every datasets of every online classifier. However, 3 window size is
 
enough to observe the changes of the accuracies when closely observed. The results of batch classification for every classifier are given separately in their parts.
