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
