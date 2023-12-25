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

HT

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/dd076d61-cf5f-4015-8640-5629c6570dd9)

Figure 9: Plotting of Batch classification of HF with different window sizes

As it can be observed from the graph of the batch classification of HF, window sizes affect the resulting accuracy. It is important to find the best window size for a given data but, the batch classification with large datasets takes too much time for trying every window size and discovering the best of that fits that dataset. For the datasets that we created in the first part of this project, the accuracies differ for three different window sizes. For the first dataset which has no noise, the accuracy increases, on the other hand, for the second and third data sets which have noise inside, the accuracy decreases for the same window sizes. It is important to note that their accuracies may be increasing for different window sizes.

KNN

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/94ac95e2-064c-4db7-9196-aae6e13d5887)

Figure 10: Plotting of Batch classification of KNN with different window sizes

As it can be observed from the plotting of the batch classification of KNN, different window sizes differ the resulting of accuracies. In the case of KNN classifier, it seems that in first two algorithms where first two datasets are trained and tested, the accuracy results increase as the window size gets to 300 from
100.	On the other hand, accuracy increased with a less slope for the third data in this window size range. Again, it may be increasing more when the window size is different. However, the range of the window size could not be widened in the KNN case because batch classification of KNN algorithm takes too much time and it is difficult and time consuming to try a large range of window sizes in this algorithm.

MLP

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/b6608aae-78f2-42f1-bf6e-c8cc813c9956)

Figure 11: Plotting of Batch classification of MLP with different window sizes

In the last single batch classification part, which is for the MLP classifier, the algorithm worked faster than the first two single classifier did. This made it possible to try a large range of window size which differs from 100 to 1000 with a step size of 100. So, 10 different window sizes are tried in the last part of fourth problem. As a result, the effects of window sizes on the accuracy of the algorithm can be better observed in the batch classification for MLP classifier. As it can be seen from the plotting that accuracy increases and decreases as the step size differs and it is possible to find the best step size for any data given to the algorithm for testing and training.
5.	Batch Classification with MV, WMV
In the last coding part of the assignment, the batch classifiers are implemented with the usage of Majority Vaoting Rule and Weighted Majority Voting Rule which include the Hoeffding Tree, K nearest neighbor and Multilayer Perceptron inside them.
MV
The results from batch classification of MV with different window sizes is given in the following figure.

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/0a6b98ae-df0e-42e0-b3ed-b4fefaaa8adc)

Figure 12: Plotting of Batch classification of MV with different window sizes

As the results show that, the accuracy of the model with all of the three data is higher than the individual ones. The reason for this is that the classification is ensembled with all three classifiers.

WMV

![image](https://github.com/MehmetOguzhanTor/DataStreamMining/assets/116079107/cc7ecd27-966d-4c43-8a45-a9519a7ce0c2)

Figure 12: Plotting of Batch classification of WMV with different window sizes

Changing the weights of the single classifiers inside the batch classification model can change the result of the accuracy.
6.	Comparison of Models
b. The comparisons are done in detail in the end of every part. To summarize them for steps to and 3 for Interleaved Test Then Train approach, for the Hoeffding Tree classifier, this approach worked perfectly, and the algorithm trained as the number of training increased and at the end, the resulting accuracies were satisfying considering their individual advantages of disadvantages of noise as some of them has more noise than others. K nearest neighbor also worked successful among single classifiers and an increase is observed as the algorithm is trained over epochs. On the other hand, for the last single online classifier which is MLP the case was not so successful as it did not accept testing the algorithm over the data without training first. Some ways are tried to observe the testing for undertrained model to train it with another dataset, yet it did not show the result that we expected from an Interleaved Test Then Train approach because when the third data were being trained on the model, the accuracy dropped unexpectedly. Other than this situation, the process was successful and both HT and KNN were successful classifiers for the model that it created.
When Ensemble classifiers are tried like Majority Voting Rule and Weighted Majority Voting Rule, the success rate increased especially with Weights in which case better classifier could have higher weight than other and this situation results in with more accuracy. So, assembling more than one classifier is a successful way to model an algorithm than trying online single classifiers one by one. The resulting shows that ensembles are better than individual models(d.).
c. Batch sizes does affect the resulting accuracies as they discussed in the batch classification part. Batch size which means window size can increase and decrease the amount of the accuracy differing from classifier or even the data that is used. In order to understand this differentiation different window sizes are tried when it was possible, and the observations are done accordingly.
e. When the necessary observations are made, it could be said that ensembled batch classifiers slightly more accuracy than single batch classifiers. The reason for this situation is that in the ensembled batch classifiers, there are more classifiers involved which increases the total accuracy. Moreover, when the weighted ensembled batch classifier is implemented, the result of accuracy is higher because weighting the classifiers is beneficial over describing the best online single classifier.
f. the improvement of the prediction accuracy is tried in every classifier as there were range of epochs and window sizes. Changing the epoch and window sizes will result in more accuracy.
Appendix

[1]	“Hoeffding Decision Trees,” streamDM. [Online]. Available: http://huawei- noah.github.io/streamDM/docs/HDT.html#:~:text=The%20Hoeffding%20tree%20is%2 0an,(or%20additive%20Chernoff%20bound).&text=Mining%20High- Speed%20Data%20Streams. [Accessed: 14-May-2021].

[2]	T. S. T. Srivastava, “K Nearest Neighbor: KNN Algorithm: KNN in Python & R,” Analytics Vidhya, 18-Oct-2020. [Online]. Available: https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm- clustering/. [Accessed: 14-May-2021].
 
[3]	J. Brownlee, “Crash Course On Multi-Layer Perceptron Neural Networks,” Machine Learning Mastery, 14-Aug-2020. [Online]. Available: https://machinelearningmastery.com/neural-networks-crash-course/. [Accessed: 14- May-2021].
