Classification of Functional Assessment from Free Recall Performance in Alzheimer's Patients
============================================================================================

Often when patients are tested for symptoms of Alzheimer’s disease, assessments are done through the use of brief cognitive measures, like tests of verbal abilities, categorization tasks, or memory tasks. In particular, people diagnosed with Alzheimer's disease tend to have poor episodic memory - or memory for past events (Tulving, 1972) - and perform poorly on clinical tests of episodic memory (e.g., Storandt & Hill, 1989; Wilson, Bacon, Fox, & Kaszniak, 1983).

Through a collaboration with a local Alzheimer's clinic, my lab has access to a large clinical data set of patients with Alzheimer’s disease. All patients were categorized into stages ranging from 1-6 using the Functional Assessment Staging Test (FAST; Reisberg, 1988). The FAST score describes symptom severity in terms of one's ability to perform daily living activities, where higher FAST stages indicate greater severity of symptoms. For example, someone categorized into FAST stage 1 has no discernable deficits or symptoms. People categorized into FAST stage 3 need help with complicated tasks, like financial planning. People categorized into FAST stage 5 or 6 are suffering from dementia and need help with basic tasks, such as dressing and bathing. The following table presents a summary of FAST scores, typical characteristics, and the number of people in the data set with that score.

###### Table 1: Characteristics of and Number of Patients in Each FAST Stage
| FAST score | Characteristics                         | Patients |
|:----------:|:--------------------------------------- |:--------:|
| 1          | No deficits whatsoever                  | 99       |
| 2          | Subjective functional deficit           | 436      |
| 3          | Difficulty accomplishing complex tasks  | 813      |
| 4          | Requires help cooking & cleaning        | 944      |
| 5          | Requires help selecting proper clothing | 210      |
| 6          | Requires help dressing & bathing        | 145      |

For the 2,647 people in this data set, we have the results of a Mild Cognitive Impairment Screen (MCIS), which includes a battery of memory tasks, such as recognition memory, metamemory judgments, and triadic comparisons. However, for this project, I will focus on the task that relies heavily on episodic memory: the free recall task.

The Free Recall Task
--------------------
In a free recall task, the participant is presented with a list of items and is then asked to recall as many items from the list as possible, in any order. As part of the MCIS, there are a total of four free recall tasks: three immediate free recall and one delayed free recall. The three immediate free recall tasks are delivered in sequence, with no delay between the presentation of the list and the recall of items. Participants hear a 10-item word list read aloud by the clinician consisting of the words BUTTER, ARM, SHORE, LETTER, QUEEN, CABIN, POLE, TICKET, GRASS, and ENGINE. This study session is followed by an immediate prompt to recall the words. The order of the presentation of the words is balanced between subjects. Within subjects, words are presented in the same order each time the list is read aloud by the clincian. After the third immediate free recall task is complete, the participant completes an unrelated task for 2 to 5 minutes. Then, unexpectedly and without any additional exposure to the study list, the participant is again asked to recall as many items from the list as possible.

The Data
--------
The dataset itself consists of binary data and each element indicates whether the participant correctly recalled a word in a specific serial position for each list. For example, in the snippet below, we see the data for the first immediate free recall task for a patient with a FAST score of 3. The "1" in position X1 indicates that this patient correctly recalled the first item on this list. They also correctly recalled the third item and the last item. They did not correctly recall any other items on this list.

| FAST score | X1 | X2 | X3 | X4 | X5 | X6 | X7 | X8 | X9 | X10 |
|:----------:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| 3          | 1  | 0  | 1  | 0  | 0  | 0  | 0  | 0  | 0  | 1   |

Each patient has an associated 40-item vector containing data from the four 10-item free recall tasks. There were 2,647 patients in this dataset, resulting in a data matrix with dimensions 2,647 x 40.

The Neural Network
------------------
The goal of the neural network was the classify patients into FAST stages by extracting patterns from the free recall performance data. I split the data into training and testing sets by randomly sampling from the full dataset using a ratio of 80% training to 20% testing. 

### Version 1
The first version of the neural network was a multi-layer linear network. I used a sigmoid activation function, with the labels (the FAST score) in one-hot form. I also used an MSE loss function and the Adam optimizer:

    # create nn
    net = torch.nn.Sequential(torch.nn.Linear(40, 100), torch.nn.Sigmoid(),
                              torch.nn.Linear(100, 100), torch.nn.Sigmoid(),
                              torch.nn.Linear(100, 6), torch.nn.Sigmoid())
                              
    # create loss function
    mse_loss = torch.nn.MSELoss()

    # create optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    
In this version of the neural network, the training and testing accuracy reached asymptote quickly. Even after training for 150 epochs, both the train and test accuracy was close to 50% (train: 0.4851, test: 0.4813) (see Figure 1). 

###### Figure 1: Train and Test Accuracy for Neural Network (Version 1)
<img src="https://github.com/hollywestfall/nnml/blob/master/first_nn.png" width="600" height="400">

While this network achieved an accuracy of greater than chance (1/6 = 0.1667), there was a lot of room for improvement. After consulting with an expert in the field, I implemented a number of recommended changes in the second version of the network.

### Version 2
In the second version of the neural network, I standarized the data set to avoid having only positive values. I switched to a cross entropy loss function and left the labels in integer form. I also used a ReLU activation function:

    # create nn
    net = torch.nn.Sequential(torch.nn.Linear(40, 100), torch.nn.ReLU(), 
                              torch.nn.Linear(100, 40), torch.nn.ReLU(),
                              torch.nn.Linear(40, 6))
                          
    # create loss function
    xent_loss = torch.nn.CrossEntropyLoss()

    # create optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

Implementing these changes substantially improved the accuracy of the neural network. After training for 200 epochs, the training accuracy reached 0.9650 and the testing accuracy reached 0.8411 (see Figure 2).

###### Figure 2: Train and Test Accuracy for Neural Network (Version 2)
<img src="https://github.com/hollywestfall/nnml/blob/master/second_nn.png" width="600" height="400">

With these improvments, the new network correctly classified the patient by FAST stage 5 out of 6 times (a five-fold improvement over chance performance!). This result is particularly remarkable, because the FAST categorization is determined by the clinician independently of the free recall performance.

Next Steps
----------
While this binary data set captures patterns of primacy (a tendency to recall items earlier in the list) and recency (a tendency to recall items later in the list) in the data, it does not yield any information about the specific word that was recalled or the order in which words were recalled. Possible next steps are to include additional information collected during the free recall task about the order of word recall, or to incorporate the outcomes of other tasks in the MCIS, such as the recogntion memory task or the triadic comparison task.

References
----------

Reisberg, B. (1988). Functional assessment staging (FAST). *Psychopharmacology Bulletin, 24*(4), 653-659.

Storandt, M., & Hill, R. D. (1989). Very mild senile dementia of the Alzheimer type: II. Psychometric test performance. *Archives of Neurology, 46*(4), 383-386.

Tulving, E. (1972). Episodic and semantic memory. In E. Tulving, & W. Donaldson (Eds.), Organization of memory, (pp. 381-403). Academic Press.

Wilson, R. S., Bacon, L. D., Fox, J. H., & Kaszniak, A. W. (1983). Primary memory and secondary memory in dementia of the Alzheimer type. *Journal of Clinical and Experimental Neuropsychology, 5*(4), 337-344.
