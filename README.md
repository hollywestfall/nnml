Classification of Functional Assessment from Free Recall Performance in Alzheimer's Patients
============================================================================================

Introduction
------------
Often when patients are tested for symptoms of Alzheimer’s disease, they are done through the use of brief cognitive measures, like tests of verbal abilities, categorization tasks, or memory tasks (CITE). People diagnosed with Alzheimer's disease tend to do poorly on episodic memory tasks (e.g., CITE CITE CITE).

Through a collaboration with a local Alzheimer's clinic, my lab has access to a large clinical data set of patients with Alzheimer’s disease. All patients were diagnosed with the Functional Assessment Staging Test (FAST; Reisberg, 1988) into stages ranging from 1-6. The FAST stage describes severity of symptoms in terms of one's ability to perform daily living activities, where higher FAST stages indicate greater severity of symptoms. For example, someone categorized into FAST stage 1 has no discernable deficits or symptoms. People categorized into FAST stage 3 need help with complicated tasks, like financial planning. But by the time people reach FAST stage 5 or 6, they need help with things like dressing and bathing. The following table presents a summary of FAST scores, typical characteristics, and the number of people in the data set with that score.

| FAST score | Characteristics                         | Patients |
|:----------:|:--------------------------------------- |:--------:|
| 1          | No deficits whatsoever                  | 99       |
| 2          | Subjective functional deficit           | 436      |
| 3          | Difficulty accomplishing complex tasks  | 813      |
| 4          | Requires help cooking & cleaning        | 944      |
| 5          | Requires help selecting proper clothing | 210      |
| 6          | Requires help dressing & bathing        | 145      |

For the 2,647 people in this data set, we have the results of a Mild Cognitive Impairment Screen (MCIS), which includes a battery of memory tasks, such as recognition memory, metamemory judgments, and triadic comparisons. However, for this project, I will focus on the task that relies heavily on episodic memory - the free recall task.

The Free Recall Task
--------------------
In a free recall task, the participant is presented with a list of items and then asked to recall as many items from the list as possible, in any order. In the MCIS, there are a total of four free recall tasks: three immediate free recall and one delayed free recall. The three immediate free recall tasks are delivered back-to-back-to-back with no delay between the presentation of the list and the recall of items. Participants hear a 10-item word list from the clinician, consisting of the words BUTTER, ARM, SHORE, LETTER, QUEEN, CABIN, POLE, TICKET, GRASS, ENGINE. For each of the immediate free recall tasks, words are presented in the same order each time within subjects. The word order is balanced between subjects.

After the final immediate free recall task is complete, the participant completes an unrelated task in the MCIS. This other task generally takes about 2 to 5 minutes to complete. Then, without showing the participant the list again, the participant is again asked to recall as many items from the list as possible. Importantly, this delayed recall task is a surprise to the participant.

The Data
--------
The dataset itself is binary and indicates whether the participant correctly recalled a word in a specific serial position for each list. For example, in the data snippet below, we see the data for the first immediate free recall task for a patient in FAST stage 3. The "1" in position X1 indicates that this patient correctly recalled the first item on the list. They also correctly recalled the third item and the last time. 

| FAST score | X1 | X2 | X3 | X4 | X5 | X6 | X7 | X8 | X9 | X10 |
|:----------:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| 3          | 1  | 0  | 1  | 0  | 0  | 0  | 0  | 0  | 0  | 1   |

Each person has an associated 40-item vector containing data from the four 10-item free recall tasks. There were 2,647 patients total, resulting in a data matrix with dimensions 2,647 x 40.

The Neural Network
------------------
The goal of the neural network was the classify patients into FAST stages by extracting patterns from the free recall performance data. Having the data set up in this way allows us to capture patterns of primacy and recency in the data. It does not capture information about the specific word that was recalled or the order in which words were recalled. 

### Version 1
The first version of the neural network was a multi-layer linear network. I used a sigmoid activation function, with my labels (the FAST score) in one-hot form. I also used an MSE loss function and the Adam optimizer:

    # create nn
    net = torch.nn.Sequential(torch.nn.Linear(40, 100), torch.nn.Sigmoid(),
                              torch.nn.Linear(100, 100), torch.nn.Sigmoid(),
                              torch.nn.Linear(100, 6), torch.nn.Sigmoid())
                              
    # create loss function
    mse_loss = torch.nn.MSELoss()

    # create optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    
However, the training and testing accuracy reached asymptote quickly. Even after training for 150 epochs, both the train and test accuracy was close to 50% (train: 0.4918, test: 0.4826) (see Figure 1). 

![alt text][first_nn]

[first_nn]: https://github.com/hollywestfall/nnml/blob/master/first_nn.png "My First Neural Network"

While this network achieved an accuracy of greater than chance (1/6 = 0.1667), there was a lot of room for improvement. After consulting with an expert in the field, I implemented a number of recommended changes in the second version of the network.

### Version 2
In the second version of the neural network, I standarized the data set to avoid having only positive values. I also switched to a cross entropy loss function and left the labels in integer form. I also used a ReLU activation function:

    # create nn
    net = torch.nn.Sequential(torch.nn.Linear(40, 100), torch.nn.ReLU(), 
                              torch.nn.Linear(100, 40), torch.nn.ReLU(),
                              torch.nn.Linear(40, 6))
                          
    # create loss function
    xent_loss = torch.nn.CrossEntropyLoss()

    # create optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

Implementing these changes substantially improved the accuracy of the neural network. After training for 200 epochs, the training accuracy reached 0.9651 and the testing accuracy reached 0.8392 (see Figure 2).

![alt text][second_nn]

[second_nn]: https://github.com/hollywestfall/nnml/blob/master/second_nn.png "My Second Neural Network"

With these improvments, the new network correctly classified the patient by FAST stage 5 out of 6 times (a five-fold improvement over chance performance!). This result is particularly remarkable, because the FAST categorization is made independently of the free recall performance.

Future Directions

Incorporate info from the same data set about the order in which words were recalled.
