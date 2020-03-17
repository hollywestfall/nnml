Classification of Functional Assessment from Free Recall Performance in Alzheimer's Patients
============================================================================================

Introduction
------------
Often when patients are tested for symptoms of Alzheimer’s disease, they are done through the use of brief cognitive measures, like tests of verbal abilities, categorization tasks, or memory tasks (CITE). People diagnosed with Alzheimer's disease tend to do poorly on episodic memory tasks (e.g., CITE CITE CITE).

Through a collaboration with a local Alzheimer's clinic, my lab has access to a large clinical data set of patients with Alzheimer’s disease. All patients were diagnosed with the Functional Assessment Staging Test (FAST; Reisberg, 1988) into stages ranging from 1-6. The FAST stage describes severity of symptoms in terms of one's ability to perform daily living activities, where higher FAST stages indicate greater severity of symptoms. For example, someone categorized into FAST stage 1 has no discernable deficits or symptoms. People categorized into FAST stage 3 need help with complicated tasks, like financial planning. But by the time people reach FAST stage 5 or 6, they need help with things like dressing and bathing. The following table presents a summary of FAST scores, typical characteristics, and the number of people in the data set with that score.

| FAST score | Characteristics                         | Number |
|:----------:|:--------------------------------------- |:------:|
| 1          | No deficits whatsoever                  | 99     |
| 2          | Subjective functional deficit           | 436    |
| 3          | Difficulty accomplishing complex tasks  | 813    |
| 4          | Requires help cooking & cleaning        | 944    |
| 5          | Requires help selecting proper clothing | 210    |
| 6          | Requires help dressing & bathing        | 145    |

For the 2,647 people in this data set, we have the results of a Mild Cognitive Impairment Screen (MCIS), which includes a battery of memory tasks, such as recognition memory, metamemory judgments, and triadic comparisons. However, for this project, I will focus on the task that relies heavily on episodic memory - the free recall task.

The Free Recall Task
--------------------
In a free recall task, the participant is presented with a list of items and then asked to recall as many items from the list as possible, in any order.

# Version 1
My first nn was not so great

Check out this here code:

    # create nn
    net = torch.nn.Sequential(torch.nn.Linear(40, 100), torch.nn.Sigmoid(),
                              torch.nn.Linear(100, 100), torch.nn.Sigmoid(),
                              torch.nn.Linear(100, 6), torch.nn.Sigmoid())
                              
    # create loss function
    mse_loss = torch.nn.MSELoss()

    # create optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    
How neat!

But the accuracy was not so good.

Let's insert a figure!

![alt text][first_nn]

[first_nn]: https://github.com/hollywestfall/nnml/blob/master/first_nn.png "My First Neural Network"

# Version 2
I made a bunch of changes! Check them out!

    # create nn
    net = torch.nn.Sequential(torch.nn.Linear(40, 100), torch.nn.ReLU(), 
                              torch.nn.Linear(100, 40), torch.nn.ReLU(),
                              torch.nn.Linear(40, 6))
                          
    # create loss function
    xent_loss = torch.nn.CrossEntropyLoss()

    # create optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

Oh boy, let's have a look now!

![alt text][second_nn]

[second_nn]: https://github.com/hollywestfall/nnml/blob/master/second_nn.png "My Second Neural Network"
