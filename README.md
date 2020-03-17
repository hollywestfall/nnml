# Introduction
this is some sample text

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
