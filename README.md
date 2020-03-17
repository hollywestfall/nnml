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
