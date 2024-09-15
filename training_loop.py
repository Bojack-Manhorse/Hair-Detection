import torch

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# The training loop. Takes in a model, the training and validation data loaders, the number of epochs and the initial learning rate
def train(model, train_loader, epochs = 10, learning_rate = 0.01, model_name:str = "My Model"):

    torch.cuda.empty_cache()

    # Set the optimiser to be an instance of the stochastic gradient descent class
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(parameters, lr=learning_rate)

    # Define a learning rate scheduler as an instance of the ReduceLROnPlateau class
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=50, cooldown=7, eps=1e-20)

    # Writer will be used to track model performance with TensorBoard
    writer = SummaryWriter()

    # Keep track of the number of batches to plot model performace against
    batch_index = 0

    # Loop over the number of epochs
    for epoch in range(epochs):

        # Within each epoch, we pass through the entire training data in batches indexed by batch
        for batch in train_loader:
            # Loads features and labels into device for performance improvements
            features, labels = batch

            model.train()
            
            features = list(img.to(device) for img in features)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            loss_dict = model(features, labels)

            # Calculate the loss via cross_entropy
            loss = sum(loss for loss in loss_dict.values())

            # Create the grad attributes
            loss.backward() 

            # Clip the loss value so it doesn't become NaN
            torch.nn.utils.clip_grad_norm_(parameters, 4)

            # Print the performance
            print(f"Epoch: {epoch}, batch index: {batch_index}, learning rate: {scheduler.get_last_lr()}, loss:{loss.item()}")

            # Perform one step of stochastic gradient descent
            optimiser.step()

            # Zero the gradients (Apparently set_to_none=True imporves performace)
            optimiser.zero_grad(set_to_none=True)

            # Feed the loss amount into the learning rate scheduler to decide the next learning rate
            scheduler.step(loss.item())

            # Write the performance to the TensorBoard plot
            writer.add_scalar('loss', loss.item(), batch_index)

            # Increment the batch index
            batch_index += 1
            
            torch.cuda.empty_cache()
    
    # Save the model to the path specified in 'model_name'
    print(f"Saving model as {model_name}.pt")
    torch.save(model.state_dict(), f'{model_name}.pt')
        

