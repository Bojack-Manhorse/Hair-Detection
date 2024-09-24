import torch

from model import get_device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class TrainModel:
    """
    A class containing methods to train a neural network model.
    """
    def __init__(self, model, train_dataset:torch.utils.data.Dataset, validation_dataset:torch.utils.data.Dataset) -> None:
        """
        Initialises the training class.

        Args:
            model: The model to be trained
            train_dataset: The dataset to train on.
            validation_dataset: The dataset used for validation.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.device = get_device()

    def train_one_epoch(self, print_freq:int = 10):
        """
        Trains self.model for one epoch, using self.train_dataset.

        Args:
            print_freq: Frequency of number of batches to print training loss.
        """
        batch_index = 0

        for batch in self.train_loader:
            # Loads features and labels into device for performance improvements
            features, labels = batch

            self.model.train()
            
            features = list(img.to(self.device) for img in features)
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

            loss_dict = self.model(features, labels)

            # Calculate the loss via cross_entropy
            loss = sum(loss for loss in loss_dict.values())

            # Create the grad attributes
            loss.backward() 

            # Clip the loss value so it doesn't become NaN
            torch.nn.utils.clip_grad_norm_(self.parameters, 4)

            # Print the performance, if the batch index is a multiple of print_freq
            if batch_index % print_freq == 0:
                print(f"Epoch: {self.current_epoch}, batch index: {batch_index}, learning rate: {self.scheduler.get_last_lr()}, loss:{loss.item()}")

            # Perform one step of stochastic gradient descent
            self.optimiser.step()

            # Zero the gradients (Apparently set_to_none=True imporves performace)
            self.optimiser.zero_grad(set_to_none=True)

            # Feed the loss amount into the learning rate scheduler to decide the next learning rate
            self.scheduler.step(loss.item())

            # Write the performance to the TensorBoard plot
            self.writer.add_scalar('loss', loss.item(), self.current_epoch * len(self.train_loader) +  batch_index)

            # Increment the batch index
            batch_index += 1
            
            torch.cuda.empty_cache()

    def validate(self):
        """
        Calculates the average loss across the validation dataset.

        Returns:
            accuracy_score: The average loss of each item in the validation dataset.
        """
        with torch.no_grad():
            losses = torch.zeros(0).to(self.device)

            for _, batch in enumerate(self.valid_loader):

                features, labels = batch

                self.model.train()

                features = list(img.to(self.device) for img in features)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

                loss_dict = self.model(features, labels)
                loss = sum(loss for loss in loss_dict.values())

                losses = torch.cat((losses, loss.view(1)))

            accuracy_score = torch.sum(losses) / len(losses)

            return accuracy_score

    def train_n_epochs(self, batch_size:int, num_epochs:int, initial_learning_rate:float = 1):
        """
        Trains self.model for a number of epochs.

        Args:
            batch_size: The batch size to use in the train and validation dataloaders.
            num_epochs: The number of epochs (passes over the training dataset) to perform.
            initial_learning_rate: The initial learning rate, will be modified during training via self.scheduler.
        """

        # Picks out the parameters that we will be updating.
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        # Sets the optimiser.
        self.optimiser = torch.optim.SGD(self.parameters, lr=initial_learning_rate)

        # Sets the learning rate scheduler.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', patience=50, cooldown=7, eps=1e-20)

        # Initialises the number of epochs past, starts at 0.
        self.current_epoch = 0

        # Create a summary writer for tensorboard.
        self.writer = SummaryWriter()

        # Create trainig and validation dataloaders from thier respective datasets.
        self.train_loader = self.get_dataloader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = self.get_dataloader(self.validation_dataset, batch_size=batch_size, shuffle=False)

        # Iterate over the number of epochs.
        while self.current_epoch < num_epochs:
            print(f"Starting epoch {self.current_epoch}.")
            self.train_one_epoch()
            print(f"Completed epoch {self.current_epoch}, calculating validation accuracy.")
            validation_accuracy = self.validate()
            print(f"Epoch {self.current_epoch} validation score: {validation_accuracy}.")
            self.current_epoch += 1
        

    def get_model(self):
        """
        Returns self.model
        """
        return self.model
    
    def save_model_parameters(self, file_path:str):
        """
        Save the model parameters to file_path.
        """
        torch.save(self.model.state_dict(), file_path)

    @staticmethod
    def get_dataloader(dataset:torch.utils.data.Dataset, batch_size:int, shuffle:bool = False):
        """
        Static method to create a dataloader from a dataset.
        """
        return DataLoader(
            dataset,
            batch_size = batch_size,
            collate_fn=lambda batch: tuple(zip(*batch)),
            shuffle=shuffle
        )
