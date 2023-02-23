"""
The code in this file is derived from the 25-hour PyTorch lesson by Daniel Bourke through freeCodeCamp.org
https://www.youtube.com/watch?v=V_xro1bcAuA&t=24005s
Author: Logan Endes
"""
import time
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Check PyTorch version
print(torch.__version__)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



# Create known data
a = 4.2
b = -42
c = 69

start = 0
end = 10
step = 0.01
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = (a * (X)**2) + (b * X) + c

# Create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Create a function to visually represent the data
def plot_predictions(train_data = X_train, 
                     train_labels = y_train, 
                     test_data = X_test, 
                     test_labels = y_test,
                     predictions = None):
    """
    Plots training data, test data, and compares predictions to known data.
    Args:
        train_data (torch.tensor, optional): Array (tensor) representing the X axis of the training data. Defaults to X_train.
        train_labels (torch.tensor, optional): Array (tensor) representing the y axis of the training data. Defaults to y_train.
        test_data (torch.tensor, optional): Array (tensor) representing the X axis of the test data. Defaults to X_test.
        test_labels (torch.tensor, optional): Array (tensor) representing the y axis of the test data. Defaults to y_test.
        predictions (torch.tensor, optional): Array (tensor) representing the predicted output of the model. Defaults to None.
    """
    plt.figure(figsize=(10, 7))
    
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    # Plot the test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    
    # Check if predictions are given
    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
        
    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()

# Create a linear regression model class
class Parabola(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a * (x)**2) + (self.b * x) + self.c
    
# Create a random seed
SEED = 69
torch.manual_seed(SEED)

# Create an instance of the model
model_0 = Parabola()

# Make untrained predictions
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)

start = time.time()

# Choose a loss function
loss_fn = nn.L1Loss()

# Choose an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# Create a training and testing loop
epochs = 300000

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    # Set the model to training mode
    model_0.train()
    
    # 1. Forward pass
    y_pred = model_0(X_train)
    
    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)
    
    # Adjust the learning rate as the model gets more accurate (don't know if this is better than a scheduler or not)
    if loss < 4.6:
        if loss < .1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001
        else:  
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    
    # 4. Perform backpropagation
    loss.backward()
    
    # 5. Step the optimzier (gradient descent)
    optimizer.step()
    
    # Start the testing loop
    model_0.eval()
    with torch.inference_mode():
        # 1. Do the forward pass
        test_pred = model_0(X_test)
        
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)
        
    if epoch % 20 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
        print(model_0.state_dict())
     
endtime = time.time()   

def plot_loss(epoch_count=epoch_count, loss_values=loss_values, test_loss_values=test_loss_values):
    """
    Function to plot the loss curve given lists of epoch counts, training loss values, and test loss values
    Args:
        epoch_count (List, optional): List of epochs. Defaults to epoch_count.
        loss_values (List, optional): List of training loss values. Defaults to loss_values.
        test_loss_values (List, optional): List of test loss values. Defaults to test_loss_values.
    """
    # Plot the loss curves
    plt.plot(epoch_count, np.array(torch.tensor(loss_values).cpu().numpy()), label = "Training loss")
    plt.plot(epoch_count, test_loss_values, label = "Test loss")
    plt.title("Training and Test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    
plot_loss()

with torch.inference_mode():
    y_preds_new = model_0(X_test)
    
plot_predictions(predictions=y_preds_new)
print(f"Training took {endtime-start} seconds.")

# Saving the Parabola model

# 1. Create model's directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "parabola_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state_dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model_0.state_dict(), MODEL_SAVE_PATH)
