# Databricks notebook source
# MAGIC %md
# MAGIC # Deep Learning with Microsoft Fabric

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing PyTorch in our environment

# COMMAND ----------

# MAGIC %pip install torch

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Dataset into a Spark Dataframe

# COMMAND ----------

df = spark.read.format("csv").option("header","true").load("Files/diabetes.csv").dropna()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preparation and Feature Engineering

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split

# Split the data into training and testing datasets   
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' , 'DiabetesPedigreeFunction' , 'Age' ]
label = 'Outcome'

# Split data 70%-30% into training set and test set
x_train, x_test, y_train, y_test = train_test_split(df.toPandas()[features].values,
                                                    df.toPandas()[label].values,
                                                    test_size=0.30,
                                                    random_state=0)
   
x_train = x_train.astype(np.float32)  # Or np.int64, based on your needs
y_train = y_train.astype(np.int64)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.int64)

print ('Training Set: %d rows, Test Set: %d rows \n' % (len(x_train), len(x_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing PyTorch in our CodeBase

# COMMAND ----------

import torch
import torch.nn as nn
import torch.utils.data as td
import torch.nn.functional as F
   
# Set random seed for reproducability
torch.manual_seed(0)
   
print("Libraries imported - ready to use PyTorch", torch.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating Data Loaders
# MAGIC

# COMMAND ----------

# Create a dataset and loader for the training data and labels
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x,train_y)
train_loader = td.DataLoader(train_ds, batch_size=20,
    shuffle=False, num_workers=1)

# Create a dataset and loader for the test data and labels
test_x = torch.Tensor(x_test).float()
test_y = torch.Tensor(y_test).long()
test_ds = td.TensorDataset(test_x,test_y)
test_loader = td.DataLoader(test_ds, batch_size=20,
                             shuffle=False, num_workers=1)
print('Ready to load data')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining the Neural Network

# COMMAND ----------

h1 = 10 

# Define the neural network
class DiabetesNet(nn.Module):
 def __init__(self):
    super(DiabetesNet, self).__init__()
    self.fc1 = nn.Linear(len(features), h1) # defining the input layer
    self.fc2 = nn.Linear(h1,h1) # defining the hidden layers
    self.fc3 = nn.Linear(h1,2) # defining the output layer

 def forward(self, x):
    fc1_output = torch.relu(self.fc1(x))
    fc2_output = torch.relu(self.fc2(fc1_output))
    y = F.log_softmax(self.fc3(fc2_output).float(), dim=1)
    return y

# Create a model instance from the network
model = DiabetesNet()
print(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Functions to Test and Train a Neural Network Model

# COMMAND ----------

def train(model, data_loader, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # set the model to training mode
    model.train()
    train_loss=0

    for batch,tensor in enumerate(data_loader):
        data, target = tensor
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss = train_loss + loss.item()

        # backpropagate adjustments to the weight
        loss.backward()
        optimizer.step()
    
    # Return the average loss 
    avg_loss = train_loss / (batch+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def test(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # switch the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
         batch_count = 0
         for batch, tensor in enumerate(data_loader):
             batch_count += 1
             data, target = tensor
             # get the predictions
             out = model(data)

             # calculate the loss
             test_loss = loss_criteria(out, target).item() + test_loss

             # calculate the accuracy
             _, predicted = torch.max(out.data,1)
             correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss/batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
       
    # return average loss for the epoch
    return avg_loss
     

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training the Model

# COMMAND ----------

# Specify the loss criteria (we'll use CrossEntropyLoss for multi-class classification)
loss_criteria = nn.CrossEntropyLoss()
   
# Use an optimizer to adjust weights and reduce loss
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()
   
# We'll track metrics for each epoch in these arrays
epoch_nums = []
training_loss = []
validation_loss = []
   
# Train over 100 epochs
epochs = 100
for epoch in range(1, epochs + 1):
   
    # print the epoch number
    print('Epoch: {}'.format(epoch))
       
    # Feed training data into the model
    train_loss = train(model, train_loader, optimizer)
       
    # Feed the test data into the model to check its performance
    test_loss = test(model, test_loader)
       
    # Log the metrics for this epoch
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review the Training and Validation Loss

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC from matplotlib import pyplot as plt
# MAGIC    
# MAGIC plt.plot(epoch_nums, training_loss)
# MAGIC plt.plot(epoch_nums, validation_loss)
# MAGIC plt.xlabel('epoch')
# MAGIC plt.ylabel('loss')
# MAGIC plt.legend(['training', 'validation'], loc='upper right')
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the Learned Weights and Biases

# COMMAND ----------

for param_tensor in model.state_dict():
    print(param_tensor, "\n", model.state_dict()[param_tensor].numpy())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the Trained Model

# COMMAND ----------

import os
model_file = "/lakehouse/default/Files/diabetes_predictor.pt"

# Ensure the parent directory exists
parent_dir = os.path.dirname(model_file)
os.makedirs(parent_dir, exist_ok=True)

# Save the model weights
torch.save(model.state_dict(), model_file)
del model
print('Model saved as', model_file)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference the Saved Model

# COMMAND ----------

# New Diabetes Features
x_new = [[8,85,65,29,0,26.6,0.672,32]]
print ('New sample: {}'.format(x_new))
   
# Create a new model class and load weights
model = DiabetesNet()
model.load_state_dict(torch.load(model_file))
   
# Set model to evaluation mode
model.eval()
   
# Get a prediction for the new data sample
x = torch.Tensor(x_new).float()
_, predicted = torch.max(model(x).data, 1)
   
print('Prediction:',predicted.item())