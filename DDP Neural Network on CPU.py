# Basic imports
import os
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm    
    
# Visualization imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
    
# Scikit-learn imports
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    
# Dask-related imports
import dask
import dask.array as da
import dask.dataframe as dd
from dask_ml.impute import SimpleImputer
from dask.distributed import Client, config
from dask.diagnostics import ProgressBar
from dask_ml.model_selection import GridSearchCV,train_test_split
    
# Set the daemon configuration for Dask workers
dask.config.set({'distributed.worker.daemon': False})
    
# PyTorch imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
print("Library Imports done")

def dataloading(path='./dataset.csv'):
    astro_ds = dd.read_csv(path)
    # Dropping irrelavant columns early in the process
    columns_to_drop = ['id', 'full_name', 'pdes', 'name', 'prefix', 'neo', 'pha', 'orbit_id', 'equinox']
    astro_ds = astro_ds.drop(columns_to_drop, axis=1)
    print("Data Loaded and dropped irrelavant columns!!")
    return astro_ds

def nullvalue_handling(astro_ds):
    # Prepare features for imputation (excluding the target 'class')
    X = astro_ds.drop(['class'], axis=1)
    y = astro_ds['class']
    # Applying SimpleImputer from Dask-ML to handle missing values in parallel
    imputer = SimpleImputer(strategy='median')

    # Since Dask works with lazy evaluation, use compute() to perform the computation
    X_imputed = imputer.fit_transform(X)
    # Ensure that imputation and other transformations are computed efficiently

    with tqdm(total=2, desc="Computing DataFrames") as pbar:
        X_imputed = X_imputed.compute()
        pbar.update(1)  # Update progress after computing X_imputed
        y = y.compute()
        pbar.update(1)  # Update progress after computing y
    print("Null Value Handling Done!")
    return X_imputed,y

def feature_importance(X_imputed,y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2,shuffle =True ,random_state=42)
    # Initialize the ExtraTreesClassifier
    # We can adjust n_estimators, max_depth, and other parameters as needed
    etc = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # To leverage parallel computation with multiprocessing when performing model fitting
    with joblib.parallel_backend('multiprocessing'):
        #Fit the model
        etc.fit(X_train, y_train)
    # Compute and print feature importances
    feature_importances = etc.feature_importances_
    
    threshold = np.mean(feature_importances)  # Define your threshold here
    
    # Selecting features with importance greater than the threshold
    selected_features = [feature for feature, importance in zip(X_train.columns, feature_importances) if importance > threshold]
    
    # Include 'class' in selected features
    selected_features.append('class')
    
    # Filter the original Dask DataFrame to include only selected features
    astro_ds_filtered = astro_ds[selected_features]
    print("Feature Extraction done!")
    return astro_ds_filtered


def dataprep(X_train,X_test,y_train,y_test):
    # Scaling the data using Dask-ML's StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Encoding the target label using LabelEncoder
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    print("Scaling and Encoding Done!")
    return X_train_scaled,X_test_scaled,y_train_encoded,y_test_encoded

def tensordataset(X_train, y_train, X_test, y_test):
    # Convert test and train sets to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    #Generation of Train and Test Tensor dataset    
    train_dataset=TensorDataset(X_train_tensor,y_train_tensor)
    val_dataset=TensorDataset(X_test_tensor,y_test_tensor)
    input_size=X_train_tensor.shape[1]
    
    print(f"Tensor Conversion and dataset prep done!!Input Size:{input_size}")
    return train_dataset,val_dataset,input_size

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Linear layer mapping input to hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Linear layer mapping hidden layer to output classes

    def forward(self, x):
        out = self.fc1(x)  # Pass input through first linear layer
        out = self.relu(out)  # Apply ReLU activation function
        out = self.fc2(out)  # Pass through second linear layer to get class scores
        return out

class RobustNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RobustNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # First linear layer
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization for first hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization with 50% probability
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)  # Second linear layer reducing size
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)  # Batch normalization for second hidden layer
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)  # Final linear layer to output classes

    def forward(self, x):
        x = self.layer1(x)  # Pass input through the first layer
        x = self.bn1(x)  # Apply batch normalization
        x = self.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.layer2(x)  # Pass through the second linear layer
        x = self.bn2(x)  # Apply batch normalization
        x = self.relu(x)  # Apply ReLU activation
        x = self.layer3(x)  # Output layer to get class scores
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '65150'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Initialization complete. Using world size {world_size}.")

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size,input_size,train_dataset,val_dataset, batch_size=32, num_epochs=1):
    setup(rank, world_size)
    
    model = RobustNN(input_size=input_size, hidden_size=13, num_classes=13)
    #model = SimpleNN(input_size=input_size, hidden_size=13, num_classes=13)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None)  # device_ids must be None to use CPU
  
    # Create samplers and loaders for training and validation datasets
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, pin_memory=False, num_workers=0)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, pin_memory=False, num_workers=0)

    # Setup optimizer and loss function
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Metrics to keep track of progress
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epoch_times = []
    
    total_start_time = time.time()
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)  # Ensures proper shuffling per epoch
        train_loss = 0.0
        correct = 0
        total = 0
        ddp_model.train()  # Set model to training mode

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Log training metrics
        epoch_loss = train_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracies.append(100 * correct / total)
        print(f"[Rank {rank}] Epoch {epoch+1} average loss: {epoch_loss}")

        # Validation Logic
        ddp_model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
   
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # Log validation metrics
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)
        print(f"[Rank {rank}] Epoch {epoch+1} validation loss: {val_loss / len(val_loader)}")
        print(f"[Rank {rank}] Epoch {epoch+1} validation accuracy: {100 * correct / total:.4f}")

        ddp_model.train()  # Switch back to training mode
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        print(f"Rank {rank} and Epoch {epoch+1} Time Taken:{epoch_time:.2f} seconds")
    total_training_time = time.time() - total_start_time
    print(f"Rank {rank}: Total training time: {total_training_time:.2f} seconds")
    cleanup()
    print("Training Done!")

if __name__ == '__main__':
    # Load dataset from CSV
    astro_ds = dataloading(path='./dataset.csv')
    
    # Handle null values and separate features and labels
    X_imputed, y = nullvalue_handling(astro_ds)
    
    # Select important features based on some criteria
    astro_ds_filtered = feature_importance(X_imputed, y)
    
    # Re-handle null values after filtering features
    X_imputed, y = nullvalue_handling(astro_ds_filtered)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, shuffle=True, random_state=42)
    
    # Prepare data by scaling features and encoding labels
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = dataprep(X_train, X_test, y_train, y_test)
    
    # Convert datasets to tensor datasets suitable for PyTorch
    train_dataset, val_dataset, input_size = tensordataset(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded)

    # Start timing the distributed training setup
    start_time = time.time()
    
    #Number of CPUs
    world_size = 3
#    world_size = mp.cpu_count()-4
    print(f"World Size:{world_size}")
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(target=train, args=(rank, world_size,input_size,train_dataset,val_dataset,32,1))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Elapsed time: {elapsed_time} seconds")    