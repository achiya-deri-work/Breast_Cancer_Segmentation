import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import time
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.transforms import v2

from Functions import randint_distinct, get_exp_fn
from models import UNet, RUNet, AttUNet, RAttUNet

torch.random.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


folds_names = ['benign', 'malignant', 'normal']
wd_path = os.getcwd().replace('\\', '/')  + "/python_data/breast_cancer_ultrasound_imgs"
image_sz = 256

X = torch.tensor([], dtype=torch.float32) # Images (ultrasound images of breasts)
y = torch.tensor([], dtype=torch.float32) # Masks (ground truth masks ultrasound images of breasts)

for fold_name in folds_names:
    imgs = os.listdir(os.path.join(wd_path, fold_name))
    for img in tqdm(imgs):
        if "mask" not in img:
            img_array = cv.imread(os.path.join(os.path.join(wd_path, fold_name), img), cv.IMREAD_GRAYSCALE)
            img_array = cv.resize(img_array, (image_sz, image_sz))

            masks = [mask for mask in imgs if "mask" in mask and img.split('.')[0] in mask] 
            masks = [cv.imread(os.path.join(os.path.join(wd_path, fold_name), mask), cv.IMREAD_GRAYSCALE) for mask in masks]

            mask = masks[0]
            for i in range(1, len(masks)):
                mask += masks[i]

            mask = np.clip(cv.resize(mask, (image_sz, image_sz)), a_min=0, a_max=255)

            X = torch.cat((X, torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)), dim = 0)
            y = torch.cat((y, torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)), dim = 0)

# X = X.transpose(1, 2).transpose(2, 3) / 255
X = X / 255
y = y / 255

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

def compute_accuracy(predicted, target):
    # Compute the absolute difference between the predicted and target values
    diff = (predicted - target).abs()

    # Compute the mean of the absolute difference
    mae = diff.mean()

    # Compute the accuracy as 1 - MAE
    accuracy = 1 - mae

    return accuracy


# # Initialize Reccurent U-Net model
# model_1 = RUNet(1, 1, 32, 0.5).to(device)

# # Define Hyperparameters
# batch_size = 16
# learning_rate = 3e-4
# num_epochs = 6000

# # Define the loss function and optimizer
# loss_fn = nn.BCELoss()
# optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     # Get random batches of data and one hot encode the labels
#     batch_idxs = randint_distinct(0, X_train.size(0), batch_size)
#     X_batch, y_batch = X_train[batch_idxs], y_train[batch_idxs]

#     # Forward pass, compute loss
#     y_pred = model_1(X_batch)
#     loss = loss_fn(y_pred, y_batch)
#     train_acc = compute_accuracy(y_pred, y_batch)

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Evaluate the model on a test set every 10 epochs
#     if (epoch + 1) % 20 == 0:
#         # model.epochs_done += 10
#         model_1.eval()
#         batch_idxs = randint_distinct(0, X_test.size(0), X_test.size(0))
#         test_loss = 0
#         test_acc = 0
#         if train_acc > 0.9:
#             with torch.no_grad():
#                 for i in range(13):
#                     # Get random batches of data and one hot encode the labels
#                     X_batch, y_batch = X_test[i*X_test.size(0)//13:(i+1)*X_test.size(0)//13], y_test[i*X_test.size(0)//13:(i+1)*X_test.size(0)//13]

#                     test_logits = model_1(X_batch)
#                     test_loss += loss_fn(test_logits, y_batch).item()
#                     test_acc += compute_accuracy(test_logits, y_batch).item()
        
#         test_loss /= 13
#         test_acc /= 13
#         model_1.train()

#         # Print the loss and accuracy
#         print(f"Epoch: {epoch+1} | Train loss: {loss.item():.4f} | Train Acc: {train_acc.item():.4f} | Test loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Learning Rate: {optimizer.param_groups[0]['lr']}\n")

#         # Save the model if the test accuracy is higher than 0.8 and higher than the previous best accuracy
#         if test_acc > 0.6 and test_acc > model_1.best_acc:
#             # Get current model accuracy and index
#             model_1.best_acc = test_acc
#             idxs = [int(file.split("Idx")[1][:-4]) for file in os.listdir(os.getcwd().replace('\\', '/') + "/python_data/breast_cancer_ultrasound_imgs" + "/saves") if "Idx" in file]
#             idx = max(idxs) + 1 if idxs else 0
            
#             # Save the model
#             torch.save(model_1.state_dict(), os.getcwd().replace('\\', '/') + "/python_data/breast_cancer_ultrasound_imgs" + f"/saves/RSegModel{test_acc:.5f}Idx{idx}.pth")
#             print(f"Model saved successfully with accuracy: {test_acc:.4f}")

#     if epoch % 2000 == 0:
#         learning_rate = (3 - epoch // 2000) * 1e-4
#         optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

# Initialize Attention U-Net model
# model_2 = AttUNet(1, 1, 32, 0.4).to(device)

# # Define Hyperparameters
# batch_size = 16
# learning_rate = 3e-4
# num_epochs = 6000

# # Define the loss function and optimizer
# loss_fn = nn.BCELoss()
# optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     # Get random batches of data and one hot encode the labels
#     batch_idxs = randint_distinct(0, X_train.size(0), batch_size)
#     X_batch, y_batch = X_train[batch_idxs], y_train[batch_idxs]

#     # Forward pass, compute loss
#     y_pred = model_2(X_batch)
#     loss = loss_fn(y_pred, y_batch)
#     train_acc = compute_accuracy(y_pred, y_batch)

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Evaluate the model on a test set every 10 epochs
#     if (epoch + 1) % 20 == 0:
#         # model.epochs_done += 10
#         model_2.eval()
#         batch_idxs = randint_distinct(0, X_test.size(0), X_test.size(0))
#         test_loss = 0
#         test_acc = 0
#         if train_acc > 0.9:
#             with torch.no_grad():
#                 for i in range(13):
#                     # Get random batches of data and one hot encode the labels
#                     X_batch, y_batch = X_test[i*X_test.size(0)//13:(i+1)*X_test.size(0)//13], y_test[i*X_test.size(0)//13:(i+1)*X_test.size(0)//13]

#                     test_logits = model_2(X_batch)
#                     test_loss += loss_fn(test_logits, y_batch).item()
#                     test_acc += compute_accuracy(test_logits, y_batch).item()
        
#         test_loss /= 13
#         test_acc /= 13
#         model_2.train()

#         # Print the loss and accuracy
#         print(f"Epoch: {epoch+1} | Train loss: {loss.item():.4f} | Train Acc: {train_acc.item():.4f} | Test loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Learning Rate: {optimizer.param_groups[0]['lr']:.4f}\n")

#         # Save the model if the test accuracy is higher than 0.8 and higher than the previous best accuracy
#         if test_acc > 0.95 and test_acc > model_2.best_acc:
#             # Get current model accuracy and index
#             model_2.best_acc = test_acc
#             idxs = [int(file.split("Idx")[1][:-4]) for file in os.listdir(os.getcwd().replace('\\', '/') + "/python_data/breast_cancer_ultrasound_imgs" + "/saves") if "Idx" in file]
#             idx = max(idxs) + 1 if idxs else 0
            
#             # Save the model
#             torch.save(model_2.state_dict(), os.getcwd().replace('\\', '/') + "/python_data/breast_cancer_ultrasound_imgs" + f"/saves/AttSegModel{test_acc:.5f}Idx{idx}.pth")
#             print(f"Model saved successfully with accuracy: {test_acc:.4f}")

#     if epoch % 2000 == 0:
#         learning_rate = (3 - epoch // 2000) * 1e-4
#         optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
        
        
# Initialize Reccurrent Attention U-Net model
model_3 = UNet(1, 1, 32, 0.3).to(device)
start = time.time()

# Define Hyperparameters
batch_size = 16
learning_rate = 3e-4
num_epochs = 6000

# Define the loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model_3.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # Get random batches of data and one hot encode the labels
    batch_idxs = randint_distinct(0, X_train.size(0), batch_size)
    X_batch, y_batch = X_train[batch_idxs], y_train[batch_idxs]

    # Forward pass, compute loss
    y_pred = model_3(X_batch)
    loss = loss_fn(y_pred, y_batch)
    train_acc = compute_accuracy(y_pred, y_batch)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluate the model on a test set every 10 epochs
    if (epoch + 1) % 20 == 0:
        # model.epochs_done += 10
        model_3.eval()
        batch_idxs = randint_distinct(0, X_test.size(0), X_test.size(0))
        test_loss = 0
        test_acc = 0
        if train_acc > 0.9:
            with torch.no_grad():
                for i in range(13):
                    # Get random batches of data and one hot encode the labels
                    X_batch, y_batch = X_test[i*X_test.size(0)//13:(i+1)*X_test.size(0)//13], y_test[i*X_test.size(0)//13:(i+1)*X_test.size(0)//13]

                    test_logits = model_3(X_batch)
                    test_loss += loss_fn(test_logits, y_batch).item()
                    test_acc += compute_accuracy(test_logits, y_batch).item()
        
        test_loss /= 13
        test_acc /= 13
        model_3.train()

        # Print the loss and accuracy
        print(f"Epoch: {epoch+1} | Time elapsed (mins): {(time.time() - start)/60:.4f} | Train loss: {loss.item():.4f} | Train Acc: {train_acc.item():.4f} | Test loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Learning Rate: {optimizer.param_groups[0]['lr']}\n")

        # Save the model if the test accuracy is higher than 0.8 and higher than the previous best accuracy
        if test_acc > 0.5 and test_acc > model_3.best_acc:
            # Get current model accuracy and index
            model_3.best_acc = test_acc
            idxs = [int(file.split("Idx")[1][:-4]) for file in os.listdir(os.getcwd().replace('\\', '/') + "/python_data/breast_cancer_ultrasound_imgs" + "/saves") if "Idx" in file]
            idx = max(idxs) + 1 if idxs else 0
            
            # Save the model
            torch.save(model_3.state_dict(), os.getcwd().replace('\\', '/') + "/python_data/breast_cancer_ultrasound_imgs"  + f"/saves/RAttSegModel{test_acc:.5f}Idx{idx}.pth")
            print(f"Model saved successfully with accuracy: {test_acc:.4f}")


    if epoch % 2000 == 0:
        learning_rate = (3 - epoch // 2000) * 1e-4
        optimizer = torch.optim.Adam(model_3.parameters(), lr=learning_rate)










