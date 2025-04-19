# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
The experiment aims to develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron. The model will be trained on a dataset containing images of various defected and non-defected capacitors to enhance defect detection accuracy. Optimization techniques will be applied to improve performance, and the model will be evaluated to ensure reliable classification for capacitor quality assessment in manufacturing.


## DESIGN STEPS
### STEP 1: Problem Statement  
Define the objective of distinguishing between defected and non-defected capacitors using a binary classification model based on a pretrained VGG19.  

### STEP 2: Dataset Collection  
Use a dataset containing images of defected and non-defected capacitors for model training and evaluation.  

### STEP 3: Data Preprocessing  
Resize images to match VGG19 input dimensions, normalize pixel values, and create DataLoaders for efficient batch processing.  

### STEP 4: Model Architecture  
Modify the pretrained VGG19 by replacing the last layer with a single neuron using a sigmoid activation function for binary classification.  

### STEP 5: Model Training  
Train the model using a suitable loss function (Binary Cross-Entropy) and optimizer (Adam) for multiple epochs to enhance defect detection accuracy.  

### STEP 6: Model Evaluation  
Evaluate the model on unseen data using accuracy, precision, recall, and an ROC curve to assess classification performance.  

### STEP 7: Model Deployment & Visualization  
Save the trained model, visualize predictions, and integrate it into a manufacturing quality assessment system if needed.

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(pretrained=True)


# Modify the final fully connected layer to match the dataset classes
in_features=model.classifier[-1].in_features
num_classes = len(train_dataset.classes)
model.classifier[-1] = nn.Linear(in_features, 1)


# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)



# Train the model
def train_model(model, train_loader, test_loader, num_epochs=50):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # Ensure proper shape for BCEWithLogitsLoss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Gokularamanan K      ")
    print("Register Number:212222230040     ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/e0d6a53d-d0fa-4e57-89a2-6fb5a4c2d7ba)

![image](https://github.com/user-attachments/assets/11b51e46-8afd-4588-94fe-42aed23f9407)



### Confusion Matrix

![image](https://github.com/user-attachments/assets/6abbc580-cbd3-4819-a9c1-1445eafc3102)



### Classification Report
![image](https://github.com/user-attachments/assets/7414cb4e-0a5c-4430-bb30-02dd8ab93d71)


### New Sample Prediction
![image](https://github.com/user-attachments/assets/70b73e2a-a33a-4b2e-a62c-be226e978bba)
![image](https://github.com/user-attachments/assets/ad39a1bb-914b-43b7-8880-4ddd1674b11d)



## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
