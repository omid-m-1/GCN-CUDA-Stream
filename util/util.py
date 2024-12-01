import torch
from torch.cuda.amp import autocast, GradScaler

# Training function
def train_model(model, features, labels, train_mask, criterion, optimizer, epochs=200, device='cuda', mixed=False):
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    if mixed : scaler = GradScaler()

    for epoch in range(epochs):
        model.train()

        # Zero gradients
        optimizer.zero_grad()

        if mixed:
            with autocast():
                # Forward pass
                output = model(features)
                loss = criterion(output[train_mask], labels[train_mask])
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            output = model(features)
            loss = criterion(output[train_mask], labels[train_mask])
            # Backward pass
            loss.backward()
            optimizer.step()

        if (epoch+1) % 500 == 0:
            _, predicted = output.max(dim=1)
            correct = predicted[train_mask].eq(labels[train_mask]).sum().item()
            accuracy = correct / train_mask.sum().item() * 100
            train_loss = loss.item() / train_mask.sum().item()
            print(f'Epoch [{epoch + 1}] - Train Loss: {train_loss:.5} | Train Accuracy: {accuracy:.2f}%')

# Test function
def evaluate_model(model, features, labels, test_mask, criterion, device='cuda', mixed=False):
    model.eval() # Set model to evaluation mode
    features = features.to(device)
    labels = labels.to(device)
    test_mask = test_mask.to(device)
    with torch.no_grad():
        if mixed:
            with autocast():
                # Forward pass
                output = model(features)
                loss = criterion(output[test_mask], labels[test_mask])
        else:
            # Forward pass
            output = model(features)
            loss = criterion(output[test_mask], labels[test_mask])
        _, predicted = output.max(dim=1)
        correct = predicted[test_mask].eq(labels[test_mask]).sum().item()
        accuracy = correct / test_mask.sum().item() * 100
        test_loss = loss.item() / test_mask.sum().item()
        print(f'Test Loss: {test_loss:.5f} | Test Accuracy: {accuracy:.2f}%')

