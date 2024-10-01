import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.guided_dropout_model import GuidedDropoutModel
from models.regular_dropout_model import RegularDropoutModel
from models.no_dropout_model import NoDropoutModel
import argparse
import os

def train(model, device, train_loader, optimizer, epoch, total_epochs):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}/{total_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f}')
    accuracy = 100. * correct / total
    print(f'Training Accuracy: {accuracy:.2f}%')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0  # Keep track of total samples for accuracy
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    test_loss /= total
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Guided Dropout CIFAR-10 Experiment')
    parser.add_argument('--model', type=str, default='guided', choices=['guided', 'regular', 'none'],
                        help='Model type: guided, regular, or none')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    args = parser.parse_args()

    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize model
    if args.model == 'guided':
        model = GuidedDropoutModel(dropout_rate=0.2).to(device)
    elif args.model == 'regular':
        model = RegularDropoutModel(dropout_rate=0.2).to(device)
    else:
        model = NoDropoutModel().to(device)

    # Define optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Learning rate scheduler: reduce LR by factor of 10 every 50 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    total_epochs = args.epochs

    if args.model == 'guided':
        # Training schedule for Guided Dropout
        strength_learning_epochs = 40
        guided_dropout_schedule = [
            {'epochs': 60, 'dropout_rate': 0.2},
            {'epochs': 50, 'dropout_rate': 0.15},
            {'epochs': 50, 'dropout_rate': 0.1}
        ]
        current_schedule_index = 0
        schedule_epoch_counter = 0
    else:
        # For regular dropout and no dropout, use consistent dropout rate
        strength_learning_epochs = 0  # No strength learning phase

    for epoch in range(1, total_epochs + 1):
        print(f"Starting Epoch {epoch}/{total_epochs}")
        if args.model == 'guided':
            if epoch <= strength_learning_epochs:
                # Disable dropout during strength learning phase
                model.dropout_enabled = False
                print("Strength Learning Phase: Dropout Disabled")
            else:
                # Enable guided dropout
                model.dropout_enabled = True
                # Update dropout rate according to schedule
                if schedule_epoch_counter >= guided_dropout_schedule[current_schedule_index]['epochs']:
                    # Move to next schedule
                    current_schedule_index += 1
                    schedule_epoch_counter = 0
                    if current_schedule_index >= len(guided_dropout_schedule):
                        # No more schedules left
                        current_schedule_index = len(guided_dropout_schedule) - 1  # Stay at last schedule
                # Set dropout rate
                model.dropout_rate = guided_dropout_schedule[current_schedule_index]['dropout_rate']
                schedule_epoch_counter += 1
                print(f"Guided Dropout Enabled: Dropout Rate = {model.dropout_rate}")
        else:
            # For regular dropout and no dropout
            if args.model == 'regular':
                print("Regular Dropout Enabled")
            else:
                print("No Dropout Applied")

        train(model, device, train_loader, optimizer, epoch, total_epochs)
        accuracy = test(model, device, test_loader)
        scheduler.step()  # Adjust learning rate

        # Save the model after training
        model_filename = os.path.join('trained_models', f'model_{args.model}.pth')
        torch.save(model.state_dict(), model_filename)
        print(f'Model saved as {model_filename}')

if __name__ == '__main__':
    main()
