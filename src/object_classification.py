import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from dataset import EEGImageNetDataset
from de_feat_cal import de_feat_cal, de_feat_temp
from model.simple_model import SimpleModel
from model.eegnet import EEGNet
from model.mlp import MLP
from model.mlplus import MLPlus
from model.rgnn import RGNN, get_edge_weight
from model.lstm import LSTM
from utilities import *


def model_init(args, if_simple, num_classes, device):
    if if_simple:
        _model = SimpleModel(args)
    elif args.model.lower() == 'eegnet':
        _model = EEGNet(args, num_classes)
    elif args.model.lower() == 'mlp':
        _model = MLP(args, num_classes)
    elif args.model.lower() == 'mlplus':
        _model = MLPlus(args, num_classes)
    elif args.model.lower() == 'rgnn':
        edge_index, edge_weight = get_edge_weight()
        _model = RGNN(device, 62, edge_weight, edge_index, 5, 200, num_classes, 2)
    elif args.model.lower() == 'lstm':
        _model = LSTM(args, num_classes)
    else:
        raise ValueError(f"Couldn't find the model {args.model}")
    return _model


def plot_accuracies(train_accs, val_accs, max_val_acc_epoch, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accs) + 1)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.axvline(x=max_val_acc_epoch + 1, color='black', linestyle='--', label=f'Best Val Acc ({max(val_accs):.2%})')
    plt.title(f'{args.model.upper()}: Training vs. Validation Accuracy (Subject {args.subject}, {args.granularity})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def model_main(args, model, train_loader, test_loader, criterion, optimizer, num_epochs, patience, device, labels):
    model = model.to(device)
    unique_labels = torch.from_numpy(labels).unique()
    label_mapping = {original_label.item(): new_label for new_label, original_label in enumerate(unique_labels)}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    running_loss = 0.0
    max_acc = 0.0
    max_acc_epoch = -1
    report_batch = len(train_loader) / 2

    train_accuracies = []
    val_accuracies = []
    
    # Early stopping parameters
    epochs_without_improvement = 0
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            labels = torch.tensor([label_mapping[label.item()] for label in labels])
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, dim=1)
            train_total += len(labels)
            train_correct += accuracy_score(labels.cpu(), predicted.cpu(), normalize=False)
            
            if batch_idx % report_batch == report_batch - 1:
                print(f"[epoch {epoch}, batch {batch_idx}] loss: {running_loss / report_batch}")
                running_loss = 0.0
        
        train_acc = train_correct / train_total
        train_accuracies.append(train_acc)
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0
            for (inputs, labels) in test_loader:
                labels = torch.tensor([label_mapping[label.item()] for label in labels])
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, dim=1)
                total += len(labels)
                correct += accuracy_score(labels.cpu(), predicted.cpu(), normalize=False)
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > max_acc:
            max_acc = val_acc
            max_acc_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model}_s{args.subject}_1x_22.pth'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping...")
                break
    
    if args.plot:
        plot_path = os.path.join(args.output_dir, f'{args.model}_s{args.subject}_accuracy_plot.png')
        plot_accuracies(train_accuracies, val_accuracies, max_acc_epoch, plot_path)
    
    return max_acc, max_acc_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    parser.add_argument("--use_cv", action="store_true", help="use cross-validation for simple models (default: False)")
    parser.add_argument("--plot", action="store_true", help="save a plot of training and validation accuracies (default: False)")
    args = parser.parse_args()
    print(args)
    print('Loading dataset...')
    dataset = EEGImageNetDataset(args)
    eeg_data = np.stack([i[0].numpy() for i in dataset], axis=0)
    print('EEG data loaded with shape:', eeg_data.shape)

    # Extract frequency domain features
    de_feat = de_feat_cal(eeg_data, args)
    print('Differential entropy features calculated, shape:', de_feat.shape)
    dataset.add_frequency_feat(de_feat)
    print('Frequency features added')

    # Extract temporal domain features
    # de_temp = de_feat_temp(eeg_data, args)
    # print('Temporal differential entropy features calculated, shape:', de_temp.shape)
    # dataset.add_temporal_feat(de_temp)
    # print('Temporal features added')

    labels = np.array([i[1] for i in dataset])
    print('Labels shape:', labels.shape)

    # 60-40 train-test split (first 30 images=train, last 20 images=test)
    train_index = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_index = np.array([i for i in range(len(dataset)) if i % 50 > 29])
    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)

    simple_model_list = ['svm', 'rf', 'knn', 'dt', 'ridge']
    if_simple = args.model.lower() in simple_model_list
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_init(args, if_simple, len(dataset) // 50, device)
    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, str(args.pretrained_model))))
    if if_simple:
        cv_mode = " with cross-validation" if args.use_cv else ""
        print(f'Training {args.model} model{cv_mode}...')
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        train_feat = de_feat[train_index]
        test_feat = de_feat[test_index]
        model.fit(train_feat, train_labels)
        y_pred = model.predict(test_feat)
        acc = accuracy_score(test_labels, y_pred)
        print('Test accuracy:', acc)
        with open(os.path.join(args.output_dir, "results.txt"), "a") as f:
            f.write(f"{args.model.upper()} Test Accuracy: {acc} (subject={args.subject}, granularity={args.granularity})")
            f.write("\n")
    else:
        if args.model.lower() == 'eegnet':
            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9)
            acc, epoch = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, 200, device,
                                    labels)
        elif args.model.lower() == 'mlp':
            dataset.use_frequency_feat = True
            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
            acc, epoch = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, 200, device,
                                    labels)
        elif args.model.lower() == 'mlplus':
            dataset.use_frequency_feat = True
            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
            acc, epoch = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, 200, device,
                                    labels)
        elif args.model.lower() == 'rgnn':
            dataset.use_frequency_feat = True
            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            acc, epoch = model_main(args, model, train_dataloader, test_dataloader, criterion, optimizer, 1000, 200, device,
                                    labels)
        with open(os.path.join(args.output_dir, "results.txt"), "a") as f:
            f.write(f"{args.model.upper()} Test Accuracy: {acc} (subject={args.subject}, granularity={args.granularity})")
            f.write("\n")
