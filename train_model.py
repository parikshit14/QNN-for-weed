import sys
import torch
import torchvision.models as models
from torch.quantization import convert
import time
import numpy as np
from torchvision import models, transforms
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
from prepare_data.preprocess_dataset import get_dataloaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 9


def evaluate_model(model, data_loader):
    model.eval()
    corrects = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            corrects += torch.sum(preds == labels.data).detach().cpu().numpy()
            total_samples += labels.size(0)
            total_loss += loss.item() * inputs.size(0)

    # accuracy = corrects.double() / total_samples
    accuracy = float(corrects) / total_samples
    average_loss = total_loss / total_samples

    return average_loss, accuracy


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    since = time.time()
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data).detach().cpu().numpy()
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = float(corrects) / total_samples

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        )
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    # Plot the training and validation curves
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    #
    # plt.plot(np.arange(1, num_epochs + 1), train_losses, label='Train')
    # plt.plot(np.arange(1, num_epochs + 1), val_losses, label='Validation')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(np.arange(1, num_epochs + 1), train_accuracies, label='Train')
    # plt.plot(np.arange(1, num_epochs + 1), val_accuracies, label='Validation')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    print("train_losses: ", train_losses)
    print("val_losses: ", val_losses)
    print("train_accuracies: ", train_accuracies)
    print("val_accuracies: ", val_accuracies)

    # return train_losses, val_losses, train_accuracies, val_accuracies


def prepare_resnet50():

    pretrained_resnet50 = models.resnet50(pretrained=True)

    # Replace the last fully connected (FC) layer with your custom FC layer
    num_ftrs = pretrained_resnet50.fc.in_features
    # num_classes = 10  # Replace with the number of classes in your task
    custom_fc_layer = nn.Linear(num_ftrs, num_classes)
    pretrained_resnet50.fc = custom_fc_layer

    print(
        "Total parameters: ", sum(p.numel() for p in pretrained_resnet50.parameters())
    )
    print(
        "Trainable parameters: ",
        sum(p.numel() for p in pretrained_resnet50.parameters() if p.requires_grad),
    )
    return pretrained_resnet50


def prepare_inceptionv3():
    pretrained_inception = models.inception_v3(pretrained=True)

    # Replace the last fully connected (FC) layer with your custom FC layer
    num_ftrs = pretrained_inception.fc.in_features
    custom_fc_layer = nn.Linear(num_ftrs, num_classes)
    pretrained_inception.fc = custom_fc_layer
    pretrained_inception.aux_logits = False
    pretrained_inception.AuxLogits = None

    print(
        "Total parameters: ", sum(p.numel() for p in pretrained_inception.parameters())
    )
    print(
        "Trainable parameters: ",
        sum(p.numel() for p in pretrained_inception.parameters() if p.requires_grad),
    )
    return pretrained_inception


def prepare_quantized_resnet():
    def create_combined_model(model_fe):
        # Step 1. Isolate the feature extractor.
        model_fe_features = nn.Sequential(
            model_fe.quant,  # Quantize the input
            model_fe.conv1,
            model_fe.bn1,
            model_fe.relu,
            model_fe.maxpool,
            model_fe.layer1,
            model_fe.layer2,
            model_fe.layer3,
            model_fe.layer4,
            model_fe.avgpool,
            model_fe.dequant,  # Dequantize the output
        )

        # Step 2. Create a new "head"
        new_head = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
        )

        # Step 3. Combine, and don't forget the quant stubs.
        new_model = nn.Sequential(
            model_fe_features,
            nn.Flatten(1),
            new_head,
        )
        return new_model

    import torchvision.models.quantization as models

    model = models.resnet50(quantize=False)
    global num_ftrs
    num_ftrs = model.fc.in_features

    # Step 1
    model.train()
    model.fuse_model()

    model_ft = create_combined_model(model)

    model_ft[
        0
    ].qconfig = (
        torch.ao.quantization.default_qat_qconfig
    )  # Use default QAT configuration

    # Step 3
    model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)
    return model_ft


def prepare_quantized_inception():
    import torchvision.models.quantization as models

    inception_quantized = models.inception_v3(quantize=False)
    num_ftrs = inception_quantized.fc.in_features

    import torch
    from torch import nn

    def create_combined_model(model_fe):
        # Step 1. Isolate the feature extractor.
        model_fe_features = nn.Sequential(
            model_fe.quant,  # Quantize the input
            model_fe.Conv2d_1a_3x3,
            model_fe.Conv2d_2a_3x3,
            model_fe.Conv2d_2b_3x3,
            model_fe.maxpool1,
            model_fe.Conv2d_3b_1x1,
            model_fe.Conv2d_4a_3x3,
            model_fe.maxpool2,
            model_fe.Mixed_5b,
            model_fe.Mixed_5c,
            model_fe.Mixed_5d,
            model_fe.Mixed_6a,
            model_fe.Mixed_6b,
            model_fe.Mixed_6c,
            model_fe.Mixed_6d,
            model_fe.Mixed_6e,
            # model_fe.AuxLogits,
            model_fe.Mixed_7a,
            model_fe.Mixed_7b,
            model_fe.Mixed_7c,
            model_fe.avgpool,
            model_fe.dequant,  # Dequantize the output
        )

        # Step 2. Create a new "head"
        new_head = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
        )

        # Step 3. Combine, and don't forget the quant stubs.
        new_model = nn.Sequential(
            model_fe_features,
            nn.Flatten(1),
            new_head,
        )
        return new_model

    inception_quantized.train()
    inception_quantized.fuse_model()

    model_ft = create_combined_model(inception_quantized)

    model_ft[
        0
    ].qconfig = (
        torch.ao.quantization.default_qat_qconfig
    )  # Use default QAT configuration

    # Step 3
    model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)
    return model_ft


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Please model ID i.e \n 1) ResNet50 (fp32) \n 2) Inceptionv3 (fp32) \n 3) Quantized Resnet (int8) \n 4) Quantized Inception (int8)"
        )
        sys.exit(1)

    model_id = sys.argv[0]

    if model_id == 1:
        model = prepare_resnet50()
    elif model_id == 2:
        model = prepare_inceptionv3()
    elif model_id == 3:
        model = prepare_quantized_resnet()
    else:
        model = prepare_quantized_inception()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(model, train_loader, val_loader, criterion, optimizer, 30)

    if model_id > 1:
        device = "cpu"
        model.to(device)
        model_quantized_and_trained = convert(model, inplace=False)

    # To test on test Dataset
    _, acc = evaluate_model(model_quantized_and_trained, test_loader)
    print("Test data accuracy:", acc)
