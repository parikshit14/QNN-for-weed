import time
import torch
import numpy as np
from torchvision import models, transforms
from torch import nn
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from prepare_data.preprocess_dataset import get_dataloaders
from torch.utils.mobile_optimizer import optimize_for_mobile


def quantized_resnet_modeling():
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
    # model_ft[0].qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')

    # Step 3
    model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)

    from torch.quantization import convert

    model_ft.cpu()

    model_loaded = convert(model_ft)

    model_loaded.load_state_dict(torch.load("models/QuantizedTrainedResNet.pth"))
    return model_loaded


def quantized_inception_modeling():
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

    from torch.quantization import convert

    model_ft.cpu()

    model_to_load = convert(model_ft, inplace=False)
    model_to_load.load_state_dict(torch.load("models/QuantizedTrainedInception.pth"))
    # model_to_load.load_state_dict(torch.load('/content/drive/MyDrive/QWID Trained Models/QuantizedTrainedInceptionQNNpack.pth'))
    return model_to_load


def non_quantized_modeling(model_type):
    if model_type == 0:
        model = torch.load("models/TrainedResNet.pth", map_location="cpu")
    else:
        model = torch.load(
            "models/TrainedInception.pth", map_location=torch.device("cpu")
        )
        model.aux_logits = False
        model.AuxLogits = None
    return model


def top_k_inference(model):
    top1_correct = 0
    top3_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            _, top3_predicted = torch.topk(outputs, 3, dim=1)
            total_examples += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            top3_correct += torch.sum(top3_predicted == labels.view(-1, 1)).item()

    top1_accuracy = (top1_correct / total_examples) * 100
    top3_accuracy = (top3_correct / total_examples) * 100

    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")


def dummy_inference(model):
    model.eval()
    input_tensor = torch.randn(
        1, 3, 224, 224
    )  # Batch size 1, 3 channels, 224x224 image

    # Warm-up inference (optional, to ensure that CUDA is initialized, etc.)
    with torch.no_grad():
        _ = model(input_tensor)

    # Number of iterations you want to measure (adjust as needed)
    num_iterations = 100

    # Measure inference time for num_iterations iterations
    total_time = 0.0
    for _ in range(num_iterations):
        start_time = time.time()

        with torch.no_grad():
            _ = model(input_tensor)  # Perform inference

        end_time = time.time()
        iteration_time = end_time - start_time
        total_time += iteration_time

    # Calculate microseconds per iteration
    microseconds_per_iteration = (total_time / num_iterations) * 1e6
    print(microseconds_per_iteration)


def save_for_RPI_inference(model):
    net = torch.jit.script(model)
    torchscript_model_optimized = optimize_for_mobile(net)
    torchscript_model_optimized._save_for_lite_interpreter("model.pt")


num_classes = 9
device = "cpu"
criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":

    print("#####Quantized Resnet#####")
    quant_resnet = quantized_resnet_modeling()
    top_k_inference(quant_resnet)
    dummy_inference(quant_resnet)

    print("#####Resnet#####")
    resnet = non_quantized_modeling(0)
    top_k_inference(resnet)
    dummy_inference(resnet)

    print("#####Quantized Inception#####")
    quant_inception = quantized_inception_modeling()
    top_k_inference(quant_inception)
    dummy_inference(quant_inception)

    print("#####Inception#####")
    inception = non_quantized_modeling(1)
    top_k_inference(inception)
    dummy_inference(inception)
