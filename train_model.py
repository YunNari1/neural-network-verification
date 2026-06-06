"""
Trains a small fully-connected neural network on MNIST and exports it to ONNX.

Architecture: 784 -> 64 -> 32 -> 10  (ReLU activations, no softmax)
The model is intentionally small so that alpha-beta-CROWN can verify it
in reasonable time with complete verification.

Usage:
    python train_model.py [--epochs N] [--output PATH]
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
import numpy as np


class MnistMLP(nn.Module):
    """Small MLP for MNIST: 784 -> 64 -> 32 -> 10, ReLU activations.

    Accepts both (batch, 784) flat inputs and (batch, 1, 28, 28) image inputs.
    The Flatten is included in the ONNX graph so alpha-beta-CROWN's MNIST
    data loader (which returns 28x28 images) works without reshaping.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def get_loaders(batch_size: int = 128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, 784).to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels.to(device)).sum().item()
    return 100.0 * correct / len(loader.dataset)


def export_onnx(model: nn.Module, path: str):
    model.eval()
    # Use (1, 1, 28, 28) so alpha-beta-CROWN's MNIST loader (returns images) works directly.
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"  Exported to {path}")


def verify_onnx(model: nn.Module, path: str, device):
    model.eval()
    sample = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        pt_out = model(sample.to(device)).cpu().numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": sample.numpy()})[0]
    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"  ONNX sanity check - max diff (PyTorch vs ORT): {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX export mismatch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", type=str, default="models/mnist_mlp.onnx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders()
    model = MnistMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining MnistMLP (784->64->32->10)...")
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch}/{args.epochs}  loss={loss:.4f}  train_acc={train_acc:.2f}%")

    test_acc = evaluate(model, test_loader, device)
    print(f"\nTest accuracy: {test_acc:.2f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print("\nExporting to ONNX...")
    export_onnx(model, args.output)
    verify_onnx(model, args.output, device)

    pt_path = args.output.replace(".onnx", ".pth")
    torch.save(model.state_dict(), pt_path)
    print(f"  PyTorch weights saved to {pt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
