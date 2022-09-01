# Add offline training code for Track-2 here.
# On completion of training, automatically save the trained model to `track2/submission` directory.

import os
import argparse
import matplotlib.pyplot as plt
import sys
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

# To import submission folder
sys.path.insert(0, str(Path(__file__).parents[1]))

from submission.model_IL import MainNet
from utility_IL import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)


def create_datasets(dataset_path, save_path=None, cache=False, batch_size=32):
    class TensorDatasetTransforms(torch.utils.data.TensorDataset):
        def __init__(self, x, y):
            super().__init__(x, y)

        def __getitem__(self, index):
            tensor = data_transform(self.tensors[0][index])

            return (tensor,) + tuple(t[index] for t in self.tensors[1:])

    class CustomTensorDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __getitem__(self, index):
            img = data_transform(torch.tensor(self.x[index][0]))
            goal = torch.tensor(self.x[index][1])
            y = torch.tensor(self.y[index])

            return [img, goal], y

        def __len__(self):

            return self.x.shape[0]

    x, y = load_data(dataset_path, save_path, cache=cache)

    train_set = CustomTensorDataset(x, y)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader


def save_losses(
    train_total_losses,
    train_total_loss,
    train_dx_losses,
    train_dx_loss,
    train_dy_losses,
    train_dy_loss,
    test_total_losses,
    test_total_loss,
    test_dx_losses,
    test_dx_loss,
    test_dy_losses,
    test_dy_loss,
):

    train_total_losses.append(train_total_loss)
    train_dx_losses.append(train_dx_loss)
    train_dy_losses.append(train_dy_loss)
    test_total_losses.append(test_total_loss)
    test_dx_losses.append(test_dx_loss)
    test_dy_losses.append(test_dy_loss)

    return (
        train_total_losses,
        train_dx_losses,
        train_dy_losses,
        test_total_losses,
        test_dx_losses,
        test_dy_losses,
    )


def train(
    model,
    dataset_path,
    checkpoint_path,
    cache=False,
    lr=0.001,
    num_epochs=100,
    batch_size=32,
    save_steps=30,
):
    """
    Train method
    :param model: the network
    """
    model = model.to(device)

    train_total_losses, train_dx_losses, train_dy_losses = ([], [], [])
    test_total_losses, test_dx_losses, test_dy_losses = ([], [], [])
    epochs = []

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Read datasets
    train_loader = create_datasets(
        dataset_path, save_path=checkpoint_path, cache=cache, batch_size=batch_size
    )

    # Setup log dir
    log_dir = Path(__file__).absolute().parents[0] / "logs"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # Train
    save_step = 0

    for epoch in range(num_epochs):
        save_step += 1
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        epochs.append(epoch)
        train_total_loss, train_dx_loss, train_dy_loss = train_epoch(
            model, optimizer, train_loader
        )
        test_total_loss = 0
        test_dx_loss = 0
        test_dy_loss = 0

        (
            train_total_losses,
            train_dx_losses,
            train_dy_losses,
            test_total_losses,
            test_dx_losses,
            test_dy_losses,
        ) = save_losses(
            train_total_losses,
            train_total_loss,
            train_dx_losses,
            train_dx_loss,
            train_dy_losses,
            train_dy_loss,
            test_total_losses,
            test_total_loss,
            test_dx_losses,
            test_dx_loss,
            test_dy_losses,
            test_dy_loss,
        )
        plt.ylim(0, 1)
        plt.plot(epochs, train_total_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.savefig(log_dir / "losses.png")
        plt.clf()

        # save model
        if save_step == save_steps:
            save_step = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_history": train_total_loss,
                },
                os.path.join(checkpoint_path, "model_IL_{}.ckpt".format(epoch)),
            )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss_history": train_total_loss,
        },
        os.path.join(checkpoint_path, "model_IL.ckpt"),
    )


def train_epoch(model, optimizer, data_loader):
    """Train for a single epoch"""
    dx_losses = 0.0
    dy_losses = 0.0
    dh_losses = 0.0
    current_loss = 0.0

    model.train()

    for i, (inputs, labels) in enumerate(data_loader):
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            labels = labels.float()
            loss, dx_l, dy_l, dh_l = model.compute_loss(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        current_loss += loss.item() * inputs[0].size(0)
        dx_losses += dx_l.item() * inputs[0].size(0)
        dy_losses += dy_l.item() * inputs[0].size(0)
        dh_losses += dh_l.item() * inputs[0].size(0)

    total_loss = current_loss / len(data_loader.dataset)
    total_dx_loss = dx_losses / len(data_loader.dataset)
    total_dy_loss = dy_losses / len(data_loader.dataset)
    total_dh_loss = dh_losses / len(data_loader.dataset)

    print("Train Loss: {:.4f}".format(total_loss))
    print("Train dx loss: {:.4f}".format(total_dx_loss))
    print("Train dy loss: {:.4f}".format(total_dy_loss))
    print("Train dh loss: {:.4f}".format(total_dh_loss))

    return total_loss, total_dx_loss, total_dy_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/SMARTS/competition/offline_dataset",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output_path",
        default="/SMARTS/competition/track2/submission",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--cache",
        default=False,
        type=bool,
        required=False,
    )
    parser.add_argument(
        "--save_steps",
        default=30,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--num_epochs",
        default=2,
        type=int,
        required=False,
    )
    args = parser.parse_args()

    print("Training...")
    m = MainNet()
    train(
        m,
        args.dataset_path,
        args.output_path,
        cache=args.cache,
        lr=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_steps=args.save_steps,
    )
    print("Training Done.")


if __name__ == "__main__":
    main()
