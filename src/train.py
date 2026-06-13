import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from utils import load_data


class RecommenderNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layer(x)


def main():
    df = load_data("data/features/user_features.csv")

    X = torch.tensor(df[["view", "addtocart"]].values, dtype=torch.float32)
    y = torch.tensor(df["transaction"].values, dtype=torch.float32).view(-1, 1)

    model = RecommenderNet(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("=============================================================")
    print("Iniciando treinamento do modelo PyTorch...")

    epochs = 50

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{epochs}] | Loss (Perda): {loss.item():.4f}")

    print(f"Treinamento concluído. Loss final: {loss.item():.4f}")

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / "recommender_model.pth")
    print(f'Modelo salvo em: {output_dir / "recommender_model.pth"}')


if __name__ == "__main__":
    main()
