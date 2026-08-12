"""
Fair baseline: identical node features, identical training data, identical
train/test split, identical MLP head architecture, the ONLY difference is
that this model mean-pools node features directly (ignoring the edge list
entirely) instead of running them through DAG-aware message passing first.

This isolates the actual scientific question for the GNN's contribution:
if the GNN beats this baseline on the HELD-OUT test set specifically, the
improvement is attributable to graph structure, not to having more
parameters, more training, or better features, since both models share
all of those.
"""
import torch
import torch.nn as nn


class FlatBaseline(nn.Module):
    """Same input dimension, same hidden width, same head, no message passing."""
    def __init__(self, in_dim=6, hidden_dim=32):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, node_features, edges=None):
        h = self.encode(node_features)
        graph_embedding = h.mean(dim=0)
        return self.head(graph_embedding)


def _standalone_run():
    import pickle
    import random
    import time

    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    torch.manual_seed(7)
    model = FlatBaseline(in_dim=6, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    BATCH_SIZE, N_EPOCHS = 16, 25

    def to_tensor(sample):
        x = torch.tensor(sample['node_features'], dtype=torch.float32)
        y = torch.tensor([sample['makespan'], sample['cost']], dtype=torch.float32)
        return x, y

    def evaluate(dataset, label):
        model.eval()
        total_loss, mae_m, mae_c = 0.0, 0.0, 0.0
        with torch.no_grad():
            for sample in dataset:
                x, y = to_tensor(sample)
                pred = model(x)
                total_loss += loss_fn(pred, y).item()
                mae_m += abs(pred[0].item() - y[0].item())
                mae_c += abs(pred[1].item() - y[1].item())
        n = len(dataset)
        print(f'  [{label}] MSE={total_loss/n:.4f}  MAE(makespan)={mae_m/n:.4f}  MAE(cost)={mae_c/n:.4f}')
        model.train()

    rng = random.Random(7)
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        order = list(range(len(train_data)))
        rng.shuffle(order)
        for batch_start in range(0, len(order), BATCH_SIZE):
            batch_idx = order[batch_start:batch_start + BATCH_SIZE]
            optimizer.zero_grad()
            batch_loss = 0.0
            for i in batch_idx:
                x, y = to_tensor(train_data[i])
                pred = model(x)
                batch_loss = batch_loss + loss_fn(pred, y)
            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{N_EPOCHS}  elapsed={time.time()-t0:.1f}s')
            evaluate(test_data, 'HELD-OUT TEST (structure-blind baseline)')

    print('\nFinal evaluation (structure-blind baseline):')
    evaluate(train_data, 'train (seen during training)')
    evaluate(test_data, 'held-out test (unseen workflows)')


if __name__ == '__main__':
    _standalone_run()
