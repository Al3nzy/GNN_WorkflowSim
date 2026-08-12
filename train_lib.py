import torch
import torch.nn as nn
import random


def to_tensor_gnn(sample):
    x = torch.tensor(sample['node_features'], dtype=torch.float32)
    y = torch.tensor([sample['makespan'], sample['cost']], dtype=torch.float32)
    return x, sample['edges'], y


def evaluate_gnn(model, dataset, loss_fn):
    model.eval()
    total_loss, mae_m, mae_c = 0.0, 0.0, 0.0
    with torch.no_grad():
        for sample in dataset:
            x, edges, y = to_tensor_gnn(sample)
            pred = model(x, edges)
            total_loss += loss_fn(pred, y).item()
            mae_m += abs(pred[0].item() - y[0].item())
            mae_c += abs(pred[1].item() - y[1].item())
    n = len(dataset)
    model.train()
    return total_loss / n, mae_m / n, mae_c / n


def train_gnn_one_seed(model_cls, train_data, test_data, seed, n_epochs=40,
                        batch_size=16, lr=1e-3, lr_decay_every=15, lr_decay_factor=0.5,
                        verbose=False):
    torch.manual_seed(seed)
    model = model_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay_factor)
    loss_fn = nn.MSELoss()
    rng = random.Random(seed)

    for epoch in range(n_epochs):
        order = list(range(len(train_data)))
        rng.shuffle(order)
        for batch_start in range(0, len(order), batch_size):
            batch_idx = order[batch_start:batch_start + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            for i in batch_idx:
                x, edges, y = to_tensor_gnn(train_data[i])
                pred = model(x, edges)
                batch_loss = batch_loss + loss_fn(pred, y)
            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            optimizer.step()
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            test_mse, test_mae_m, test_mae_c = evaluate_gnn(model, test_data, loss_fn)
            print(f'    seed={seed} epoch={epoch+1}/{n_epochs} test_MSE={test_mse:.4f}')

    train_mse, train_mae_m, train_mae_c = evaluate_gnn(model, train_data, loss_fn)
    test_mse, test_mae_m, test_mae_c = evaluate_gnn(model, test_data, loss_fn)
    return {
        'seed': seed, 'model': model,
        'train_mse': train_mse, 'train_mae_makespan': train_mae_m, 'train_mae_cost': train_mae_c,
        'test_mse': test_mse, 'test_mae_makespan': test_mae_m, 'test_mae_cost': test_mae_c,
    }
