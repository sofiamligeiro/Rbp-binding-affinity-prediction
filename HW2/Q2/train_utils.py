import copy
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import utils_w_masking as utils


def load_data(protein, config):
    """Loads RNAcompete datasets (train, validation, test) for a given protein
    Args:
        protein (str): Protein name
        config (dict): Configuration parameters for data loading
    Returns:
        tuple: (train_ds, val_ds, test_ds) Datasets for training, validation, and testing.
    """
    print(f"Loading datasets for {protein}...")
    train_ds = utils.load_rnacompete_data(protein, "train", config)
    val_ds = utils.load_rnacompete_data(protein, "val", config)
    test_ds = utils.load_rnacompete_data(protein, "test", config)
    return train_ds, val_ds, test_ds


def train_batch(batch, model, optimizer, device, **kwargs):
    """Performs a single training step on a batch
    Args:
        batch: (x, y, mask) tensors for input, target, and mask
        model (torch.nn.Module): a PyTorch defined model
        optimizer (torch.optim.Optimizer): optimizer used in gradient step
        device (torch.device): device to run computations on
    Returns:
        float: loss value for this batch
    """
    model.train()

    x, seq_mask, y, mask = batch
    x = x.to(device)
    seq_mask = seq_mask.to(device)
    y = y.to(device)
    mask = mask.to(device)

    optimizer.zero_grad()
    preds = model(x, seq_mask=seq_mask)  

    loss = utils.masked_mse_loss(preds, y, mask)
    loss.backward()
    optimizer.step()

    return loss.item()



@torch.no_grad()
def predict(model, X, device):
    """Computes model predictions for input X
    Args:
        model (torch.nn.Module): a PyTorch defined model
        X (torch.Tensor): input tensor of shape (batch_size, seq_len, 4)
        device (torch.device): device to run computations on
    Returns:
        torch.Tensor: predictions 
    """
    model.eval()
    X = X.to(device)
    preds = model(X)
    return preds



@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluates model performance on a dataset
       Computes masked MSE loss and masked Spearman correlation
    Args:
        model (torch.nn.Module): a PyTorch defined model
        dataloader (DataLoader): DataLoader yielding (x, y, mask)
        device (torch.device): device to run computations on
    Returns:
        tuple: (total_loss, spearman) 
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_masks = [], [], []

    for batch in dataloader:
        x, seq_mask, y, mask = batch
        x = x.to(device)
        seq_mask = seq_mask.to(device)
        y = y.to(device)
        mask = mask.to(device)

        preds = model(x, seq_mask=seq_mask)

        loss = utils.masked_mse_loss(preds, y, mask)
        total_loss += loss.item() * x.size(0)

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        all_masks.append(mask.cpu())

    total_loss /= len(dataloader.dataset)

    spearman = utils.masked_spearman_correlation(
        torch.cat(all_preds),
        torch.cat(all_targets),
        torch.cat(all_masks)
    ).item()

    return total_loss, spearman


def train_model(
        model, 
        train_loader, 
        val_loader, 
        lr, 
        num_epochs, 
        device,
        optimizer_class=torch.optim.Adam, 
        optimizer_kwargs=None
):
    """Trains a PyTorch model and tracks metrics
    Args:
        model (torch.nn.Module): a PyTorch defined model
        train_loader (DataLoader): DataLoader for training set
        val_loader (DataLoader): DataLoader for validation set
        lr (float): learning rate
        num_epochs (int): number of epochs
        device (torch.device): device to run computations on
        optimizer_class (class, optional): optimizer class (defaults to Adam)
        optimizer_kwargs (dict, optional): additional optimizer args
    Returns:
        dict: Training results containing:
            - best_state_dict (dict): model parameters with best validation Spearman
            - best_val_spearman (float)
            - best_epoch (int)
            - train_losses, val_losses (list of float)
            - train_spearmans, val_spearmans (list of float)
    """
    
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_kwargs)

    best_val_spearman = -1.0
    best_state_dict = None
    best_epoch = 0

    train_losses, val_losses = [], []
    train_spearmans, val_spearmans = [], []

    for ep in range(1, num_epochs + 1):
        epoch_train_loss = 0.0

        for batch in train_loader:
            loss = train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                device=device
            )
            epoch_train_loss += loss * batch[0].size(0)

        epoch_train_loss /= len(train_loader.dataset)

        train_loss_eval, train_spearman = evaluate(model, train_loader, device)
        val_loss, val_spearman = evaluate(model, val_loader, device)

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        train_spearmans.append(train_spearman)
        val_spearmans.append(val_spearman)

        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = ep

        print(
            f"Epoch {ep:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Spearman: {train_spearman:.4f} | "
            f"Val Spearman: {val_spearman:.4f}"
        )

    return {
        "best_state_dict": best_state_dict,
        "best_val_spearman": best_val_spearman,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_spearmans": train_spearmans,
        "val_spearmans": val_spearmans
    }


def run_grid_search(
        train_ds, 
        val_ds, 
        model_class, 
        device,
        hyperparam_combinations, 
        model_kwargs_func,
        optimizer_class=torch.optim.Adam, 
        optimizer_kwargs=None
):
    """Performs grid search over hyperparameters for a given model class
    Args:
        train_ds (Dataset): training dataset
        val_ds (Dataset): validation dataset
        model_class (nn.Module): model class
        device (torch.device): device to run computations on
        hyperparam_combinations (list[tuple]): hyperparameter combinations
        model_kwargs_func (function): converts hyperparam tuple to model kwargs
        optimizer_class (class, optional): optimizer class (defaults to Adam)
        optimizer_kwargs (dict, optional): additional optimizer args
    Returns:
        list[dict]: results for each hyperparameter combination
    """
    results = []

    for hparams in hyperparam_combinations:
        print("\nTraining with hyperparams:", hparams)

        batch_size = hparams[1]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model_kwargs = model_kwargs_func(hparams)
        model = model_class(**model_kwargs).to(device)

        lr = hparams[0]
        num_epochs = hparams[-1]

        res = train_model(
            model, 
            train_loader, 
            val_loader, 
            lr, 
            num_epochs, 
            device,
            optimizer_class=optimizer_class, 
            optimizer_kwargs=optimizer_kwargs
        )

        res["hyperparams"] = model_kwargs.copy()
        res["hyperparams"]["lr"] = lr
        res["hyperparams"]["num_epochs"] = num_epochs
        results.append(res)

    return results


def save_results(results, out_dir, tag="model"):
    """Saves grid search or training results including best model and plots
    Args:
        results (list[dict]): output from `run_grid_search`
        out_dir (str): directory to save results
        tag (str, optional): tag for filename
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame([{
        **r["hyperparams"],
        "best_val_spearman": r["best_val_spearman"],
        "best_epoch": r["best_epoch"]
    } for r in results])

    csv_path = os.path.join(out_dir, f"{tag}_gridsearch_results.csv")
    df.to_csv(csv_path, index=False)

    best_idx = df["best_val_spearman"].idxmax()
    best_res = results[best_idx]

    model_path = os.path.join(out_dir, f"{tag}_best_model.pt")
    torch.save(best_res["best_state_dict"], model_path)

    epochs = range(1, best_res["hyperparams"]["num_epochs"] + 1)
    utils.plot(
        epochs,
        {
            "Train Loss": best_res["train_losses"],
            "Val Loss": best_res["val_losses"]
        },
        filename=os.path.join(out_dir, f"{tag}_loss_curves.png")
    )
    utils.plot(
        epochs,
        {
            "Train Spearman": best_res["train_spearmans"],
            "Val Spearman": best_res["val_spearmans"]
        },
        filename=os.path.join(out_dir, f"{tag}_spearman_curves.png")
    )

    print("\nBest hyperparameters:", best_res["hyperparams"])
    print("Best Val Spearman:", best_res["best_val_spearman"])
    print("Results saved in:", out_dir)


def test_model(model_class, best_state_dict, model_kwargs, test_ds, batch_size, device):
    """Evaluates a trained PyTorch model on a test dataset"""
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(best_state_dict)

    test_loss, test_spearman = evaluate(model, test_loader, device)

    return {"test_loss": test_loss, "test_spearman": test_spearman}
