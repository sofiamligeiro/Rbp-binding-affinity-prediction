import itertools
import torch
from models import CNN
import train_utils
from config import RNAConfig
import utils_w_masking as utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

utils.configure_seed(42)

OUT_DIR = "cnn_results"
protein = "RBFOX1"
config = RNAConfig()


LRs = [0.001, 0.0001, 0.0005]
BATCH_SIZES = [64]
DROPOUTS = [0.2, 0.4]
BASE_FILTERS = [16, 32]
EPOCHS = [30]

hyperparam_combinations = list(itertools.product(
    LRs, BATCH_SIZES, DROPOUTS, BASE_FILTERS, EPOCHS
))


def cnn_kwargs(hparams):
    lr, batch_size, dropout, base_filters,num_epochs = hparams
    return {
        "input_channels": 4,
        "base_filters": base_filters,
        "out_dim": 1,
        "dropout_prob": dropout
    }


def main():
    train_ds, val_ds, test_ds = train_utils.load_data(protein, config)

    results = train_utils.run_grid_search(
        train_ds=train_ds,
        val_ds=val_ds,
        model_class=CNN,
        device=DEVICE,
        hyperparam_combinations=hyperparam_combinations,
        model_kwargs_func=cnn_kwargs,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 1e-5}
    )


    train_utils.save_results(results, OUT_DIR, tag="cnn")
    best_res = max(results, key=lambda x: x["best_val_spearman"])

    model_kwargs = {
        "input_channels": best_res["hyperparams"]["input_channels"],
        "base_filters": best_res["hyperparams"]["base_filters"],
        "out_dim": best_res["hyperparams"]["out_dim"],
        "dropout_prob": best_res["hyperparams"]["dropout_prob"],
    }

    test_results = train_utils.test_model(
        model_class=CNN,
        best_state_dict=best_res["best_state_dict"],
        model_kwargs=model_kwargs,
        test_ds=test_ds,
        batch_size=64,
        device=DEVICE
    )

    print("\n===== FINAL TEST RESULTS (CNN) =====")
    print(f"Test MSE: {test_results['test_loss']:.4f}")
    print(f"Test Spearman: {test_results['test_spearman']:.4f}")

if __name__ == "__main__":
    main()
