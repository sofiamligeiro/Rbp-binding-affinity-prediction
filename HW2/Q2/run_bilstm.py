import itertools
import torch
import train_utils
import utils_w_masking as utils
from models import BiLSTM
from config import RNAConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

utils.configure_seed(42)

OUT_DIR = "bilstm_results"
protein = "RBFOX1"
config = RNAConfig()


LRs = [0.001, 0.0001, 0.0005]
BATCH_SIZES = [64]
HIDDEN_SIZES = [64, 128]
NUM_LAYERS = [2, 4]
DROPOUTS = [0.2, 0.4]
EPOCHS = [30]

hyperparam_combinations = list(itertools.product(
    LRs, BATCH_SIZES, HIDDEN_SIZES, NUM_LAYERS, DROPOUTS, EPOCHS
))


def bilstm_kwargs(hparams):
    lr, batch_size, hidden_size, num_layers, dropout, num_epochs = hparams
    return {
        "input_size": 4,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout
    }

def main():
    train_ds, val_ds, test_ds = train_utils.load_data(protein, config)


    results = train_utils.run_grid_search(
        train_ds=train_ds,
        val_ds=val_ds,
        model_class=BiLSTM,
        device=DEVICE,
        hyperparam_combinations=hyperparam_combinations,
        model_kwargs_func=bilstm_kwargs,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 1e-4}
    )

    train_utils.save_results(results, OUT_DIR, tag="bilstm")
    best_res = max(results, key=lambda x: x["best_val_spearman"])
    model_kwargs = {
        "input_size": best_res["hyperparams"]["input_size"],
        "hidden_size": best_res["hyperparams"]["hidden_size"],
        "num_layers": best_res["hyperparams"]["num_layers"],
        "dropout": best_res["hyperparams"]["dropout"],
    }

    test_results = train_utils.test_model(
        model_class=BiLSTM,
        best_state_dict=best_res["best_state_dict"],
        model_kwargs=model_kwargs,
        test_ds=test_ds,
        batch_size=64,
        device=DEVICE
    )
    print("\n===== FINAL TEST RESULTS (BiLSTM) =====")
    print(f"Test MSE: {test_results['test_loss']:.4f}")
    print(f"Test Spearman: {test_results['test_spearman']:.4f}")


if __name__ == "__main__":
    main()
