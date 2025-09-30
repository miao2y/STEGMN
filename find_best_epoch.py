import json
import os

root_dir = "logs/md17_logs"
exp_dirs = os.listdir(root_dir)

for exp in exp_dirs:
    # read json
    print(os.path.join(root_dir, exp, "loss.json"))
    if not os.path.exists(os.path.join(root_dir, exp, "loss.json")):
        continue
    with open(os.path.join(root_dir, exp, "loss.json"), "r") as f:
        losses = json.load(f)
        """
        {
            "epochs": [...],
            "test loss": [...],
            "val loss": [...],
            "train loss": [...]
        }
        """
        min_val_loss = min(losses["val loss"])
        best_index = losses["val loss"].index(min_val_loss)
        best_epoch = losses["epochs"][losses["val loss"].index(min_val_loss)]
        train_loss = losses["train loss"][best_index]
        test_loss = losses["test loss"][best_index]
        print(f"===========Experiment: {exp}=========")
        print(f"Best epoch: {best_epoch}")
        print(f"Train loss: {train_loss:8f}")
        print(f"Test loss: {test_loss:8f}")
        print(f"Val loss: {min_val_loss:8f}")
