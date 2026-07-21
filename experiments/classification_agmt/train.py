import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))
import diffusion_net
from agmt_dataset import BIMElementDataset


DEFAULT_CONFIG_PATH = "/opt/ml/input/config/hyperparameters.json"
DEFAULT_HYPERPARAMS = {
    "input_features": "hks",
    "max_epoch": 200,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "label_smoothing": 0.0,
    "k_eig": 128,
    "width": 64,
    "blocks": 4,
    "dropout": True,
    "num_workers": 0,
    "seed": 42,
    "scheduler_gamma": 0.98,
}


def add_subparser(subparsers):
    train_parser = subparsers.add_parser("train", help="train the classifier")
    train_parser.set_defaults(handler=run)
    train_parser.add_argument("--data_dir", required=True)
    train_parser.add_argument("--output_dir", default="runs/agmt_diffusionnet")
    train_parser.add_argument("--target_categories", default=None)
    train_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="path to a JSON hyperparameter file",
    )
    train_parser.add_argument("--checkpoint", default=None)
    train_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return train_parser


def _to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "t")
    return bool(value)


def load_hyperparams(path, defaults=DEFAULT_HYPERPARAMS):
    hyperparams = dict(defaults)
    if not path:
        return hyperparams
    try:
        with open(path) as config_file:
            config = json.load(config_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return hyperparams

    for key, default in defaults.items():
        if key in config:
            try:
                if isinstance(default, bool):
                    hyperparams[key] = _to_bool(config[key])
                else:
                    hyperparams[key] = type(default)(config[key])
            except (TypeError, ValueError):
                hyperparams[key] = default
    return hyperparams


def validate_hyperparams(hyperparams):
    if hyperparams["input_features"] not in ("xyz", "hks"):
        raise ValueError("input_features must be 'xyz' or 'hks'")
    if hyperparams["max_epoch"] < 1:
        raise ValueError("max_epoch must be at least 1")
    if hyperparams["learning_rate"] <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if hyperparams["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= hyperparams["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if hyperparams["k_eig"] < 1:
        raise ValueError("k_eig must be at least 1")
    if hyperparams["width"] < 1:
        raise ValueError("width must be at least 1")
    if hyperparams["blocks"] < 1:
        raise ValueError("blocks must be at least 1")
    if hyperparams["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative")
    if hyperparams["scheduler_gamma"] <= 0:
        raise ValueError("scheduler_gamma must be greater than 0")


def load_target_categories(path):
    if path is None:
        return None
    with open(path) as config_file:
        config = json.load(config_file)
    targets = config.get("targets") if isinstance(config, dict) else config
    if not isinstance(targets, list) or not targets:
        raise ValueError("Target categories must be a JSON list or an object with a 'targets' list")
    return set(targets)


def move_data(data, device):
    return tuple(value.to(device) for value in data)


def make_features(input_features, verts, evals, evecs):
    if input_features == "xyz":
        return verts
    return diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)


def run_epoch(model, loader, device, input_features, optimizer=None, label_smoothing=0.0):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0

    with torch.set_grad_enabled(is_training):
        for data in tqdm(loader, leave=False):
            verts, faces, frames, mass, laplacian, evals, evecs, grad_x, grad_y, labels = move_data(data, device)
            if is_training and input_features == "xyz":
                verts = diffusion_net.utils.random_rotate_points(verts)
            features = make_features(input_features, verts, evals, evecs)
            predictions = model(
                features,
                mass,
                L=laplacian,
                evals=evals,
                evecs=evecs,
                gradX=grad_x,
                gradY=grad_y,
                faces=faces,
            )
            loss = diffusion_net.utils.label_smoothing_log_loss(
                predictions, labels, label_smoothing
            )

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += predictions.argmax(dim=-1).eq(labels).sum().item()

    return total_loss / len(loader), correct / len(loader)


def run(args):
    hyperparams = load_hyperparams(args.config)
    validate_hyperparams(hyperparams)
    print("Configuration:", hyperparams)

    torch.manual_seed(hyperparams["seed"])
    device = torch.device(args.device)
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    split_counts = {
        "train": BIMElementDataset.collect_category_counts(train_dir),
        "test": BIMElementDataset.collect_category_counts(test_dir),
    }
    categories = set(split_counts["train"]) | set(split_counts["test"])
    targets = load_target_categories(args.target_categories)
    if targets is not None:
        categories &= targets
    class_names = sorted(categories)
    if not class_names:
        raise ValueError("No selected categories were found in the dataset")
    if len(class_names) < 2:
        raise ValueError("Training requires at least two selected categories")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "classes.json"), "w") as classes_file:
        json.dump(class_names, classes_file, indent=2)
    cache_dir = os.path.join(args.output_dir, "op_cache")
    train_dataset = BIMElementDataset(
        train_dir,
        class_to_idx,
        k_eig=hyperparams["k_eig"],
        op_cache_dir=cache_dir,
    )
    test_dataset = BIMElementDataset(
        test_dir,
        class_to_idx,
        k_eig=hyperparams["k_eig"],
        op_cache_dir=cache_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=hyperparams["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=hyperparams["num_workers"],
    )
    print("Classes ({}): {}".format(len(class_names), ", ".join(class_names)))
    print("Elements: {} train, {} test".format(len(train_dataset), len(test_dataset)))

    model = diffusion_net.layers.DiffusionNet(
        C_in=3 if hyperparams["input_features"] == "xyz" else 16,
        C_out=len(class_names),
        C_width=hyperparams["width"],
        N_block=hyperparams["blocks"],
        last_activation=lambda values: torch.nn.functional.log_softmax(values, dim=-1),
        outputs_at="global_mean",
        dropout=hyperparams["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=hyperparams["scheduler_gamma"]
    )
    start_epoch = 0
    best_accuracy = 0.0

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if checkpoint["class_names"] != class_names:
            raise ValueError("Checkpoint categories do not match the current dataset")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint.get("best_accuracy", 0.0)

    for epoch in range(start_epoch, hyperparams["max_epoch"]):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            device,
            hyperparams["input_features"],
            optimizer=optimizer,
            label_smoothing=hyperparams["label_smoothing"],
        )
        test_loss, test_accuracy = run_epoch(
            model, test_loader, device, hyperparams["input_features"]
        )
        scheduler.step()
        print(
            "Epoch {:03d} | train loss {:.4f} acc {:.2%} | test loss {:.4f} acc {:.2%}".format(
                epoch, train_loss, train_accuracy, test_loss, test_accuracy
            )
        )

        state = {
            "epoch": epoch,
            "class_names": class_names,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_accuracy": max(best_accuracy, test_accuracy),
            "args": {
                key: value for key, value in vars(args).items() if key != "handler"
            },
            "hyperparameters": hyperparams,
        }
        torch.save(state, os.path.join(args.output_dir, "last.pt"))
        if test_accuracy >= best_accuracy:
            best_accuracy = test_accuracy
            torch.save(state, os.path.join(args.output_dir, "best.pt"))