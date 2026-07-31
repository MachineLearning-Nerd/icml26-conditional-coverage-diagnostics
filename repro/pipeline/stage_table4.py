"""Claim 5 — the classification over/under decomposition of Table 4.

Paper (Table 4, Section 4.3.2): the classification experiments report KL+-ERT
and KL--ERT values that diverge across the two conformal strategies, and the
paper reads that divergence as the mechanism behind a sign disagreement between
L1-ERT and KL-ERT on CIFAR100 — the likelihood strategy produces more empty
prediction sets, whose conditional coverage is zero, which KL weights far more
heavily than L1 and which lands on the under-coverage side.

Everything here is transcribed from the release's own per-dataset drivers
(`_generate_MNIST_experiments.py`, `_generate_FashionMNIST_experiments.py`,
`_generate_CIFAR10_experiments.py`), which already run on CPU:

    dataset        train  calibration  test    epochs   ERT classifier
    MNIST          5%     20%          75%     1        BinaryImageClassifier(1, 28)
    FashionMNIST   15%    20%          65%     5        BinaryImageClassifier(1, 28)
    CIFAR10        40%    20%          40%     10       BinaryImageClassifier(3, 32)

with alpha = 0.1, Adam at lr 1e-3, batch 128 for f-hat and batch 32 for the ERT
classifier over 10 epochs, and five-fold ERT.  The two strategies are the
release's `ClassificationNegativeLikelihood` (S = -p(X)_Y) and
`ClassificationCumulativeLikelihood` (S = sum of probabilities at least as
large as the true class's).

The release's `ERT_underconfident_*` fields clip predictions from below at
1-alpha and `ERT_overconfident_*` clip from above, which are covmetrics'
`*_over` and `*_under` losses respectively.  Table 4's KL+-ERT is therefore
ERT_logloss_over and KL--ERT is ERT_logloss_under; the mapping is asserted in
the verifier rather than assumed here.

Empty-set rate is recorded alongside every cell because it is the quantity the
paper's explanation turns on, and a reproduction that matched the numbers while
contradicting the mechanism would not support the claim.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .emit import artifact, note, row
from .metrics import ert_pinned_parallel
from .provenance import Timer

ALPHA = 0.1

SPECS = {
    "MNIST": {"torchvision": "MNIST", "in_channels": 1, "image_size": 28, "classes": 10,
              "train_frac": 0.05, "cal_frac": 0.20, "epochs": 1,
              "normalize": ((0.5,), (0.5,))},
    "FashionMNIST": {"torchvision": "FashionMNIST", "in_channels": 1, "image_size": 28, "classes": 10,
                     "train_frac": 0.15, "cal_frac": 0.20, "epochs": 5,
                     "normalize": ((0.5,), (0.5,))},
    "CIFAR10": {"torchvision": "CIFAR10", "in_channels": 3, "image_size": 32, "classes": 10,
                "train_frac": 0.40, "cal_frac": 0.20, "epochs": 10,
                "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))},
}

PAPER_TABLE4 = {
    ("CIFAR10", "cumulative"): {"L1": 0.072, "KL": -0.017, "KL_plus": -0.030, "KL_minus": 0.012},
    ("CIFAR10", "likelihood"): {"L1": 0.016, "KL": 0.028, "KL_plus": 0.007, "KL_minus": 0.022},
    ("CIFAR100", "cumulative"): {"L1": 0.041, "KL": 0.191, "KL_plus": 0.016, "KL_minus": 0.175},
    ("CIFAR100", "likelihood"): {"L1": 0.007, "KL": 0.409, "KL_plus": 0.085, "KL_minus": 0.323},
    ("FashionMNIST", "cumulative"): {"L1": 0.165, "KL": -0.260, "KL_plus": -0.185, "KL_minus": -0.075},
    ("FashionMNIST", "likelihood"): {"L1": 0.098, "KL": -0.068, "KL_plus": -0.042, "KL_minus": -0.026},
    ("MNIST", "cumulative"): {"L1": 0.150, "KL": -0.216, "KL_plus": -0.159, "KL_minus": -0.057},
    ("MNIST", "likelihood"): {"L1": 0.145, "KL": -0.187, "KL_plus": -0.128, "KL_minus": -0.059},
}


class SimpleCNN(nn.Module):
    """networks-equivalent f-hat from the release's classification drivers."""

    def __init__(self, in_channels: int, image_size: int, n_outputs: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            out = self.pool(self.relu(self.conv1(dummy)))
            out = self.pool(self.relu(self.conv2(out)))
            flat_dim = out.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, n_outputs)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

    def predict_proba(self, x, batch_size: int = 512):
        self.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                out.append(F.softmax(self.forward(x[start:start + batch_size]), dim=1))
        return torch.cat(out).numpy()


class BinaryImageClassifier(nn.Module):
    """networks.py lines 329-400 — the ERT classifier for image covariates."""

    def __init__(self, in_channels: int, image_size: int, epochs: int = 10, lr: float = 1e-3,
                 batch_size: int = 32, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            out = self.pool(self.relu(self.conv1(dummy)))
            out = self.pool(self.relu(self.conv2(out)))
            flat_dim = out.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self._shape = (in_channels, image_size, image_size)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.sigmoid(self.fc2(x))

    def _as_images(self, x):
        tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return tensor.view(-1, *self._shape)

    def fit(self, x, y):
        from torch.utils.data import DataLoader, TensorDataset

        images = self._as_images(x)
        labels = torch.as_tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(images, labels), batch_size=self.batch_size, shuffle=True)
        self.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = self.criterion(self(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, x, batch_size: int = 512):
        images = self._as_images(x)
        self.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                out.append(self(images[start:start + batch_size]).squeeze(1))
        p = torch.cat(out).numpy()
        return np.column_stack([1.0 - p, p])


class _FoldFit:
    """Picklable fold worker: fit a fresh ERT classifier and score the held-out rows."""

    def __init__(self, in_channels: int, image_size: int, seed: int, threads: int = 4):
        self.in_channels, self.image_size, self.seed, self.threads = \
            in_channels, image_size, seed, threads

    def __call__(self, fold):
        from .metrics import fold_data

        train_index, test_index = fold
        x, cover = fold_data()
        x_train, cover_train, x_test = x[train_index], cover[train_index], x[test_index]
        torch.set_num_threads(self.threads)
        model = BinaryImageClassifier(self.in_channels, self.image_size, seed=self.seed)
        model.fit(x_train, cover_train)
        return model.predict_proba(x_test)[:, 1]


def negative_likelihood_scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """sadinle2019: S(X, Y) = -p(X)_Y."""
    return -probabilities[np.arange(len(labels)), labels]


def cumulative_scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """romano2020 / angelopoulos2020: sum of probabilities at least the true one."""
    true_probability = probabilities[np.arange(len(labels)), labels]
    return (probabilities * (probabilities >= true_probability[:, None])).sum(axis=1)


def set_sizes(probabilities: np.ndarray, threshold: float, strategy: str) -> np.ndarray:
    """The release's `get_volumes`, used only to record the empty-set rate."""
    if strategy == "likelihood":
        return (probabilities >= -threshold).sum(axis=1)
    ordered = np.sort(probabilities, axis=1)[:, ::-1]
    return (np.cumsum(ordered, axis=1) < threshold).sum(axis=1)


def _load(spec: dict, seed: int, root: str = "data/torchvision"):
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*spec["normalize"])])
    dataset = getattr(torchvision.datasets, spec["torchvision"])(
        root=root, train=True, download=True, transform=transform)
    total = len(dataset)
    n_train = int(spec["train_frac"] * total)
    n_cal = int(spec["cal_frac"] * total)
    generator = torch.Generator().manual_seed(seed)
    train, calibration, test = torch.utils.data.random_split(
        dataset, [n_train, n_cal, total - n_train - n_cal], generator=generator)

    def stack(subset):
        loader = torch.utils.data.DataLoader(subset, batch_size=1024, shuffle=False)
        xs, ys = zip(*[(x, y) for x, y in loader])
        return torch.cat(xs), torch.cat(ys)

    return stack(train), stack(calibration), stack(test)


def _cell(dataset: str, seed: int, fold_threads: int = 4) -> list[dict]:
    spec = SPECS[dataset]
    torch.manual_seed(seed)
    np.random.seed(seed)

    (x_train, y_train), (x_cal, y_cal), (x_test, y_test) = _load(spec, seed)

    model = SimpleCNN(spec["in_channels"], spec["image_size"], spec["classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    with Timer() as fit_timer:
        model.train()
        for _ in range(spec["epochs"]):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

    probabilities_cal = model.predict_proba(x_cal)
    probabilities_test = model.predict_proba(x_test)
    accuracy = float((probabilities_test.argmax(1) == y_test.numpy()).mean())
    note(f"table4 {dataset} seed={seed} test accuracy={accuracy:.4f} "
         f"(f-hat fit {fit_timer.wall_s:.0f}s)")

    flat_test = x_test.reshape(len(x_test), -1).numpy()
    records = []
    for strategy, scorer in (("likelihood", negative_likelihood_scores),
                             ("cumulative", cumulative_scores)):
        calibration_scores = np.sort(scorer(probabilities_cal, y_cal.numpy()))
        index = int(np.ceil((len(y_cal) + 1) * (1 - ALPHA)))
        threshold = float(calibration_scores[index])
        cover = (scorer(probabilities_test, y_test.numpy()) < threshold).astype(np.int64)
        sizes = set_sizes(probabilities_test, threshold, strategy)

        with Timer() as timer:
            values = ert_pinned_parallel(
                _FoldFit(spec["in_channels"], spec["image_size"], seed, threads=fold_threads),
                flat_test, cover, alpha=ALPHA,
            )
        record = {
            "dataset": dataset, "experiment": int(seed), "method": strategy,
            "n_train": int(len(y_train)), "n_calibration": int(len(y_cal)),
            "n_test": int(len(y_test)), "predictor_accuracy": accuracy,
            "marginal_coverage": float(cover.mean()),
            "empty_set_rate": float((sizes == 0).mean()),
            "mean_set_size": float(sizes.mean()),
            "conformal_threshold": threshold,
            "ert_seconds": round(timer.wall_s, 2),
            **values,
        }
        records.append(record)
        row("table4", record)
        note(f"table4 {dataset} seed={seed} {strategy}: L1-ERT={values['ERT_L1_miscoverage']:+.4f} "
             f"KL-ERT={values['ERT_logloss']:+.4f} KL+={values['ERT_logloss_over']:+.4f} "
             f"KL-={values['ERT_logloss_under']:+.4f} empty={record['empty_set_rate']:.4f} "
             f"({timer.wall_s:.0f}s)")
    return records


class _Seed:
    """Picklable seed worker, so whole seeds can run alongside each other.

    A single fold's ERT classifier takes minutes, and only the five folds of one
    strategy are in flight at a time, so the box sits half idle.  Seeds are
    independent - their own split, predictor and conformal threshold - so a
    couple of them running at once fills the machine without widening the fold
    pool past the point where a 32-row torch batch stops scaling.
    """

    def __init__(self, dataset: str, fold_threads: int):
        self.dataset, self.fold_threads = dataset, fold_threads

    def __call__(self, seed: int) -> list[dict]:
        return _cell(self.dataset, seed, fold_threads=self.fold_threads)


def _download(spec: dict, root: str = "data/torchvision") -> None:
    """Fetch the dataset once, before any fork, so workers never race on it."""
    import torchvision

    getattr(torchvision.datasets, spec["torchvision"])(root=root, train=True, download=True)


def run(config: dict) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    datasets = list(config.get("datasets", SPECS))
    seeds = list(config.get("seeds", range(10)))
    fold_threads = int(config.get("fold_threads", 8))
    seed_workers = int(config.get("seed_workers", 2))

    rows: list[dict] = []
    for dataset in datasets:
        _download(SPECS[dataset])
        worker = _Seed(dataset, fold_threads)
        if seed_workers <= 1:
            for seed in seeds:
                rows.extend(worker(seed))
        else:
            context = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(seed_workers, len(seeds)),
                                     mp_context=context) as pool:
                for records in pool.map(worker, seeds):
                    rows.extend(records)

    result = {"claim": "5", "protocol": {
        "alpha": ALPHA, "specs": {k: SPECS[k] for k in datasets}, "seeds": seeds,
        "ert_folds": 5, "ert_classifier": "BinaryImageClassifier(epochs=10, lr=1e-3, batch=32)",
        "strategies": ["likelihood (S = -p(X)_Y)", "cumulative (S = sum p >= p_Y)"],
        "source_drivers": [f"_generate_{d}_experiments.py" for d in datasets],
        "kl_plus_field": "ERT_logloss_over", "kl_minus_field": "ERT_logloss_under",
    }, "paper_table4": {f"{k[0]}|{k[1]}": v for k, v in PAPER_TABLE4.items()},
        "rows": rows}
    artifact(f"claim5_table4/raw_{'_'.join(datasets)}.json", result)
    return result
