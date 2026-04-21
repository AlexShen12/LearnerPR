"""Evaluate LearnerPR weights on project-vpr dataset_a / dataset_b.

Loads a LearnerPR checkpoint (DINOv2-S + GeM), encodes the project-vpr
database and query images, and prints R@1,5,10,20 + mAP@20 using the same
ground-truth helpers and metrics as project-vpr/evaluate.py.

Usage (run from LearnerPR/LearnerPR/):
    python scripts/eval_project_vpr.py \\
        --weights  checkpoints/best.pt \\
        --project_vpr_root  ../../project-vpr \\
        --datasets_root     ../../project-vpr/datasets \\
        --datasets dataset_a dataset_b \\
        --batch_size 64 \\
        --device cuda

Optional: --save_predictions DIR writes per-dataset CSVs compatible with
project-vpr/evaluate.py for double-checking with the course harness.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── Repo bootstrap ────────────────────────────────────────────────────
# scripts/ lives one level below the inner repo root (LearnerPR/LearnerPR/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add src/ so bare imports like `from models.student import ...` work too.
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.student import DINOv2GeMStudent  # noqa: E402


# ── Model loading ─────────────────────────────────────────────────────

def load_learnerpr_model(weights_path: str, device: torch.device) -> DINOv2GeMStudent:
    model = DINOv2GeMStudent(
        backbone_name="facebook/dinov2-small",
        embed_dim=384,
        gem_p_init=3.0,
        use_projection=False,
    )
    if weights_path and os.path.isfile(weights_path):
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        # Strip projection head keys — not used when use_projection=False.
        state = {k: v for k, v in state.items() if not k.startswith("projection")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [weights] {len(missing)} missing keys (e.g. {missing[0]})")
        if unexpected:
            print(f"  [weights] {len(unexpected)} unexpected keys (e.g. {unexpected[0]})")
        print(f"Loaded weights from: {weights_path}")
    else:
        print("WARNING: no weights file found — using ImageNet pretrain only.")
    return model.to(device).eval()


# ── Image dataset for inference ───────────────────────────────────────

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _PathDataset(Dataset):
    def __init__(self, root: str, rel_paths: list[str]) -> None:
        self.root = root
        self.paths = rel_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(self.root, self.paths[idx])).convert("RGB")
        return _EVAL_TRANSFORM(img), idx


# ── Encoding ──────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_paths(
    model: DINOv2GeMStudent,
    root: str,
    rel_paths: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str = "Encoding",
) -> np.ndarray:
    """Encode a list of image paths; return (N, D) L2-normalised numpy array."""
    ds = _PathDataset(root, rel_paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    emb_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    for imgs, indices in tqdm(loader, desc=desc, leave=False):
        embs = model(imgs.to(device))
        emb_parts.append(embs.cpu().numpy())
        idx_parts.append(indices.numpy())
    embeddings = np.vstack(emb_parts)
    order = np.argsort(np.concatenate(idx_parts))
    return embeddings[order]


# ── Dataset ordering (mirrors project-vpr/train_example.py) ──────────

def load_paths(dataset_root: str, dataset_name: str) -> tuple[list[str], list[str]]:
    df = pd.read_parquet(os.path.join(dataset_root, "test.parquet"))
    if dataset_name == "dataset_a":
        db_df = df[df["split"] == "database"].sort_values("image_path")
        q_df = df[df["split"] == "queries"].sort_values("image_path")
    else:
        db_df = df[df["role"] == "database"]
        q_df = df[df["role"] == "queries"]
    return db_df["image_path"].tolist(), q_df["image_path"].tolist()


# ── Dynamic import of project-vpr metrics ────────────────────────────

def _import_vpr_evaluate(project_vpr_root: Path):
    eval_path = project_vpr_root / "evaluate.py"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"project-vpr/evaluate.py not found at {eval_path}. "
            "Pass --project_vpr_root to point at the project-vpr directory."
        )
    spec = importlib.util.spec_from_file_location("vpr_evaluate", eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Per-dataset evaluation ────────────────────────────────────────────

def eval_dataset(
    dataset_name: str,
    dataset_root: str,
    model: DINOv2GeMStudent,
    vpr_mod,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    save_predictions: str | None,
    top_k: int = 20,
) -> dict:
    parquet = os.path.join(dataset_root, "test.parquet")
    if not os.path.exists(parquet):
        print(f"  SKIP {dataset_name}: {parquet} not found.")
        return {}

    db_paths, q_paths = load_paths(dataset_root, dataset_name)
    print(f"  Database: {len(db_paths)} | Queries: {len(q_paths)}")

    db_emb = encode_paths(model, dataset_root, db_paths, batch_size, num_workers, device,
                           desc="  Encoding db")
    q_emb = encode_paths(model, dataset_root, q_paths, batch_size, num_workers, device,
                          desc="  Encoding queries")

    # Cosine retrieval (embeddings already L2-normalised).
    sim = q_emb @ db_emb.T
    rankings = np.argsort(-sim, axis=1)[:, :top_k]

    # Build ground truth.
    if dataset_name == "dataset_a":
        positives_per_query, _, _ = vpr_mod.load_dataset_a_gt(dataset_root)
    else:
        positives_per_query, _, _ = vpr_mod.load_dataset_b_gt(dataset_root)

    k_values = [1, 5, 10, 20]
    recalls = vpr_mod.compute_recall_at_k(rankings, positives_per_query, k_values)
    map_at_k = vpr_mod.compute_map_at_k(rankings, positives_per_query, k=top_k)

    results = {**recalls, f"mAP@{top_k}": map_at_k}

    # Optional CSV dump.
    if save_predictions:
        Path(save_predictions).mkdir(parents=True, exist_ok=True)
        csv_path = os.path.join(save_predictions, f"{dataset_name}.csv")
        rows = [
            {"query_index": i, "ranked_database_indices": ",".join(map(str, rankings[i]))}
            for i in range(len(q_paths))
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Predictions saved to: {csv_path}")

    return results


# ── Main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate LearnerPR weights on project-vpr datasets."
    )
    p.add_argument("--weights", type=str, default=None,
                   help="Path to a LearnerPR checkpoint (.pt). Default: checkpoints/best.pt")
    p.add_argument("--project_vpr_root", type=str, default=None,
                   help="Path to the project-vpr directory (must contain evaluate.py). "
                        "Default: <repo_root>/../../project-vpr")
    p.add_argument("--datasets_root", type=str, default=None,
                   help="Root containing dataset_a/ and/or dataset_b/. "
                        "Default: <project_vpr_root>/datasets")
    p.add_argument("--datasets", nargs="+", default=["dataset_a"],
                   choices=["dataset_a", "dataset_b"],
                   help="Datasets to evaluate (default: dataset_a).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_predictions", type=str, default=None,
                   help="Optional directory to write per-dataset CSV predictions.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve defaults relative to repo root.
    weights = args.weights or str(_REPO_ROOT / "checkpoints" / "best.pt")
    project_vpr_root = Path(args.project_vpr_root) if args.project_vpr_root \
        else (_REPO_ROOT / ".." / ".." / "project-vpr").resolve()
    datasets_root = args.datasets_root or str(project_vpr_root / "datasets")

    device = torch.device(
        args.device if args.device == "cpu" or not torch.cuda.is_available()
        else args.device
    )
    if not torch.cuda.is_available() and args.device != "cpu":
        print(f"WARNING: CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("=" * 60)
    print("LearnerPR → project-vpr evaluation")
    print("=" * 60)
    print(f"Weights:       {weights}")
    print(f"project-vpr:   {project_vpr_root}")
    print(f"datasets_root: {datasets_root}")
    print(f"Datasets:      {args.datasets}")
    print(f"Device:        {device}")
    print("=" * 60)

    vpr_mod = _import_vpr_evaluate(project_vpr_root)
    model = load_learnerpr_model(weights, device)

    all_results: dict[str, dict] = {}
    for ds_name in args.datasets:
        ds_root = os.path.join(datasets_root, ds_name)
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds_name}  ({ds_root})")
        results = eval_dataset(
            dataset_name=ds_name,
            dataset_root=ds_root,
            model=model,
            vpr_mod=vpr_mod,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            save_predictions=args.save_predictions,
        )
        all_results[ds_name] = results
        if results:
            print(f"\n  Results for {ds_name}:")
            for metric, val in results.items():
                print(f"    {metric}: {val:.2f}%")

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Summary")
        print("=" * 60)
        for ds_name, res in all_results.items():
            if res:
                line = "  ".join(f"{k}: {v:.2f}%" for k, v in res.items())
                print(f"  {ds_name}: {line}")

    if device.type == "cuda":
        alloc_peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        reserved_peak = torch.cuda.max_memory_reserved(device) / (1024**3)
        print(f"\n{'─' * 60}")
        print(
            "GPU memory (PyTorch peak this run): "
            f"allocated max {alloc_peak:.3f} GiB, reserved max {reserved_peak:.3f} GiB"
        )


if __name__ == "__main__":
    main()
