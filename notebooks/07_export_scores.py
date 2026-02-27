"""
Export precomputed per-entity similarity scores for the interactive frontend.

For each entity tag (excluding meta/book types) with >= 3 explicitly tagged
entries, this script computes:
  - Multi-prototype reference embeddings (1, 2, or 3 via k-means)
  - Per-entry cosine similarity scores against each prototype
  - Calibration stats (mean, std, p10-p50 in steps of 5) from explicitly-tagged entries
  - Specificity score: log(total_entries / entries_above_floor)

Inputs:
- data/embeddings_cache/<model>.npy  -- entry embeddings (16282, dims)
- data/embeddings_cache/entry_ids.json -- ordered entry IDs matching rows
- data/tag_classifications.json -- tag name -> {type, count}
- ../../words-of-brandon/wob_entries.json -- raw WoB entries (for explicit tags)

Outputs:
- data/scores_{model}.json -- per-entity scores, calibration, and metadata
- data/scores_manifest.json -- manifest listing available models (with --all)
- data/scores.json -- backward-compat copy of the default model's output

Usage:
    python 07_export_scores.py
    python 07_export_scores.py --model azure_openai --floor 0.70
    python 07_export_scores.py --all
    python 07_export_scores.py --two-proto-min 5 --three-proto-min 10
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from common.paths import DATA_DIR as data_dir, CACHE_DIR as cache_dir, WOB_PATH as wob_path
from common.models import ALL_MODELS, EXCLUDE_TYPES, MODEL_DISPLAY_NAMES
from common.embeddings import load_embeddings, normalize_embeddings

# -- CLI ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Export per-entity similarity scores for the interactive frontend",
)
parser.add_argument(
    "--model",
    choices=ALL_MODELS,
    default="azure_openai",
    help="Embedding model to use (default: azure_openai)",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Export scores for all models with cached embeddings",
)
parser.add_argument(
    "--floor",
    type=float,
    default=0.60,
    help="Minimum max-across-prototypes score to include an entry (default: 0.60)",
)
parser.add_argument(
    "--two-proto-min",
    type=int,
    default=5,
    help="Minimum explicit entries for 2 prototypes (default: 5)",
)
parser.add_argument(
    "--three-proto-min",
    type=int,
    default=10,
    help="Minimum explicit entries for 3 prototypes (default: 10)",
)
parser.add_argument(
    "--default-model",
    default="gemini",
    help="Default model for the manifest and backward-compat scores.json (default: gemini)",
)
args = parser.parse_args()

MIN_ENTRIES_FOR_REF = 3

# -- Shared data (loaded once) -----------------------------------------------

with open(wob_path) as f:
    raw_entries = json.load(f)

with open(data_dir / "tag_classifications.json") as f:
    tag_class = json.load(f)

entity_tags = {t for t, info in tag_class.items() if info["type"] not in EXCLUDE_TYPES}

entry_explicit_tags = {}
for e in raw_entries:
    eid = e["id"]
    explicit = [t for t in e["tags"] if t in entity_tags]
    if explicit:
        entry_explicit_tags[eid] = set(explicit)


def export_model(model_name, floor, two_proto_min, three_proto_min):
    """Export scores for a single model. Returns (output_dict, scores_path, entities_output)."""
    print(f"\n{'='*60}")
    print(f"Exporting: {model_name}")
    print(f"{'='*60}")

    embeddings, entry_ids = load_embeddings(model_name)
    eid_to_idx = {eid: idx for idx, eid in enumerate(entry_ids)}
    total_entries = len(entry_ids)

    print(f"  Embeddings: {embeddings.shape} ({embeddings.dtype})")
    print(f"  Floor: {floor}")

    # L2-normalize
    entries_norm = normalize_embeddings(embeddings).astype(np.float32)

    # Process each entity
    entities_output = {}
    skipped = []
    proto_distribution = {1: 0, 2: 0, 3: 0}

    for tag in sorted(entity_tags):
        tag_eids = [
            eid for eid, tags in entry_explicit_tags.items()
            if tag in tags and eid in eid_to_idx
        ]

        if len(tag_eids) < MIN_ENTRIES_FOR_REF:
            skipped.append((tag, len(tag_eids)))
            continue

        n_explicit = len(tag_eids)
        indices = [eid_to_idx[eid] for eid in tag_eids]
        tag_embeddings = entries_norm[indices]

        if n_explicit < two_proto_min:
            n_proto = 1
        elif n_explicit < three_proto_min:
            n_proto = 2
        else:
            n_proto = 3

        proto_distribution[n_proto] += 1

        if n_proto == 1:
            proto = tag_embeddings.mean(axis=0)
            pnorm = np.linalg.norm(proto)
            if pnorm > 0:
                proto = proto / pnorm
            proto_matrix = proto.reshape(1, -1)
        else:
            kmeans = KMeans(n_clusters=n_proto, n_init=10, random_state=42)
            kmeans.fit(tag_embeddings)
            centers = kmeans.cluster_centers_
            center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
            center_norms = np.where(center_norms == 0, 1.0, center_norms)
            proto_matrix = (centers / center_norms).astype(np.float32)

        sim = entries_norm @ proto_matrix.T

        explicit_indices = np.array(indices)
        explicit_sims = sim[explicit_indices]
        explicit_max = explicit_sims.max(axis=1)

        cal_mean = float(np.mean(explicit_max))
        cal_std = float(np.std(explicit_max))
        calibration = {"mean": round(cal_mean, 4), "std": round(cal_std, 4)}
        for pct in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
            key = f"p{pct}"
            calibration[key] = round(float(np.percentile(explicit_max, pct)), 4)

        max_scores = sim.max(axis=1)
        above_floor_mask = max_scores > floor
        entries_above_floor = int(above_floor_mask.sum())

        if entries_above_floor == 0:
            specificity = float(math.log(total_entries))
        else:
            specificity = float(math.log(total_entries / entries_above_floor))

        scores_dict = {}
        for idx in np.where(above_floor_mask)[0]:
            eid = entry_ids[idx]
            score_arr = [round(float(s), 2) for s in sim[idx]]
            scores_dict[str(eid)] = score_arr

        entities_output[tag] = {
            "specificity": round(specificity, 4),
            "entries_above_floor": entries_above_floor,
            "calibration": calibration,
            "prototypes": n_proto,
            "scores": scores_dict,
        }

    output = {
        "meta": {
            "model": model_name,
            "floor": floor,
            "total_entries": total_entries,
            "proto_thresholds": {
                "two": two_proto_min,
                "three": three_proto_min,
            },
        },
        "entities": entities_output,
    }

    # Write per-model scores file
    scores_path = data_dir / f"scores_{model_name}.json"
    with open(scores_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = scores_path.stat().st_size / (1024 * 1024)

    print(f"\n  Entities exported: {len(entities_output)}")
    print(f"  Entities skipped (< {MIN_ENTRIES_FOR_REF} entries): {len(skipped)}")
    print(f"  Prototype distribution:")
    for n, count in sorted(proto_distribution.items()):
        print(f"    {n} prototype(s): {count} entities")
    print(f"  scores_{model_name}.json: {size_mb:.1f} MB")

    # Sample calibration
    for sample in ["kaladin", "hoid", "cosmere", "kelsier"]:
        if sample in entities_output:
            ent = entities_output[sample]
            cal = ent["calibration"]
            print(
                f"    {sample}: mean={cal['mean']:.3f}, std={cal['std']:.3f}, "
                f"p50={cal['p50']:.3f}, proto={ent['prototypes']}, "
                f"spec={ent['specificity']:.2f}, above_floor={ent['entries_above_floor']}"
            )

    return output, scores_path, entities_output


# -- Main --------------------------------------------------------------------

if args.all:
    # Find all models with cached .npy files
    available = [m for m in ALL_MODELS if (cache_dir / f"{m}.npy").exists()]
    if not available:
        print("No cached embeddings found in", cache_dir)
        sys.exit(1)
    print(f"Found cached embeddings for: {', '.join(available)}")
    models_to_export = available
else:
    models_to_export = [args.model]

results = {}
for model in models_to_export:
    output, scores_path, entities_output = export_model(
        model, args.floor, args.two_proto_min, args.three_proto_min
    )
    results[model] = {
        "output": output,
        "path": scores_path,
        "dimensions": output["meta"]["total_entries"],
    }

# -- Generate manifest (when --all) ------------------------------------------

if args.all:
    # Determine default model
    default_model = args.default_model if args.default_model in results else list(results.keys())[0]

    manifest = {
        "models": [],
        "default": default_model,
    }
    for model_id in models_to_export:
        if model_id not in results:
            continue
        # Read actual embedding dimensions from the .npy shape
        npy_path = cache_dir / f"{model_id}.npy"
        dims = int(np.load(npy_path, mmap_mode='r').shape[1]) if npy_path.exists() else 0
        manifest["models"].append({
            "id": model_id,
            "label": MODEL_DISPLAY_NAMES.get(model_id, model_id),
            "dimensions": dims,
            "file": f"scores_{model_id}.json",
        })

    manifest_path = data_dir / "scores_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")
    print(f"  Models: {[m['id'] for m in manifest['models']]}")
    print(f"  Default: {default_model}")

    # Backward compat: copy default model's output to scores.json
    default_scores_path = data_dir / f"scores_{default_model}.json"
    compat_path = data_dir / "scores.json"
    shutil.copy2(default_scores_path, compat_path)
    print(f"  Copied {default_scores_path.name} -> scores.json (backward compat)")
else:
    # Single model: also write scores.json for backward compat
    single_path = results[args.model]["path"]
    compat_path = data_dir / "scores.json"
    shutil.copy2(single_path, compat_path)
    print(f"\nCopied {single_path.name} -> scores.json")

print("\nDone.")
