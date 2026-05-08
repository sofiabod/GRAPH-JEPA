"""one-off migration: copy /sequential-ablation/seedN/ -> /<dataset>/sequential-ablation/seedN/

needed because the original sequential-ablation training wrote to a shared,
dataset-agnostic path. when reruns happened across datasets, last write won
and earlier datasets' checkpoints were silently lost.

usage (after sequential-ablation has been trained for the dataset whose
checkpoints currently live at /sequential-ablation/):

    modal run scripts/migrate_seq_ablation_path.py::main --dataset tgbn_trade

after this, the legacy path is preserved (this is a copy, not a move). once
both datasets' new paths are populated, delete the legacy one with:

    modal volume rm tgjepa-results /sequential-ablation
"""
import modal

app = modal.App("tgjepa-migrate-paths")
vol = modal.Volume.from_name("tgjepa-results", create_if_missing=False)


@app.function(volumes={"/results": vol}, timeout=600)
def migrate(dataset: str, seeds: list[int]):
    import shutil
    from pathlib import Path

    legacy_root = Path("/results/sequential-ablation")
    new_root = Path(f"/results/{dataset}/sequential-ablation")
    new_root.mkdir(parents=True, exist_ok=True)

    moved = []
    skipped = []
    for seed in seeds:
        src = legacy_root / f"seed{seed}"
        dst = new_root / f"seed{seed}"
        if not src.exists():
            skipped.append((seed, f"src missing: {src}"))
            continue
        if dst.exists():
            skipped.append((seed, f"dst already exists: {dst}"))
            continue
        # copytree preserves checkpoint.pt + any other files written by training
        shutil.copytree(src, dst)
        moved.append((seed, str(dst)))

    vol.commit()
    return {"moved": moved, "skipped": skipped}


@app.local_entrypoint()
def main(dataset: str = "tgbn_trade", seeds: str = "0,1,2,3,4"):
    seed_list = [int(s) for s in seeds.split(",")]
    print(f"migrating /sequential-ablation/seed{seed_list} -> "
          f"/{dataset}/sequential-ablation/seed{seed_list}")
    result = migrate.remote(dataset, seed_list)
    print(f"moved {len(result['moved'])} seed dirs:")
    for seed, dst in result["moved"]:
        print(f"  seed {seed} -> {dst}")
    if result["skipped"]:
        print(f"skipped {len(result['skipped'])}:")
        for seed, reason in result["skipped"]:
            print(f"  seed {seed}: {reason}")
