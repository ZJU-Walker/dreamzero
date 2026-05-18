"""Combine long_task_1_part{1,2,3} into plate_task_0505 with renumbered demos."""

import re
import shutil
from pathlib import Path

DATASET_ROOT = Path("/iris/u/kewalk/dataset")
SOURCE_PARTS = [
    DATASET_ROOT / "long_task_1_part1",
    DATASET_ROOT / "long_task_1_part2",
    DATASET_ROOT / "long_task_1_part3",
]
OUTPUT_DIR = DATASET_ROOT / "plate_task_0505"


def demo_sort_key(path: Path) -> int:
    match = re.match(r"demo(\d+)$", path.name)
    if not match:
        raise ValueError(f"Unexpected folder name: {path}")
    return int(match.group(1))


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. Remove it first or choose another name."
        )
    OUTPUT_DIR.mkdir(parents=True)

    demos = []
    for part in SOURCE_PARTS:
        if not part.is_dir():
            raise FileNotFoundError(f"Source part not found: {part}")
        part_demos = sorted(
            (p for p in part.iterdir() if p.is_dir() and p.name.startswith("demo")),
            key=demo_sort_key,
        )
        demos.extend(part_demos)
        print(f"{part.name}: {len(part_demos)} demos")

    print(f"Total: {len(demos)} demos -> {OUTPUT_DIR}")

    for new_idx, src in enumerate(demos, start=1):
        dst = OUTPUT_DIR / f"demo{new_idx}"
        print(f"  {src.parent.name}/{src.name} -> {dst.name}")
        shutil.copytree(src, dst)

    print(f"Done. Wrote {len(demos)} demos to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
