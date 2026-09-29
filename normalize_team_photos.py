from __future__ import annotations

import argparse
import re
from pathlib import Path

TEAM = {
    "Abdul Hammad": "abdul_hammad",
    "MD Faisal Raza": "md_faisal_raza",
    "Zishan Afroz": "zishan_afroz",
    "Zarish Parveen": "zarish_parveen",
    "Mobashra Fatima": "mobashra_fatima",
    "Nasiba Hoda": "nasiba_hoda",
}
ALLOWED = {".png", ".jpg", ".jpeg", ".webp"}


def norm_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").casefold()
    return re.sub(r"_+", "_", value)


def canonical_candidates(files: list[Path], target_stem: str, member_name: str) -> list[Path]:
    stems = {
        norm_stem(target_stem),
        norm_stem(member_name),
    }
    # Legacy project spellings retained only to migrate existing assets.
    if target_stem == "mobashra_fatima":
        stems.add("moobashra_fatima")
    if target_stem == "md_faisal_raza":
        stems.add("faisal_raza")

    matches = []
    for p in files:
        stem = norm_stem(p.stem)
        if stem in stems:
            matches.append(p)
    return sorted(
        matches,
        key=lambda p: (
            1 if p.stem.casefold().endswith("@2x") or p.stem.casefold().endswith("_2x") else 0,
            4 if p.suffix.casefold() == ".png" else 3 if p.suffix.casefold() == ".jpg" else 3 if p.suffix.casefold() == ".jpeg" else 2,
            p.stat().st_size if p.exists() else 0,
        ),
        reverse=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize Team Fresh Minds photos to first_lastname.PNG/JPG/JPEG/WEBP."
    )
    parser.add_argument(
        "--team-dir",
        default="dashboard/assets/team",
        help="Team photo directory (default: dashboard/assets/team)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files; without this flag the script is a dry-run.",
    )
    args = parser.parse_args()

    team_dir = Path(args.team_dir).expanduser().resolve()
    team_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in team_dir.iterdir() if p.is_file() and p.suffix.casefold() in ALLOWED]

    changes: list[tuple[Path, Path]] = []
    for member_name, target_stem in TEAM.items():
        matches = canonical_candidates(files, target_stem, member_name)
        source = matches[0] if matches else None
        if source is None:
            print(f"MISSING  {member_name:18s} -> {target_stem}.PNG/JPG/JPEG/WEBP")
            continue

        target = team_dir / f"{target_stem}{source.suffix.lower()}"
        if source.resolve() == target.resolve():
            print(f"OK       {member_name:18s} -> {target.name}")
            continue
        if target.exists():
            print(f"CONFLICT {member_name:18s} -> {target.name} already exists; leaving files unchanged")
            continue
        changes.append((source, target))
        print(f"RENAME   {source.name} -> {target.name}")

    if not args.apply:
        print(f"\nDry-run only: {len(changes)} rename(s) proposed. Add --apply to execute them.")
        return 0

    for source, target in changes:
        source.rename(target)
    print(f"\nApplied {len(changes)} rename(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
