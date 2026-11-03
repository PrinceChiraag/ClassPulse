"""Back up and optionally remove exact duplicate training images and labels.

Dry-run is the default. Pass ``--apply`` to create verified backups and remove
redundant files. For each exact-image group, the first sorted filename is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 digest, reading it in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_duplicate_groups(images_dir: Path) -> list[tuple[str, list[Path]]]:
    """Group regular image files by exact content hash."""
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for image in sorted(images_dir.iterdir()):
        if image.is_symlink() or not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        by_hash[file_sha256(image)].append(image)
    return [
        (digest, sorted(paths))
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]


def matching_labels_are_identical(
    paths: list[Path], labels_dir: Path
) -> tuple[bool, str, dict[Path, bytes]]:
    """Require a regular matching label file for each image and equal bytes."""
    contents: dict[Path, bytes] = {}
    for image in paths:
        label = labels_dir / f"{image.stem}.txt"
        if label.is_symlink() or not label.is_file():
            return False, f"matching label missing or not a regular file: {label}", contents
        try:
            contents[image] = label.read_bytes()
        except OSError as exc:
            return False, f"could not read matching label {label}: {exc}", contents

    reference = contents[paths[0]]
    for image in paths[1:]:
        if contents[image] != reference:
            label = labels_dir / f"{image.stem}.txt"
            return False, f"label differs: {label}", contents
    return True, "labels identical", contents


def verify_group(
    digest: str, paths: list[Path], labels_dir: Path
) -> tuple[bool, str, dict[Path, bytes]]:
    """Recheck source image hashes and verify the matching labels."""
    for image in paths:
        if image.is_symlink() or not image.is_file():
            return False, f"image missing or not a regular file: {image}", {}
        try:
            if file_sha256(image) != digest:
                return False, f"image SHA-256 changed: {image}", {}
        except OSError as exc:
            return False, f"could not read image {image}: {exc}", {}
    return matching_labels_are_identical(paths, labels_dir)


def ensure_verified_backup(
    source: Path,
    destination: Path,
    *,
    expected_hash: str | None = None,
    expected_bytes: bytes | None = None,
) -> tuple[bool, str]:
    """Create or reuse a backup only when its contents match the source data."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file():
                return False, f"backup destination is not a regular file: {destination}"
            if expected_hash is not None:
                matches = file_sha256(destination) == expected_hash
            else:
                matches = destination.read_bytes() == expected_bytes
            if matches:
                return True, "existing verified backup reused"
            return False, f"backup already exists with different content: {destination}"

        shutil.copy2(source, destination)
        if expected_hash is not None:
            matches = file_sha256(destination) == expected_hash
        else:
            matches = destination.read_bytes() == expected_bytes
        if not matches:
            return False, f"backup verification failed: {destination}"
        return True, "backup created and verified"
    except OSError as exc:
        return False, f"backup operation failed for {destination}: {exc}"


def verified_backup_pair(
    image_backup: Path,
    label_backup: Path,
    image_hash: str,
    label_bytes: bytes,
) -> tuple[bool, str]:
    """Confirm both backups exist as regular files and match expected content."""
    try:
        for path in (image_backup, label_backup):
            if path.is_symlink() or not path.is_file():
                return False, f"backup missing or not a regular file: {path}"
        if file_sha256(image_backup) != image_hash:
            return False, f"image backup hash mismatch: {image_backup}"
        if label_backup.read_bytes() != label_bytes:
            return False, f"label backup content mismatch: {label_backup}"
        return True, "both backups verified"
    except OSError as exc:
        return False, f"could not verify backup pair: {exc}"


def restore_from_verified_backups(
    image: Path,
    label: Path,
    image_backup: Path,
    label_backup: Path,
    image_hash: str,
    label_bytes: bytes,
) -> tuple[bool, list[str]]:
    """Restore missing originals without overwriting files with changed content."""
    messages: list[str] = []
    backups_ok, reason = verified_backup_pair(
        image_backup, label_backup, image_hash, label_bytes
    )
    if not backups_ok:
        return False, [f"cannot restore: {reason}"]

    restored_all = True
    for source, backup, expected in (
        (image, image_backup, image_hash),
        (label, label_backup, label_bytes),
    ):
        try:
            if source.is_symlink():
                restored_all = False
                messages.append(f"RESTORE FAILED: refusing to overwrite symlink {source}")
                continue
            if source.exists():
                if source.is_file():
                    matches = (
                        file_sha256(source) == expected
                        if isinstance(expected, str)
                        else source.read_bytes() == expected
                    )
                    if matches:
                        messages.append(f"RESTORED/OK: original already present and verified: {source}")
                    else:
                        restored_all = False
                        messages.append(f"RESTORE FAILED: original exists with changed content: {source}")
                else:
                    restored_all = False
                    messages.append(f"RESTORE FAILED: original path is not a regular file: {source}")
                continue

            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, source)
            matches = (
                file_sha256(source) == expected
                if isinstance(expected, str)
                else source.read_bytes() == expected
            )
            if matches:
                messages.append(f"RESTORED: {source} from {backup}")
            else:
                restored_all = False
                messages.append(f"RESTORE FAILED: restored content did not verify: {source}")
        except OSError as exc:
            restored_all = False
            messages.append(f"RESTORE FAILED: {source}: {exc}")
    return restored_all, messages


def path_is_within(path: Path, directory: Path) -> bool:
    """Return whether path is directory itself or one of its descendants."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "datasets" / "phone_detection"
    default_images = dataset_root / "train" / "images"
    default_labels = dataset_root / "train" / "labels"
    default_backup = dataset_root / "duplicate_backup"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up verified redundant pairs, then remove them",
    )
    parser.add_argument("--images", type=Path, default=default_images, help="Training images directory")
    parser.add_argument("--labels", type=Path, default=default_labels, help="Matching training labels directory")
    parser.add_argument("--backup", type=Path, default=default_backup, help="Backup root directory")
    args = parser.parse_args()

    images_dir = args.images.resolve()
    labels_dir = args.labels.resolve()
    backup_root = args.backup.resolve()
    if not images_dir.is_dir():
        parser.error(f"Images directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        parser.error(f"Labels directory does not exist: {labels_dir}")
    if path_is_within(backup_root, images_dir) or path_is_within(backup_root, labels_dir):
        parser.error("Backup directory must be outside both source images and labels directories.")

    mode = "APPLY" if args.apply else "DRY RUN (no files will be changed)"
    print(f"Mode: {mode}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Backup root: {backup_root}")

    groups = discover_duplicate_groups(images_dir)
    pair_count = sum(len(paths) - 1 for _, paths in groups)
    print(f"Found {len(groups)} duplicate hash group(s), {pair_count} redundant image(s).")
    if not groups:
        print("Nothing to do.")
        return 0

    errors = 0
    for digest, paths in groups:
        keep = paths[0]
        redundant = paths[1:]
        safe, reason, initial_labels = verify_group(digest, paths, labels_dir)
        print(f"\nSHA-256 {digest}")
        print(f"  KEEP: {keep}")
        if not safe:
            errors += len(redundant)
            for image in redundant:
                print(f"  SKIP: {image} ({reason})")
            continue

        for image in redundant:
            label = labels_dir / f"{image.stem}.txt"
            image_backup = backup_root / "train" / "images" / image.name
            label_backup = backup_root / "train" / "labels" / label.name
            expected_label = initial_labels[image]

            if not args.apply:
                print(f"  BACKUP (planned): {image_backup}")
                print(f"  BACKUP (planned): {label_backup}")
                print(f"  REMOVE (planned): {image}")
                print(f"  REMOVE (planned): {label}")
                continue

            # Recheck this kept/redundant pair before making backups.
            pair_safe, pair_reason, current_labels = verify_group(
                digest, [keep, image], labels_dir
            )
            if (
                not pair_safe
                or current_labels.get(keep) != initial_labels.get(keep)
                or current_labels.get(image) != expected_label
            ):
                errors += 1
                print(f"  SKIP: {image} (source verification failed: {pair_reason})")
                continue

            image_ok, image_message = ensure_verified_backup(
                image, image_backup, expected_hash=digest
            )
            label_ok, label_message = ensure_verified_backup(
                label, label_backup, expected_bytes=expected_label
            )
            if not image_ok or not label_ok:
                errors += 1
                print(f"  IMAGE BACKUP: {image_message}")
                print(f"  LABEL BACKUP: {label_message}")
                print("  REMOVE: nothing removed; both backups are required")
                continue

            backups_ok, backup_reason = verified_backup_pair(
                image_backup, label_backup, digest, expected_label
            )
            if not backups_ok:
                errors += 1
                print(f"  BACKUP VERIFICATION FAILED: {backup_reason}")
                print("  REMOVE: nothing removed")
                continue

            # Recheck source hashes and labels immediately before unlinking.
            final_safe, final_reason, final_labels = verify_group(
                digest, [keep, image], labels_dir
            )
            if (
                not final_safe
                or final_labels.get(keep) != initial_labels.get(keep)
                or final_labels.get(image) != expected_label
            ):
                errors += 1
                print(f"  SKIP REMOVE: source changed after backup: {final_reason}")
                continue
            backups_ok, backup_reason = verified_backup_pair(
                image_backup, label_backup, digest, expected_label
            )
            if not backups_ok:
                errors += 1
                print(f"  SKIP REMOVE: backup verification failed: {backup_reason}")
                continue

            try:
                image.unlink()
                label.unlink()
                print(f"  BACKED UP: {image} -> {image_backup}")
                print(f"  BACKED UP: {label} -> {label_backup}")
                print(f"  REMOVED: {image}")
                print(f"  REMOVED: {label}")
            except OSError as exc:
                errors += 1
                print(f"  REMOVAL ERROR for pair {image} / {label}: {exc}")
                restored, restore_messages = restore_from_verified_backups(
                    image,
                    label,
                    image_backup,
                    label_backup,
                    digest,
                    expected_label,
                )
                for message in restore_messages:
                    print(f"  {message}")
                if restored:
                    print("  ROLLBACK: original image/label pair verified present")
                else:
                    print("  ROLLBACK INCOMPLETE: inspect the paths reported above")

    if args.apply:
        print(
            f"\nApply complete with {errors} error(s). "
            "Kept one image per verified duplicate group; see per-file backup/removal lines above."
        )
    else:
        print(
            f"\nDry run complete. {pair_count - errors} redundant image/label pair(s) "
            "are eligible; no files were changed."
        )
        print("Review this report, then rerun with --apply to back up and remove eligible pairs.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
