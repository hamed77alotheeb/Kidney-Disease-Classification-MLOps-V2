from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "splits"
MANIFEST_PATH = DATA_DIR / "data_manifest_unique.csv"

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

EXPECTED_COUNTS = {
    ("train", "Cyst"): 2298,
    ("train", "Normal"): 3501,
    ("train", "Stone"): 951,
    ("train", "Tumor"): 1598,

    ("val", "Cyst"): 492,
    ("val", "Normal"): 750,
    ("val", "Stone"): 204,
    ("val", "Tumor"): 342,

    ("test", "Cyst"): 494,
    ("test", "Normal"): 751,
    ("test", "Stone"): 205,
    ("test", "Tumor"): 343,
}


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def main():
    errors = []

    print("=" * 80)
    print("KIDNEY DATASET VALIDATION")
    print("=" * 80)

    print("Project:", PROJECT_ROOT)
    print("Dataset:", DATA_DIR)
    print("Manifest:", MANIFEST_PATH)

    if not DATA_DIR.is_dir():
        print("ERROR: Dataset directory not found.")
        return 1

    if not MANIFEST_PATH.is_file():
        print("ERROR: Dataset manifest not found.")
        return 1

    # --------------------------------------------------------
    # 1. Read the manifest
    # --------------------------------------------------------

    manifest_hashes = {}

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "split", "class", "filename", "sha256"
        }

        if not required_columns.issubset(reader.fieldnames or []):
            print("ERROR: Required manifest columns are missing.")
            return 1

        for row in reader:
            key = (
                row["split"],
                row["class"],
                row["filename"]
            )

            if key in manifest_hashes:
                errors.append(
                    f"Duplicate manifest entry: {key}"
                )
            else:
                manifest_hashes[key] = row["sha256"]

    print("\nManifest entries:", len(manifest_hashes))

    # --------------------------------------------------------
    # 2. Check files, counts, image readability, and hashes
    # --------------------------------------------------------

    actual_hashes = {}
    hash_locations = defaultdict(list)
    class_counts = Counter()

    for split in SPLITS:
        print(f"\n[{split.upper()}]")

        for class_name in CLASSES:
            folder = DATA_DIR / split / class_name

            if not folder.is_dir():
                errors.append(f"Missing directory: {folder}")
                continue

            images = sorted([
                path
                for path in folder.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            ])

            key_count = (split, class_name)
            class_counts[key_count] = len(images)

            expected_count = EXPECTED_COUNTS[key_count]

            print(
                f"{class_name:10}: {len(images):5} "
                f"(expected {expected_count})"
            )

            if len(images) != expected_count:
                errors.append(
                    f"Wrong image count for {split}/{class_name}: "
                    f"{len(images)} != {expected_count}"
                )

            for image_path in images:
                key = (split, class_name, image_path.name)

                try:
                    image_hash = sha256_file(image_path)

                    with Image.open(image_path) as image:
                        image.verify()

                    actual_hashes[key] = image_hash
                    hash_locations[image_hash].append(key)

                except Exception as exception:
                    errors.append(
                        f"Unreadable image: {image_path} "
                        f"({exception})"
                    )

    # --------------------------------------------------------
    # 3. Compare manifest and files
    # --------------------------------------------------------

    actual_keys = set(actual_hashes)
    manifest_keys = set(manifest_hashes)

    files_without_manifest = actual_keys - manifest_keys
    manifest_without_files = manifest_keys - actual_keys

    for key in sorted(files_without_manifest):
        errors.append(f"File missing from manifest: {key}")

    for key in sorted(manifest_without_files):
        errors.append(f"Manifest entry missing on disk: {key}")

    hash_mismatches = []

    for key in actual_keys & manifest_keys:
        if actual_hashes[key] != manifest_hashes[key]:
            hash_mismatches.append(key)
            errors.append(f"SHA-256 mismatch: {key}")

    # --------------------------------------------------------
    # 4. Detect exact duplicates anywhere in the split dataset
    # --------------------------------------------------------

    duplicate_groups = {
        image_hash: keys
        for image_hash, keys in hash_locations.items()
        if len(keys) > 1
    }

    train_hashes = {
        image_hash
        for image_hash, keys in hash_locations.items()
        if any(key[0] == "train" for key in keys)
    }

    val_hashes = {
        image_hash
        for image_hash, keys in hash_locations.items()
        if any(key[0] == "val" for key in keys)
    }

    test_hashes = {
        image_hash
        for image_hash, keys in hash_locations.items()
        if any(key[0] == "test" for key in keys)
    }

    train_val = train_hashes & val_hashes
    train_test = train_hashes & test_hashes
    val_test = val_hashes & test_hashes

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    total_images = len(actual_hashes)
    total_train = sum(
        class_counts[("train", name)] for name in CLASSES
    )
    total_val = sum(
        class_counts[("val", name)] for name in CLASSES
    )
    total_test = sum(
        class_counts[("test", name)] for name in CLASSES
    )

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print("Total images:", total_images)
    print("Train:", total_train)
    print("Validation:", total_val)
    print("Test:", total_test)

    print("Manifest entries:", len(manifest_hashes))
    print("Unreadable / invalid files and other errors:", len(errors))
    print("SHA-256 mismatches:", len(hash_mismatches))
    print("Exact duplicate groups:", len(duplicate_groups))

    print("\nExact duplicate overlap:")
    print("Train ∩ Validation:", len(train_val))
    print("Train ∩ Test:", len(train_test))
    print("Validation ∩ Test:", len(val_test))

    if total_images != 11929:
        errors.append(
            f"Expected 11929 images, found {total_images}"
        )

    if len(manifest_hashes) != 11929:
        errors.append(
            f"Expected 11929 manifest entries, found "
            f"{len(manifest_hashes)}"
        )

    if train_val or train_test or val_test:
        errors.append("Exact duplicate hashes cross split boundaries.")

    if duplicate_groups:
        errors.append(
            "Exact duplicate images remain inside the split dataset."
        )

    print("\n" + "=" * 80)

    if errors:
        print("FAIL: Dataset validation did not pass.")
        print("\nFirst issues:")
        for error in errors[:30]:
            print("-", error)
        return 1

    print("PASS: Dataset validation completed successfully.")
    print("All image counts and manifest hashes match.")
    print("No exact duplicates were found in the split dataset.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
