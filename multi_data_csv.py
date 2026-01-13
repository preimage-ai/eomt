import csv
import pandas as pd
from pathlib import Path

def build_unified_csv(
    root,
    multi_csv_path,
    output_csv_path,
    images_dir="ade20k_images",
    masks_dir="ade20k_mask",
    img_suffix=".jpg",
    mask_suffix=".png",
):
    root = Path(root)
    images_dir = root / images_dir
    masks_dir = root / masks_dir

    # -----------------------------
    # PART 1 — Load multi-source CSV
    # -----------------------------
    print("Loading multi-source CSV:", multi_csv_path)
    df_multi = pd.read_csv(multi_csv_path)

    multi_rows = []
    for _, row in df_multi.iterrows():
        scene_name = str(row["scene_name"]).strip()
        sources = str(row["sources"]).strip()
        class_names = str(row.get("class_names", "")).strip()

        multi_rows.append({
            "mode": "multi",
            "scene_name": scene_name,
            "sources": sources,
            "class_names": class_names,
            "full_img_path": "",
            "full_mask_path": "",
        })

    # -----------------------------
    # PART 2 — Scan folder-based dataset
    # -----------------------------
    folder_rows = []
    print("Scanning folder-based ADE20K dataset...")

    images = sorted(images_dir.glob(f"*{img_suffix}"))
    count_pairs = 0

    for img_path in images:
        stem = img_path.stem   # e.g. ade_train_00000001_seed42
        mask_path = masks_dir / f"{stem}{mask_suffix}"

        if not mask_path.exists():
            continue

        folder_rows.append({
            "mode": "folder",
            "scene_name": stem,
            "sources": "",
            "class_names": "",
            "full_img_path": str(img_path),
            "full_mask_path": str(mask_path),
        })
        count_pairs += 1

    print(f"✔ Found {count_pairs} folder-based image/mask pairs")

    # -----------------------------
    # PART 3 — Combine and write CSV
    # -----------------------------
    all_rows = multi_rows + folder_rows

    print(f"Writing unified CSV: {output_csv_path}")
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "scene_name",
                "sources",
                "class_names",
                "full_img_path",
                "full_mask_path",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("🎉 Done! Unified CSV generated.")
    print(f"Total rows: {len(all_rows)}")
    print(f" - Multi-source rows: {len(multi_rows)}")
    print(f" - Folder-based rows: {len(folder_rows)}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    build_unified_csv(
        root="ade20k",                                          # <-- your dataset root
        multi_csv_path="/home/shifu/nas/jd-dainty/qwen-image-edit/eomt_jd/merged_scenes_sanitized.csv",           # <-- your original CSV
        output_csv_path="multi_source_.csv", # <-- final CSV
    )
