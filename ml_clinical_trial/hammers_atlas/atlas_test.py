import nibabel as nib
import pandas as pd
import numpy as np

# Check atlas_path
atlas_img = nib.load("Hammers_mith_atlas_n30r83_SPM5.nii.gz")
print("Atlas shape:", atlas_img.shape)
print("Atlas affine:\n", atlas_img.affine)
print("Unique labels:", set(atlas_img.get_fdata().astype(int).flatten()))

# Check atlas_labels_path (TXT)
print("\nFirst 10 lines of SPM5_n30r83_regiondef.txt:")
with open("SPM5_n30r83_regiondef.txt", "r") as f:
    for _ in range(10):
        print(f.readline().strip())

# Check the CSV mapping file
print("\nFirst 10 rows of n30r83_names_spreadsheet.csv:")
df = pd.read_csv("n30r83_names_spreadsheet.csv")
print(df.head(10))
print("\nCSV columns:", df.columns.tolist())



# --- load atlas ---
atlas_path = "Hammers_mith_atlas_n30r83_SPM5.nii.gz"
img = nib.load(atlas_path)
labels = np.unique(img.get_fdata().astype(int))

# --- parse CSV first column into ID->name ---
df = pd.read_csv("n30r83_names_spreadsheet.csv")

# first column with names
name_col = df.columns[0]
names = df[name_col].dropna().tolist()

# remove the header row if present
if names and isinstance(names[0], str) and names[0].strip().lower() == "region_name":
    names = names[1:]

# keep first 83 entries (some files have padding)
names = names[:83]

# build mapping: 1..83
id2name = {i+1: n for i, n in enumerate(names)}

# --- sanity checks ---
atlas_ids = sorted(int(x) for x in labels if x != 0)
missing = [i for i in atlas_ids if i not in id2name]
extra = [i for i in id2name if i not in atlas_ids]

print(f"Parsed {len(names)} names. Example: {names[:5]}")
print("Atlas nonzero label IDs:", atlas_ids[:10], "…")
print("Missing IDs in mapping:", missing)
print("Extra IDs in mapping:", extra)

# optional: save a clean txt for your preprocessor
with open("n30r83_id2name_clean.txt", "w") as f:
    for i in range(1, 84):
        f.write(f"{i} {id2name.get(i, 'MISSING_NAME')}\n")
print("Wrote n30r83_id2name_clean.txt")


# # 3. Create the preprocessor
# pre = JmapACTPreprocessor(
#     jmap_features=["t1_volume"],
#     strategy="flatten",  # could also be "stats" or "pca"
#     atlas_path="Hammers_mith_atlas_n30r83_SPM5.nii.gz",
#     atlas_labels_path="n30r83_id2name_clean.txt",
#     keep_channel_axis=False,
#     scale_volume=True,
#     verbose=True
# )