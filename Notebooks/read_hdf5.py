import os

path = os.path.join(os.getcwd(), "tvr_clip-B32_text_word_feats.hdf5")
print("Using path:", path)

with h5py.File(path, "r") as f:
    for i in list(f.keys())[:10]:
        print(f"This is id: {i}, Shape: {f[str(i)].shape}")

    
