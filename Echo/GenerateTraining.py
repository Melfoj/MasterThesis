import os
import random

AUGMENTED_DIR = "augmented_songs/"
OUTPUT_DIR = "training_pairs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_song_key(filename):
    # Extract song key (everything before first underscore)
    # or use another logic if you want, e.g. strip _orig, _pitch etc.
    if "_orig" in filename:
        return filename.split("_orig")[0]
    return filename.split("_")[0]

def build_file_groups():
    groups = {}
    for f in os.listdir(AUGMENTED_DIR):
        if not f.endswith(".wav"):
            continue
        key = get_song_key(f)
        groups.setdefault(key, []).append(f)
    return groups

def generate_pairs(groups, n_positive=1000, n_negative=1000):
    all_keys = list(groups.keys())
    positive_pairs = []
    negative_pairs = []

    # Positive pairs: different augmentations of the same song
    for key, files in groups.items():
        if len(files) < 2:
            continue
        # Generate pairs within this group
        for _ in range(n_positive // len(groups)):
            a, b = random.sample(files, 2)
            positive_pairs.append((a, b, 1))  # 1 = positive label

    # Negative pairs: pairs from different songs
    for _ in range(n_negative):
        key1, key2 = random.sample(all_keys, 2)
        a = random.choice(groups[key1])
        b = random.choice(groups[key2])
        negative_pairs.append((a, b, 0))  # 0 = negative label

    return positive_pairs, negative_pairs

def save_pairs(pairs, filename):
    with open(filename, "w") as f:
        for a, b, label in pairs:
            f.write(f"{a},{b},{label}\n")
    print(f"Saved {len(pairs)} pairs to {filename}")

def main():
    groups = build_file_groups()
    print(f"Found {len(groups)} songs/groups")

    pos_pairs, neg_pairs = generate_pairs(groups)
    print(f"Generated {len(pos_pairs)} positive and {len(neg_pairs)} negative pairs")

    save_pairs(pos_pairs, os.path.join(OUTPUT_DIR, "positive_pairs.txt"))
    save_pairs(neg_pairs, os.path.join(OUTPUT_DIR, "negative_pairs.txt"))

if __name__ == "__main__":
    main()
