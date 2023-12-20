def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines

def compare(f1, f2):
    f1_lines = read_file(f1)[:100]
    f2_lines = read_file(f2)[:100]

    f1_paths = [line.split()[0] for line in f1_lines]
    f2_paths = [line.split()[0] for line in f2_lines]

    common_paths = set(f1_paths) & set(f2_paths)

    print(f"Number of common path in first 100 values: {len(common_paths)}")

if __name__ == "__main__":
    compare("/storage/dataset_sorted_by_ent.txt", "/storage/dataset_no_uav_sorted_by_unc.txt")