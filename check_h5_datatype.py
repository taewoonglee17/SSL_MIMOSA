import h5py

def inspect_dtypes(filepath):
    print(f"\nInspecting: {filepath}")
    with h5py.File(filepath, 'r') as f:
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name:30s} → dtype: {obj.dtype}")

        f.visititems(print_dataset_info)

# 첫 번째 파일
inspect_dtypes('/autofs/space/marduk_001/users/tommy/data/multicoil_train/train_data.h5')

# 두 번째 파일
inspect_dtypes('/autofs/space/marduk_001/users/tommy/mimosa_data/multicoil_train/mimosa_train.h5')
