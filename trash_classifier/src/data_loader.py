import kagglehub
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch
import os
import numpy as np

def _find_imagefolder_root(base_path, max_depth=6):
    """
    Nếu dataset được đóng gói 1-2 cấp thư mục (ví dụ: .../versions/1/Trash DataSet/<class folders>),
    hàm này sẽ đi xuống các thư mục con cho tới khi tìm được thư mục chứa các folder class (thư mục con có file ảnh).
    """
    cur = base_path
    for _ in range(max_depth):
        # list thư mục con (loại bỏ file)
        try:
            entries = os.listdir(cur)
        except Exception:
            break
        subdirs = [d for d in entries if os.path.isdir(os.path.join(cur, d))]
        if not subdirs:
            break

        # nếu có nhiều thư mục con, kiểm tra xem những thư mục con đó có chứa ảnh (=> đây là root cần dùng)
        any_sub_contains_images = False
        for sd in subdirs:
            sd_path = os.path.join(cur, sd)
            try:
                sd_files = os.listdir(sd_path)
            except Exception:
                sd_files = []
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) for f in sd_files):
                any_sub_contains_images = True
                break

        if any_sub_contains_images:
            # cur là thư mục cha có các thư mục lớp bên trong => return cur
            return cur

        # nếu chỉ có 1 thư mục con, đi xuống 1 bước để "bóc lót"
        if len(subdirs) == 1:
            cur = os.path.join(cur, subdirs[0])
            continue

        # nếu có nhiều thư mục con mà không chứa ảnh trực tiếp, chọn thư mục con có nhiều thư mục con hơn làm ứng viên
        moved = False
        for sd in subdirs:
            sd_path = os.path.join(cur, sd)
            try:
                inner_subdirs = [x for x in os.listdir(sd_path) if os.path.isdir(os.path.join(sd_path, x))]
            except Exception:
                inner_subdirs = []
            if inner_subdirs:
                cur = sd_path
                moved = True
                break
        if not moved:
            # không thể tiếp tục, quay về cur hiện tại
            return cur
    return cur

def get_data_loaders(config):
    path = kagglehub.dataset_download(config.DATASET_PATH)
    print(f"✅ Dataset path: {path}")

    # tìm root thực sự chứa các thư mục lớp
    dataset_root_path = os.path.join(path, 'Trash DataSet')
    if not os.path.exists(dataset_root_path):
        dataset_root_path = path

    # nếu folder vẫn không chứa trực tiếp các class, đi sâu thêm
    dataset_root_path = _find_imagefolder_root(dataset_root_path)
    print(f"ℹ️ Sử dụng thư mục dữ liệu: {dataset_root_path}")

    transform_train = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ImageFolder(dataset_root_path)

    # Debug: in ra các class tìm được
    print(f"🔍 Tìm thấy {len(full_dataset.classes)} lớp: {full_dataset.classes}")

    if len(full_dataset) == 0:
        raise RuntimeError("Dataset rỗng! Vui lòng kiểm tra lại đường dẫn/dataset.")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val

    # WeightedRandomSampler (robust với Subset)
    if hasattr(train_dataset, 'indices'):
        train_indices = train_dataset.indices
    else:
        train_indices = list(range(len(train_dataset)))

    # Lấy nhãn từ underlying ImageFolder
    try:
        train_targets = [train_dataset.dataset.targets[i] for i in train_indices]
    except Exception:
        # fallback: nếu không có .targets thì try lấy từ samples
        train_targets = [train_dataset.dataset.samples[i][1] for i in train_indices]

    num_classes = len(full_dataset.classes)
    class_sample_count = np.array([len(np.where(np.array(train_targets) == t)[0]) for t in range(num_classes)])
    # tránh chia cho 0
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in train_targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader, full_dataset.classes, transform_val
