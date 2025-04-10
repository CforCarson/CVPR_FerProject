import os
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data_loader import FER2013FolderDataset, SyntheticDataset

def debug_dataset(dataset_class, root_dir, transform=None, mode=None):
    """
    调试数据集，输出每个类别的样本数及总图片数量。
    Args:
        dataset_class: 数据集类 (如 FER2013FolderDataset 或 SyntheticDataset)
        root_dir: 数据集根目录路径
        transform: 数据增强操作
        mode: 'train' 或 'test' (对于 FER2013 数据集)
    Returns:
        dataset: 加载成功的数据集
    """
    if issubclass(dataset_class, FER2013FolderDataset):  # 只有 FER2013FolderDataset 需要 mode 参数
        dataset = dataset_class(root_dir, mode=mode, transform=transform)
        print(f"\n=== Debugging Dataset: {dataset_class.__name__} ({mode} mode)===")
    else:
        dataset = dataset_class(root_dir, transform=transform)
        print(f"\n=== Debugging Dataset: {dataset_class.__name__} ===")

    # 输出类别信息
    print(f"Classes: {dataset.classes}")
    print(f"Class-to-Index Mapping: {dataset.class_to_idx}")
    print(f"Total Samples Loaded: {len(dataset)}")

    # 按类别统计样本数
    class_counts = {cls_name: 0 for cls_name in dataset.classes}
    for _, target in dataset.samples:
        class_name = dataset.classes[target]
        class_counts[class_name] += 1
    
    print("Per-Class Sample Counts:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count}")

    return dataset


if __name__ == "__main__":
    # 配置数据集路径
    fer2013_dir = "./data/FER2013"
    synthetic_dir = "./data/Synthetic"  # 根据项目结构假定路径

    # 检查路径
    if not os.path.exists(fer2013_dir):
        print(f"ERROR: Path not found: {fer2013_dir}")
    if not os.path.exists(synthetic_dir):
        print(f"ERROR: Path not found: {synthetic_dir}")

    # 定义 transforms
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])

    # 调试 FER2013 数据集: train 和 test
    train_dataset = debug_dataset(FER2013FolderDataset, fer2013_dir, mode="train", transform=transform)
    test_dataset = debug_dataset(FER2013FolderDataset, fer2013_dir, mode="test", transform=transform)

    # 调试 Synthetic 数据集
    synthetic_dataset = debug_dataset(SyntheticDataset, synthetic_dir, transform=transform)

    # 验证 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    print("\n=== Verifying DataLoader ===")
    print(f"Total Batches in train_loader: {len(train_loader)}")