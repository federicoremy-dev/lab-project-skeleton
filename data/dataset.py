import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir: percorso a tiny-imagenet-200/
        split: 'train' o 'val'
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # leggi le classi
        classes = sorted(os.listdir(os.path.join(root_dir, 'train')))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        if split == 'train':
            for cls in classes:
                cls_dir = os.path.join(root_dir, 'train', cls, 'images')
                for img_name in os.listdir(cls_dir):
                    self.samples.append((
                        os.path.join(cls_dir, img_name),
                        self.class_to_idx[cls]
                    ))

        elif split == 'val':
            # leggi le annotazioni del val set
            val_annotations = os.path.join(root_dir, 'val', 'val_annotations.txt')
            with open(val_annotations) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, cls = parts[0], parts[1]
                    if cls in self.class_to_idx:
                        self.samples.append((
                            os.path.join(root_dir, 'val', 'images', img_name),
                            self.class_to_idx[cls]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label