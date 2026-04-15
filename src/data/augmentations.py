"""Student-side data augmentation pipelines for VPR training."""

from torchvision import transforms


def get_train_transform(augment: bool, image_size: int = 224, cfg: dict | None = None):
    """Return the student training transform.

    Args:
        augment: If True, apply stochastic augmentations (crop, jitter, etc.).
                 If False, deterministic resize + center-crop only.
        image_size: Target spatial size.
        cfg: Optional augmentation config dict (from default.yaml).
    """
    if cfg is None:
        cfg = {}

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if not augment:
        return transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    aug_list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=tuple(cfg.get("random_resized_crop", {}).get("scale", [0.7, 1.0])),
        ),
        transforms.ColorJitter(
            brightness=cfg.get("color_jitter", {}).get("brightness", 0.4),
            contrast=cfg.get("color_jitter", {}).get("contrast", 0.4),
            saturation=cfg.get("color_jitter", {}).get("saturation", 0.4),
            hue=cfg.get("color_jitter", {}).get("hue", 0.1),
        ),
        transforms.RandomGrayscale(p=cfg.get("random_grayscale_p", 0.1)),
    ]

    blur_cfg = cfg.get("gaussian_blur", {})
    if blur_cfg.get("enabled", False):
        aug_list.append(
            transforms.GaussianBlur(
                kernel_size=blur_cfg.get("kernel_size", 23),
                sigma=tuple(blur_cfg.get("sigma", [0.1, 2.0])),
            )
        )

    flip_cfg = cfg.get("random_horizontal_flip", {})
    if flip_cfg.get("enabled", False):
        aug_list.append(transforms.RandomHorizontalFlip(p=flip_cfg.get("p", 0.5)))

    aug_list.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(aug_list)


def get_eval_transform(image_size: int = 224):
    """Deterministic eval transform — matches the no-augment train path."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
