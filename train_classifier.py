from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leaf classifier from folder dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/leaf_dataset"),
        help="Root folder dataset: data/leaf_dataset/<class_name>/*.jpg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Folder output model + metadata.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size training.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split ratio (0-1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _validate_dataset_root(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder tidak ditemukan: {data_dir}")
    class_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError("Minimal butuh 2 kelas folder untuk klasifikasi.")

    image_count = 0
    for class_dir in class_dirs:
        image_count += sum(
            1
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
            for _ in class_dir.glob(ext)
        )
    if image_count < 20:
        raise ValueError("Jumlah gambar terlalu sedikit. Saran minimal 20 gambar total.")


def train(args: argparse.Namespace) -> None:
    from tensorflow import keras

    _validate_dataset_root(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    class_names = list(train_ds.class_names)
    num_classes = len(class_names)

    autotune = keras.utils.AUTOTUNE
    train_ds = train_ds.cache().shuffle(512).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(args.img_size, args.img_size, 3)),
            data_augmentation,
            keras.layers.Rescaling(1.0 / 255.0),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="leaf_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = args.output_dir / "leaf_classifier.keras"
    labels_path = args.output_dir / "labels.json"
    model.save(model_path)
    labels_path.write_text(json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nTraining selesai.")
    print(f"Model disimpan di: {model_path}")
    print(f"Label disimpan di: {labels_path}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
