import pandas as pd
import tensorflow as tf

from src import config


def carregar_csv_dataset():
    df = pd.read_csv(config.CSV_PATH)
    return df


def exibir_resumo_csv():
    df = carregar_csv_dataset()

    print("Dimensão do CSV:", df.shape)
    print("\nColunas:")
    print(df.columns.tolist())

    print("\nPrimeiras linhas:")
    print(df.head())

    return df


def carregar_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        validation_split=config.VAL_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        validation_split=config.VAL_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    train_ds = otimizar_dataset(train_ds, embaralhar=True)
    val_ds = otimizar_dataset(val_ds, embaralhar=False)
    test_ds = otimizar_dataset(test_ds, embaralhar=False)

    return train_ds, val_ds, test_ds, class_names


def otimizar_dataset(dataset, embaralhar=False):
    autotune = tf.data.AUTOTUNE

    if embaralhar:
        dataset = dataset.shuffle(1000, seed=config.SEED)

    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=autotune)

    return dataset