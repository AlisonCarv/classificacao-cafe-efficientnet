from tensorflow import keras

from src import config


def criar_callbacks():
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=config.CHECKPOINT_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    csv_logger = keras.callbacks.CSVLogger(
        filename=config.LOG_PATH,
        separator=",",
        append=False
    )

    return [checkpoint, early_stop, reduce_lr, csv_logger]


def compilar_modelo(modelo, learning_rate):
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return modelo


def treinar_fase_1(modelo, train_ds, val_ds, callbacks):
    print("\nIniciando fase 1: transfer learning com EfficientNetB0 congelada.\n")

    historico = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FASE_1,
        callbacks=callbacks,
        verbose=1
    )

    return modelo, historico


def treinar_fase_2(modelo, train_ds, val_ds, callbacks):
    print("\nIniciando fase 2: fine-tuning das últimas camadas da EfficientNetB0.\n")

    historico = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FASE_2,
        callbacks=callbacks,
        verbose=1
    )

    return modelo, historico


def salvar_modelo(modelo):
    modelo.save(config.MODEL_PATH)
    print(f"\nModelo salvo em: {config.MODEL_PATH}\n")