import tensorflow as tf
from tensorflow import keras

from src import config


def criar_aumento_dados():
    aumento_dados = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomContrast(0.15),
        ],
        name="aumento_dados"
    )

    return aumento_dados


def criar_modelo(num_classes):
    tf.random.set_seed(config.SEED)

    aumento_dados = criar_aumento_dados()

    modelo_base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=config.IMG_SHAPE
    )

    modelo_base.trainable = False

    entrada = keras.Input(shape=config.IMG_SHAPE, name="entrada_imagem")

    x = aumento_dados(entrada)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = modelo_base(x, training=False)

    x = keras.layers.GlobalAveragePooling2D(name="pooling_global")(x)
    x = keras.layers.BatchNormalization(name="normalizacao_batch")(x)

    x = keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.L2(0.001),
        name="densa_128"
    )(x)

    x = keras.layers.Dropout(0.4, name="dropout_40")(x)

    saida = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="saida_softmax"
    )(x)

    modelo = keras.Model(
        inputs=entrada,
        outputs=saida,
        name="EfficientNetB0_CoffeeBean"
    )

    return modelo, modelo_base


def preparar_fine_tuning(modelo_base, num_camadas_liberadas):
    modelo_base.trainable = True

    for camada in modelo_base.layers[:-num_camadas_liberadas]:
        camada.trainable = False

    for camada in modelo_base.layers[-num_camadas_liberadas:]:
        camada.trainable = True

    print(f"Fine-tuning ativado nas últimas {num_camadas_liberadas} camadas.")