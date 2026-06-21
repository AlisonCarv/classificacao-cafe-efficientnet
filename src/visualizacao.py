import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import config


def plotar_amostras_dataset(dataset, class_names):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(10, 10))

    for imagens, rotulos in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)

            plt.imshow(imagens[i].numpy().astype("uint8"))

            indice_classe = np.argmax(rotulos[i].numpy())
            plt.title(class_names[indice_classe])
            plt.axis("off")

    plt.tight_layout()
    caminho = f"{config.FIGURES_DIR}/amostras_dataset.png"
    plt.savefig(caminho, dpi=300)
    plt.show()

    print(f"Amostras salvas em: {caminho}")


def plotar_historico(historico, nome_arquivo):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    acc = historico.history["accuracy"]
    val_acc = historico.history["val_accuracy"]
    loss = historico.history["loss"]
    val_loss = historico.history["val_loss"]

    epocas = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epocas, acc, label="Treinamento")
    plt.plot(epocas, val_acc, label="Validação")
    plt.title("Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    caminho_acc = f"{config.FIGURES_DIR}/{nome_arquivo}_acuracia.png"
    plt.savefig(caminho_acc, dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epocas, loss, label="Treinamento")
    plt.plot(epocas, val_loss, label="Validação")
    plt.title("Loss por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    caminho_loss = f"{config.FIGURES_DIR}/{nome_arquivo}_loss.png"
    plt.savefig(caminho_loss, dpi=300)
    plt.show()

    print(f"Histórico salvo em: {caminho_acc} e {caminho_loss}")


def plotar_metricas_por_classe(nomes_classes, valores, nome_metrica):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=nomes_classes, y=valores)

    plt.ylim(0, 1)
    plt.title(f"{nome_metrica} por Classe")
    plt.xlabel("Classe")
    plt.ylabel(nome_metrica)

    for indice, valor in enumerate(valores):
        plt.text(indice, valor + 0.01, f"{valor:.2f}", ha="center")

    plt.tight_layout()

    caminho = f"{config.FIGURES_DIR}/{nome_metrica.lower()}.png"
    plt.savefig(caminho, dpi=300)
    plt.show()

    print(f"Gráfico salvo em: {caminho}")