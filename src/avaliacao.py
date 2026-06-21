import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from src import config
from src.visualizacao import plotar_metricas_por_classe


def obter_rotulos_e_predicoes(modelo, test_ds):
    y_true = []
    y_pred = []

    for imagens, rotulos in test_ds:
        predicoes = modelo.predict(imagens, verbose=0)

        y_pred.extend(np.argmax(predicoes, axis=1))
        y_true.extend(np.argmax(rotulos.numpy(), axis=1))

    return np.array(y_true), np.array(y_pred)


def gerar_matriz_confusao(y_true, y_pred, class_names, acuracia):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    matriz = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.title(f"Matriz de Confusão - Acurácia: {acuracia:.2%}")
    plt.tight_layout()

    caminho = f"{config.FIGURES_DIR}/matriz_confusao.png"
    plt.savefig(caminho, dpi=300)
    plt.show()

    print(f"Matriz de confusão salva em: {caminho}")


def salvar_relatorio(y_true, y_pred, class_names, acuracia):
    os.makedirs(config.METRICS_DIR, exist_ok=True)

    relatorio = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    caminho = f"{config.METRICS_DIR}/relatorio_classificacao.txt"

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(f"Acurácia: {acuracia:.4f}\n\n")
        arquivo.write(relatorio)

    print("\nRelatório de classificação:\n")
    print(f"Acurácia: {acuracia:.4f}\n")
    print(relatorio)
    print(f"Relatório salvo em: {caminho}")


def gerar_graficos_metricas(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    plotar_metricas_por_classe(class_names, precision, "precision")
    plotar_metricas_por_classe(class_names, recall, "recall")
    plotar_metricas_por_classe(class_names, f1, "f1_score")


def avaliar_modelo(modelo, test_ds, class_names):
    y_true, y_pred = obter_rotulos_e_predicoes(modelo, test_ds)

    acuracia = accuracy_score(y_true, y_pred)

    gerar_matriz_confusao(y_true, y_pred, class_names, acuracia)
    gerar_graficos_metricas(y_true, y_pred, class_names)
    salvar_relatorio(y_true, y_pred, class_names, acuracia)

    return acuracia