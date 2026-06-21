SEED = 42

IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)

BATCH_SIZE = 32

VAL_SPLIT = 0.20

EPOCHS_FASE_1 = 15
EPOCHS_FASE_2 = 15

LR_FASE_1 = 1e-3
LR_FASE_2 = 1e-5

NUM_CAMADAS_FINE_TUNING = 30

DATASET_DIR = "/content/drive/MyDrive/Colab Notebooks/Kaggle Coffee Bean Dataset"

TRAIN_DIR = DATASET_DIR + "/train"
TEST_DIR = DATASET_DIR + "/test"
CSV_PATH = DATASET_DIR + "/Coffee Bean.csv"

CHECKPOINT_PATH = "outputs/checkpoints/efficientnetb0_best.weights.h5"
MODEL_PATH = "outputs/modelos/efficientnetb0_coffee.keras"
LOG_PATH = "outputs/logs/historico_treinamento.csv"

FIGURES_DIR = "outputs/figuras"
METRICS_DIR = "outputs/metricas"