# =========================================================
# IMPORTS
# =========================================================
import os
import gc
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import applications, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from SwarmPackagePy import pso, wsa
import pandas as pd
import matplotlib.pyplot as plt
import random


# =========================================================
# 1. CARREGAMENTO E PREPARAÇÃO DO DATASET
# =========================================================

# Caminho para o dataset (cada pasta = uma classe)
dataset_dir = r"C:\Users\joaop\Desktop\ISEC\IC\trabalho3\dataset"

# Tamanho exigido pela MobileNetV2
img_size = (224, 224)

# Listas para armazenar imagens e rótulos
X, y = [], []

# Lista de classes (nomes das pastas)
classes = sorted([
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d))
])

# Dicionário para guardar ficheiros por classe
files_per_class = {}
counts = []

# Contagem de imagens por classe
for c in classes:
    files = [
        f for f in os.listdir(os.path.join(dataset_dir, c))
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    files_per_class[c] = files
    counts.append(len(files))

# Undersampling: número mínimo de imagens por classe
min_samples = min(counts)
print(f"Balanceamento: {min_samples} imagens por classe")

# Construção do dataset balanceado
for idx, c in enumerate(classes):

    # Baralhar imagens da classe
    files = files_per_class[c].copy()
    np.random.shuffle(files)

    # Selecionar apenas min_samples
    files = files[:min_samples]

    for f in files:
        path = os.path.join(dataset_dir, c, f)

        # Abrir imagem e garantir RGB
        img = Image.open(path).convert("RGB")

        # Redimensionar para 224x224
        img = img.resize(img_size)

        # Converter para array float32
        arr = np.array(img, dtype=np.float32)

        # Pré-processamento correto para MobileNetV2
        # (normalização esperada pelo modelo pré-treinado)
        arr = preprocess_input(arr)

        X.append(arr)
        y.append(idx)

# Converter listas para arrays numpy
X = np.array(X)
y = np.array(y)


# =========================================================
# 2. DIVISÃO EM TREINO / VALIDAÇÃO / TESTE
# =========================================================

# 20% para teste (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=0
)

# 20% do treino para validação
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=0
)


# =========================================================
# 3. DEFINIÇÃO DO MODELO (TRANSFER LEARNING)
# =========================================================
def criar_modelo(n_neurons, lr):
    """
    Cria um modelo baseado em MobileNetV2 com camadas densas finais.
    Apenas as camadas densas são treinadas.
    """

    # Rede base pré-treinada no ImageNet
    base_model = applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Congelar pesos da rede base
    base_model.trainable = False

    # Construção do modelo completo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(int(n_neurons), activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(classes), activation="softmax")
    ])

    # Compilação do modelo
    model.compile(
        optimizer=Adam(learning_rate=float(lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================================================
# 4. SUBCONJUNTOS FIXOS PARA A FUNÇÃO DE FITNESS
# =========================================================
# Utilizados para reduzir custo computacional e tornar a fitness consistente

idx_t_fixos = np.random.choice(
    len(X_train),
    size=min(50, len(X_train)),
    replace=False
)

idx_v_fixos = np.random.choice(
    len(X_val),
    size=min(25, len(X_val)),
    replace=False
)


# =========================================================
# 5. FUNÇÃO DE FITNESS (AVALIA HIPERPARÂMETROS)
# =========================================================
def fitness(hp):
    """
    Avalia um conjunto de hiperparâmetros:
    hp[0] -> número de neurónios
    hp[1] -> learning rate
    Retorna o valor negativo da accuracy de validação
    """

    # Limpeza de memória
    tf.keras.backend.clear_session()
    gc.collect()

    n_neurons = int(hp[0])
    lr = float(hp[1])

    # Criar modelo com estes hiperparâmetros
    model = criar_modelo(n_neurons, lr)

    # Treino rápido (1 época, subconjunto fixo)
    model.fit(
        X_train[idx_t_fixos],
        y_train[idx_t_fixos],
        epochs=1,
        batch_size=32,
        verbose=0
    )

    # Avaliação no subconjunto fixo de validação
    _, val_acc = model.evaluate(
        X_val[idx_v_fixos],
        y_val[idx_v_fixos],
        verbose=0
    )

    # Retornar negativo (os algoritmos minimizam)
    return -val_acc


# =========================================================
# 6. LIMITES DOS HIPERPARÂMETROS
# =========================================================
lb = [32, 0.0001]   # limites inferiores
ub = [512, 0.01]    # limites superiores


# =========================================================
# 7. OTIMIZAÇÃO COM PSO
# =========================================================
print("\n=== A correr PSO ===")

pso_res = pso(6, fitness, lb, ub, 2, 6)
best_pso = pso_res.get_Gbest()
best_pso_score = -fitness(best_pso)


# =========================================================
# 8. OTIMIZAÇÃO COM WSA
# =========================================================
print("\n=== A correr WSA ===")

wsa_res = wsa(4, fitness, lb, ub, 2, 4)
best_wsa = wsa_res.get_Gbest()
best_wsa_score = -fitness(best_wsa)


# =========================================================
# 9. RANDOM SEARCH (BASELINE)
# =========================================================
print("\n=== A correr Random Search ===")

rs_results = []
N_TRIALS = 10

for _ in range(N_TRIALS):

    tf.keras.backend.clear_session()
    gc.collect()

    n_neurons = random.randint(32, 512)

    # Learning rate em escala logarítmica
    lr = 10 ** random.uniform(np.log10(1e-4), np.log10(1e-2))

    model = criar_modelo(n_neurons, lr)

    model.fit(
        X_train[idx_t_fixos],
        y_train[idx_t_fixos],
        epochs=1,
        batch_size=32,
        verbose=0
    )

    _, val_acc = model.evaluate(
        X_val[idx_v_fixos],
        y_val[idx_v_fixos],
        verbose=0
    )

    rs_results.append([n_neurons, lr, val_acc])

# Melhor configuração do Random Search
rs_results = np.array(rs_results)
best_rs = rs_results[np.argmax(rs_results[:, 2])]

best_rs_neurons = int(best_rs[0])
best_rs_lr = float(best_rs[1])
best_rs_score = float(best_rs[2])


# =========================================================
# 10. REGISTO DOS RESULTADOS EM EXCEL
# =========================================================
df = pd.DataFrame({
    "Algoritmo": ["PSO", "WSA", "Random Search"],
    "Neuronios": [
        int(best_pso[0]),
        int(best_wsa[0]),
        best_rs_neurons
    ],
    "LearningRate": [
        best_pso[1],
        best_wsa[1],
        best_rs_lr
    ],
    "Val_Accuracy": [
        best_pso_score,
        best_wsa_score,
        best_rs_score
    ]
})

df.to_excel("Fase3_Resultados_Otimizacao.xlsx", index=False)


# =========================================================
# 11. SELEÇÃO AUTOMÁTICA DO MELHOR ALGORITMO
# =========================================================
scores = {
    "PSO": best_pso_score,
    "WSA": best_wsa_score,
    "Random Search": best_rs_score
}

best_algo = max(scores, key=scores.get)

if best_algo == "PSO":
    best_neurons, best_lr = int(best_pso[0]), float(best_pso[1])
elif best_algo == "WSA":
    best_neurons, best_lr = int(best_wsa[0]), float(best_wsa[1])
else:
    best_neurons, best_lr = best_rs_neurons, best_rs_lr

print(f"\nMelhor algoritmo: {best_algo}")
print(f"Neurónios: {best_neurons} | Learning Rate: {best_lr}")


# =========================================================
# 12. TREINO FINAL DO MODELO
# =========================================================
final_model = criar_modelo(best_neurons, best_lr)

final_model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val)
)

final_model.save("modelo_final_fase3.keras")


# =========================================================
# 13. AVALIAÇÃO NO CONJUNTO DE TESTE
# =========================================================
y_pred = np.argmax(final_model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.title("Matriz de Confusão - Teste")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
plt.show()

print(classification_report(y_test, y_pred, target_names=classes))


# =========================================================
# 14. ANÁLISE DE SENSIBILIDADE AO TAMANHO DO TREINO
# =========================================================
percentagens = [0.25, 0.5, 0.75, 1.0]
sens = []

for p in percentagens:
    n = int(len(X_train) * p)
    idx = np.random.choice(len(X_train), size=n, replace=False)

    model_s = criar_modelo(best_neurons, best_lr)

    model_s.fit(
        X_train[idx],
        y_train[idx],
        epochs=5,
        batch_size=32,
        verbose=0
    )

    _, acc = model_s.evaluate(X_test, y_test, verbose=0)
    sens.append([p, acc])

pd.DataFrame(
    sens,
    columns=["Percentagem_Treino", "Accuracy_Teste"]
).to_excel(
    r"C:\Users\joaop\Desktop\ISEC\IC\trabalho3\Fase3_Sensibilidade_Tamanho.xlsx",
    index=False
)
