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



dataset_dir = r"C:\Users\joaop\Desktop\ISEC\IC\trabalho3\dataset"
img_size = (224, 224)

X, y = [], []
classes = sorted([d for d in os.listdir(dataset_dir)
                  if os.path.isdir(os.path.join(dataset_dir, d))])

files_per_class = {}
counts = []

for c in classes:
    files = [f for f in os.listdir(os.path.join(dataset_dir, c))
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    files_per_class[c] = files
    counts.append(len(files))

min_samples = min(counts)
print(f"Balanceamento: {min_samples} imagens por classe")

for idx, c in enumerate(classes):
    files = files_per_class[c].copy()
    np.random.shuffle(files)
    files = files[:min_samples]

    for f in files:
        path = os.path.join(dataset_dir, c, f)
        img = Image.open(path).convert("RGB")
        img = img.resize(img_size)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        X.append(arr)
        y.append(idx)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=0
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=0
)


def criar_modelo(n_neurons, lr):

    base_model = applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(int(n_neurons), activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(int(n_neurons /2 ), activayation = "relu"),
        
        layers.Dense(len(classes), activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=float(lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


idx_t_fixos = np.random.choice(len(X_train), size=min(50, len(X_train)), replace=False)
idx_v_fixos = np.random.choice(len(X_val), size=min(25, len(X_val)), replace=False)

def fitness(hp):

    tf.keras.backend.clear_session()
    gc.collect()

    n_neurons = int(hp[0])
    lr = float(hp[1])

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

    return -val_acc



lb = [32, 0.0001]
ub = [512, 0.01]


print("\n=== A correr PSO ===")
pso_res = pso(6, fitness, lb, ub, 2, 6)
best_pso = pso_res.get_Gbest()
best_pso_score = -fitness(best_pso)



print("\n=== A correr WSA ===")
wsa_res = wsa(4, fitness, lb, ub, 2, 4)
best_wsa = wsa_res.get_Gbest()
best_wsa_score = -fitness(best_wsa)


print("\n=== A correr Random Search ===")

rs_results = []
N_TRIALS = 10

for i in range(N_TRIALS):

    tf.keras.backend.clear_session()
    gc.collect()

    n_neurons = random.randint(32, 512)
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

rs_results = np.array(rs_results)
best_rs_idx = np.argmax(rs_results[:, 2])
best_rs = rs_results[best_rs_idx]

best_rs_neurons = int(best_rs[0])
best_rs_lr = float(best_rs[1])
best_rs_score = float(best_rs[2])


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


scores = {
    "PSO": best_pso_score,
    "WSA": best_wsa_score,
    "Random Search": best_rs_score
}

best_algo = max(scores, key=scores.get)

if best_algo == "PSO":
    best_neurons = int(best_pso[0])
    best_lr = float(best_pso[1])
elif best_algo == "WSA":
    best_neurons = int(best_wsa[0])
    best_lr = float(best_wsa[1])
else:
    best_neurons = best_rs_neurons
    best_lr = best_rs_lr

print(f"\nMelhor algoritmo selecionado automaticamente: {best_algo}")
print(f"Neurónios: {best_neurons} | Learning Rate: {best_lr}")

final_model = criar_modelo(best_neurons, best_lr)

final_model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val)
)

final_model.save("modelo_final_fase3.keras")


y_pred = np.argmax(final_model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm)
plt.title("Matriz de Confusão - Teste")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
plt.show()

print(classification_report(y_test, y_pred, target_names=classes))


percentagens = [0.25, 0.5, 0.75, 1.0]
sens = []

for p in percentagens:
    n = int(len(X_train) * p)
    idx = np.random.choice(len(X_train), size=n, replace=False)

    model_s = criar_modelo(best_neurons, best_lr)
    model_s.fit(X_train[idx], y_train[idx], epochs=5, batch_size=32, verbose=0)

    _, acc = model_s.evaluate(X_test, y_test, verbose=0)
    sens.append([p, acc])

pd.DataFrame(sens, columns=["Percentagem_Treino", "Accuracy_Teste"]) \
  .to_excel( r"C:\Users\joaop\Desktop\ISEC\IC\trabalho3\Fase3_Sensibilidade_Tamanho.xlsx",
    index=False)


