# 📘 Classificação de Condições Climáticas com Deep Learning
EfficientNetB0 + Transfer Learning + Fine-Tuning

Este repositório contém um projeto completo de classificação de imagens meteorológicas, utilizando Deep Learning com Transfer Learning e fine-tuning parcial. O objetivo é identificar automaticamente condições climáticas em imagens externas, classificando-as em:

☁️ Cloudy (Nublado)

🌞 Sunny (Ensolarado)

🌧️ Rain (Chuvoso)

🌅 Sunrise (Nascer do Sol)

O projeto foi desenvolvido em Python utilizando TensorFlow/Keras, com foco em execução no Google Colab.

🧠 1. Introdução

A classificação automática de condições meteorológicas a partir de imagens é extremamente útil em:

- Monitoramento de tráfego
- Sistemas de planejamento urbano
- Previsão meteorológica assistida
- Automação industrial
- Veículos autônomos

Para realizar a tarefa, empregamos:

- Transfer Learning com EfficientNetB0
- Data augmentation para ampliar robustez
- Treinamento híbrido (feature extraction → fine-tuning)
- Métricas profissionais (train/val/test)
- A arquitetura EfficientNetB0 foi escolhida por fornecer o melhor equilíbrio entre:
- Qualidade de representação visual
- Velocidade de inferência
- Risco reduzido de overfitting
- Baixa complexidade computacional

🎯 2. Objetivo

Construir um modelo capaz de classificar imagens em quatro categorias climáticas utilizando:
- TensorFlow / Keras
- Transfer Learning
- Pipeline de dados otimizado
- Treinamento em duas fases
- Fine-tuning

📦 3. Dataset

📁 Nome: Multi-class Weather Dataset

🔗 Download: (via Google Drive)

https://drive.google.com/file/d/10eg72mzwrhK0b5RDEqBg1XgOVWwZ8WTA/view

🏷️ Classes: Cloudy, Rain, Sunny, Sunrise

📸 Tamanho: ~1100 imagens

🗂️ Estrutura dos diretórios:

Multi-class Weather Dataset/
 ├── Cloudy/
 ├── Rain/
 ├── Sunny/
 └── Sunrise/

⚙️ 4. Instalação e Execução
▶️ Execução no Google Colab (recomendado)

Abra o notebook.

Ative GPU em: Runtime → Change runtime type → GPU.

Execute as células na ordem.

💻 Execução local

pip install tensorflow numpy matplotlib seaborn scikit-learn gdown

Baixe o dataset manualmente e ajuste os caminhos, se necessário.

📥 5. Download e Extração Automática do Dataset

!pip install gdown

!gdown --id 10eg72mzwrhK0b5RDEqBg1XgOVWwZ8WTA -O weather.zip

import zipfile, os

zip_path = "weather.zip"

extract_path = "weather_dataset"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:

    zip_ref.extractall(extract_path)

print("Extração concluída!")

🧭 6. Carregamento do Dataset (Treino, Validação e Teste)

Divisão utilizada:

70% → Treino

20% → Validação

10% → Teste

import tensorflow as tf

import os

base_dir = "/content/weather_dataset/Multi-class Weather Dataset"

batch_size = 32

img_size = (224, 224)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir,
    validation_split=0.30,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir,
    validation_split=0.30,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

val_size = 0.66   # 20% val + 10% test

val_ds = temp_ds.take(int(len(temp_ds) * val_size))

test_ds = temp_ds.skip(int(len(temp_ds) * val_size))

🚀 7. Otimização do Pipeline

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

test_ds = test_ds.prefetch(AUTOTUNE)

🔄 8. Data Augmentation
from tensorflow.keras import layers, Sequential

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

🧩 9. Modelo: EfficientNetB0 + Cabeçote

from tensorflow import keras

base_model = keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=img_size + (3,),
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=img_size + (3,))

x = data_augmentation(inputs)

x = keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(4, activation="softmax")(x)

model = keras.Model(inputs, outputs)

🏋️‍♂️ 10. Treinamento – Fase 1 (Feature Extraction)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)

🔧 11. Fine-Tuning (Fase 2)

Apenas as últimas camadas da EfficientNet são destravadas.

base_model.trainable = True

for layer in base_model.layers[:150]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)

📊 12. Avaliação Final no Conjunto de Teste

test_loss, test_acc = model.evaluate(test_ds)

print("Acurácia no conjunto de teste:", test_acc)


Para métricas detalhadas:

from sklearn.metrics import confusion_matrix, classification_report

import numpy as np

y_true = []

y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))

🧾 13. Conclusão

A estratégia adotada se mostrou altamente eficaz, pois:

✔ Transfer Learning reduz os requisitos de dados

✔ EfficientNetB0 extrai padrões visuais sofisticados

✔ Fine-tuning permite especializar o modelo no domínio meteorológico

✔ Data augmentation reduz overfitting

✔ Divisão 70/20/10 garante avaliação confiável

Resultado: modelo leve, rápido e com excelente acurácia, ideal para aplicações reais.

📚 14. Referências

- Modelos e Deep Learning

- Chollet, F. Deep Learning with Python. Manning, 2017.

- TensorFlow. Transfer Learning & Fine-Tuning Documentation.

- Krizhevsky, A. et al. “ImageNet Classification with Deep CNNs”. NIPS, 2012.

- Sandler, M. et al. “MobileNetV2”. Google Research, 2018.

- Suporte com IA (prompts utilizados)

- Comparação técnica entre arquiteturas (MobileNetV2, ResNet50, EfficientNetB0) para condições climáticas.

- Geração de código para carregar dataset zipado via Google Drive.

- Código inicial de análise e pipeline de classificação gerado via IA.

📌 15. Possíveis Extensões

- Exportação do modelo (model.save("weather_classifier.keras"))
- API para inferência (Flask/FastAPI)
- Dashboard visual
- Testes com EfficientNetB1–B3
- Early stopping e checkpoints
- Expansão para 10+ classes climáticas
