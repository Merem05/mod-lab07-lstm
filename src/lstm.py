import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
import random
import sys
import os
from google.colab import files

print("Генератор текста")


print("\nЗагрузка файла \n")
uploaded = files.upload()

for filename in uploaded.keys():
    with open(filename, 'rb') as f:
        text = f.read().decode('utf-8', errors='replace')
    print(f"\nЗагружен файл: {filename}")
    break

print(f"Размер текста: {len(text)} символов")

vocabulary = sorted(list(set(text)))
print(f"Размер алфавита: {len(vocabulary)} символов")
print(f"Алфавит: {''.join(vocabulary)}\n")

char_to_indices = {c: i for i, c in enumerate(vocabulary)}
indices_to_char = {i: c for i, c in enumerate(vocabulary)}

max_length = 40
step = 1

sentences = []
next_chars = []

for i in range(0, len(text) - max_length, step):
    sentences.append(text[i:i + max_length])
    next_chars.append(text[i + max_length])

print(f"Всего цепочек: {len(sentences)}")

X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}\n")

print("Построение сети\n")
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print("Модель создана:")
model.summary()



def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
print("Функция температуры создана\n")


print("Начинаем обучение\n")

model.fit(X, y, batch_size=128, epochs=50, verbose=1)

print("\nОбучение завершено!\n")



def generate_text(length, diversity):
    start_index = random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index:start_index + max_length]
    generated += sentence

    print(f"Начальная цепочка: {sentence}\n")

    for i in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_indices[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_char = indices_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        if (i + 1) % 100 == 0:
            print(f"Сгенерировано {i + 1} символов")

    return generated

print("Функция генерации текста создана\n")


print("Генерация текста\n")

generated_text = generate_text(1500, 0.2)

print("\nСгенерированный текст\n")
print(generated_text)

os.makedirs('result', exist_ok=True)

with open('result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)

print(f"Сохранено символов: {len(generated_text)}")
print("Текст сохранён в result/gen.txt")

files.download('result/gen.txt')
