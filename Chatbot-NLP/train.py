import pickle
import numpy as np
from gensim.models import Word2Vec

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

questions = []
answers = []

with open("dialogs.txt", encoding="utf-8") as f:
    for line in f:
        if "\t" in line:
            q, a = line.strip().split("\t")
            questions.append(q.lower())
            answers.append(a.lower())

print("Dataset Loaded:", len(questions))

sentences = [q.split() for q in questions + answers]

w2v = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

w2v.save("word2vec.model")

print("Word2Vec Saved")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)

X = tokenizer.texts_to_sequences(questions)

maxlen = max(len(x) for x in X)

X = pad_sequences(X, maxlen=maxlen)

vocab_size = len(tokenizer.word_index) + 1

y = np.arange(len(X))

model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=maxlen
    )
)

model.add(LSTM(128))

model.add(Dense(len(X), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(
    X,
    y,
    epochs=20,
    batch_size=32
)

model.save("chatbot_model.keras")

with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

with open("responses.pkl","wb") as f:
    pickle.dump(answers,f)

print("Training Finished")