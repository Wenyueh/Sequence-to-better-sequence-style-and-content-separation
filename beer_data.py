import numpy as np
from torch.utils.data import Dataset, DataLoader
import bcolz
import pickle
import string
import os
import json
import torch


def save_glove_embedding(glove_dir):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f"{glove_dir}/6B.50.dat", mode="w")

    with open(f"{glove_dir}/glove.6B.50d.txt", "rb") as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

        words.append("<PAD>")
        word2idx["<PAD>"] = idx
        idx += 1
        vectors.append(np.random.normal(scale=0.1, size=(50,)))

    vocab_size = len(words)
    print("this is the size of vocabulary: {}".format(vocab_size))

    vectors = bcolz.carray(
        vectors[1:].reshape((vocab_size, 50)),
        rootdir=f"{glove_dir}/6B.50.dat",
        mode="w",
    )
    vectors.flush()
    pickle.dump(words, open(f"{glove_dir}/6B.50_words.pkl", "wb"))
    pickle.dump(word2idx, open(f"{glove_dir}/6B.50_idx.pkl", "wb"))


def word_embedding(glove_dir):
    vectors = bcolz.open(f"{glove_dir}/6B.50.dat")[:]
    vocab = pickle.load(open(f"{glove_dir}/6B.50_words.pkl", "rb"))
    word2idx = pickle.load(open(f"{glove_dir}/6B.50_idx.pkl", "rb"))
    idx2word = {v: k for k, v in word2idx.items()}

    glove = {i: vectors[i] for i in list(idx2word.keys())}

    return glove, vocab, word2idx


def filter_appearance_stc(text):
    text = (
        text.replace("!", ".")
        .replace("...", ".")
        .replace("?", ".")
        .replace(";", ".")
        .replace("appearance :", ".appearance :")
        .replace("feel :", ".feel :")
        .replace("drinkability :", ".drinkability :")
        .replace("taste :", ".taste :")
        .replace("smell :", ".smell :")
        .replace(" a :", ". appearance :")
        .replace(" m -", ". m -")
        .replace(" s :", ". smell :")
        .replace(" s -", ". smell -")
    )
    sentences = text.split(".")
    filtered_sentences = []
    keywords = [
        "appear",
        "observ",
        "color",
        "colour",
        "dark",
        "brown",
        "orange",
        "white",
        "sight",
        " red",
        "amber",
        "foam",
        "clean",
        " eye",
        "black",
        "clear ",
        "golden",
    ]
    for s in sentences:
        for k in keywords:
            if k in s:
                if (
                    "smell" not in s
                    and "taste" not in s
                    and "feel" not in s
                    and "aroma" not in s
                ):
                    filtered_sentences.append(s)
                break
    text = ".".join(filtered_sentences)
    return text


# word level split, use GLoVe
def tokenization(data, vocab, word2idx):
    tokenized_data = []
    for d in data:
        contain = True
        score = d[1]
        text = d[0]
        for p in list(string.punctuation):
            text = text.replace(p, " " + p + " ")
        text = text.split()
        if text[:2] in [
            ["appearance", "-"],
            ["appearance", ":"],
            ["a", "-"],
            ["a", ":"],
            ["eye", ":"],
        ]:
            text = text[2:]
        for t in text:
            if t not in vocab:
                contain = False
                break
        if len(text) > 70:
            contain = False
        if contain:
            ids = [word2idx[t] for t in text]
            tokenized_data.append((ids, score))

    return tokenized_data


"""
def levenshtein(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1] / len(s2)


def closest_word(vocab, t):
    closest_length = float("inf")
    best = ""
    for v in vocab:
        if levenshtein(v, t) < closest_length:
            closest_length = levenshtein(v, t)
            best = v
    return best
"""

# appearance, aroma, palate, taste, and overall impression
def load_data(data_dir, glove_dir, review_type, toy):
    with open(data_dir + "train.txt") as f:
        train = f.read()
    with open(data_dir + "test.txt") as f:
        test = f.read()

    with open(data_dir + "train2.txt") as f:
        train += f.read()
    with open(data_dir + "test2.txt") as f:
        test += f.read()

    train = [
        [t.split("\t")[0].split(), t.split("\t")[1]]
        for t in train.split("\n")
        if t != ""
    ]
    test = [
        [t.split("\t")[0].split(), t.split("\t")[1]]
        for t in test.split("\n")
        if t != ""
    ]
    if toy:
        train = train[:100]
        test = test[:100]

    assert review_type in ["appearance", "aroma", "palate", "taste", "impression"]
    if review_type == "appearance":  # std 0.25
        train = [
            [filter_appearance_stc(t[1]), float(t[0][0])]
            for t in train
            if filter_appearance_stc(t[1]) != ""
        ]
        test = [
            [filter_appearance_stc(t[1]), float(t[0][0])]
            for t in test
            if filter_appearance_stc(t[1]) != ""
        ]
    elif review_type == "aroma":  # 0.18
        train = [[t[1], float(t[0][1])] for t in train]
        test = [[t[1], float(t[0][0])] for t in test]
    elif review_type == "palate":  # 0.17
        train = [[t[1], float(t[0][2])] for t in train]
        test = [[t[1], float(t[0][0])] for t in test]
    elif review_type == "taste":  # 0.1936
        train = [[t[1], float(t[0][3])] for t in train]
        test = [[t[1], float(t[0][0])] for t in test]
    elif review_type == "impression":  # 0.1954
        train = [[t[1], float(t[0][4])] for t in train]
        test = [[t[1], float(t[0][0])] for t in test]

    if "6B.50_words.pkl" in os.listdir(glove_dir) and "6B.50_idx.pkl" in os.listdir(
        glove_dir
    ):
        print("embeddings already saved")
    else:
        save_glove_embedding(glove_dir)

    _, vocab, word2idx = word_embedding(glove_dir)

    tokenized_train = tokenization(train, vocab, word2idx)
    tokenized_test = tokenization(test, vocab, word2idx)
    print("the number of training data is {}".format(len(tokenized_train)))
    print(
        "the mean of train outcome score is {}".format(
            np.mean([t[1] for t in tokenized_train])
        )
    )
    print(
        "the median of train outcome score is {}".format(
            np.median([t[1] for t in tokenized_train])
        )
    )
    print(
        "the standard deviation of train outcome score is {}".format(
            np.std([t[1] for t in tokenized_train])
        )
    )
    print(
        "the average length of training data is {}".format(
            np.mean([len(t[0]) for t in tokenized_train])
        )
    )
    print(
        "the max length of training data is {}".format(
            np.max([len(t[0]) for t in tokenized_train])
        )
    )
    print(
        "the min length of training data is {}".format(
            np.min([len(t[0]) for t in tokenized_train])
        )
    )
    print(
        "the median length of training data is {}".format(
            np.median([len(t[0]) for t in tokenized_train])
        )
    )
    print(
        "the standard deviation length of training data is {}".format(
            np.std([len(t[0]) for t in tokenized_train])
        )
    )
    print("*****")
    print("the number of test data is {}".format(len(tokenized_test)))
    print(
        "the mean of test outcome score is {}".format(
            np.mean([t[1] for t in tokenized_test])
        )
    )
    print(
        "the median of test outcome score is {}".format(
            np.median([t[1] for t in tokenized_test])
        )
    )
    print(
        "the standard deviation of test outcome score is {}".format(
            np.std([t[1] for t in tokenized_test])
        )
    )
    print(
        "the average length of test data is {}".format(
            np.mean([len(t[0]) for t in tokenized_test])
        )
    )
    print(
        "the max length of test data is {}".format(
            np.max([len(t[0]) for t in tokenized_test])
        )
    )
    print(
        "the min length of test data is {}".format(
            np.min([len(t[0]) for t in tokenized_test])
        )
    )
    print(
        "the median length of test data is {}".format(
            np.median([len(t[0]) for t in tokenized_test])
        )
    )
    print(
        "the standard deviation length of test data is {}".format(
            np.std([len(t[0]) for t in tokenized_test])
        )
    )

    if toy:
        with open(data_dir + "toy_tokenized_train.json", "w") as f:
            json.dump(tokenized_train, f)
        with open(data_dir + "toy_tokenized_test.json", "w") as f:
            json.dump(tokenized_test, f)
    else:
        with open(data_dir + "full_tokenized_train.json", "w") as f:
            json.dump(tokenized_train, f)
        with open(data_dir + "full_tokenized_test.json", "w") as f:
            json.dump(tokenized_test, f)

    return tokenized_train, tokenized_test


class BeerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]


class Collator:
    def __init__(self, vocab, max_length=512):
        self.max_length = max_length
        self.vocab = vocab

    def __call__(self, batch):
        max_length = min(max([len(d[0]) for d in batch]), self.max_length)
        new_text = []
        new_score = []
        for d in batch:
            text = d[0]
            score = float(np.square(d[1]))
            new_score.append(score)
            if len(text) < max_length:
                text += [len(self.vocab) - 1] * (max_length - len(text))
                new_text.append(text)
            else:
                text = text[:max_length]
                new_text.append(text)

        return torch.tensor(new_text), torch.tensor(new_score)


def compute_beer_data(data_dir, glove_dir, review_type, batch_size, collator, toy):
    if toy:
        if "toy_tokenized_train.json" in os.listdir(
            data_dir
        ) and "toy_tokenized_test.json" in os.listdir(data_dir):
            print("save tokenized data")
            train, test = load_data(data_dir, glove_dir, review_type, toy)
        else:
            print("tokenized data saved already")
            with open(data_dir + "toy_tokenized_train.json", "r") as f:
                train = json.load(f)
            with open(data_dir + "toy_tokenized_test.json", "r") as f:
                test = json.load(f)
    else:
        if "full_tokenized_train.json" in os.listdir(
            data_dir
        ) and "full_tokenized_test.json" in os.listdir(data_dir):
            print("save tokenized data")
            train, test = load_data(data_dir, glove_dir, review_type, toy)
        else:
            print("tokenized data saved already")
            with open(data_dir + "full_tokenized_train.json", "r") as f:
                train = json.load(f)
            with open(data_dir + "full_tokenized_test.json", "r") as f:
                test = json.load(f)
    train_dataset = BeerDataset(train)
    test_dataset = BeerDataset(test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False
    )

    return train_loader, test_loader


def tokenizer_decode(word2idx, text):
    assert isinstance(text, list)
    idx2word = {v: k for k, v in word2idx.items()}
    decoded_text = [idx2word[t] for t in text]
    return decoded_text


if __name__ == "__main__":
    data_dir = "data/"
    glove_dir = "data/glove/"

    review_type = "appearance"
    batch_size = 1
    toy = True

    _, vocab, word2idx = word_embedding(glove_dir)

    collator = Collator(vocab)

    train_loader, test_loader = compute_beer_data(
        data_dir, glove_dir, review_type, batch_size, collator, toy
    )

    for t in test_loader:
        text = t[0][0].tolist()
        score = t[1]
        print(text)
        print(tokenizer_decode(word2idx, text))
        print(score)
        print("*****")

    scores = []
    for t in train_loader:
        scores.append(float(t[1]))

    """
    print(np.mean(np.square(scores)))
    print(np.median(np.square(scores)))
    print(np.std(np.square(scores)))
    print(np.max(np.square(scores)))
    print(np.min(np.square(scores)))


    print(np.mean(scores))
    print(np.median(scores))
    print(np.std(scores))
    print(np.max(scores))
    print(np.min(scores))
    """
