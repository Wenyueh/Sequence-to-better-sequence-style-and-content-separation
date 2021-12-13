import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

# Create datasets:
vocab_size = 10
max_seq_length = 20
length_range = (10, max_seq_length)
n_train = 10000
n_val = 1000
n_test = 1000


def compute_probability(seq, PAD_ID, data_style, alphabet, vocab_size):
    prob = 1
    if PAD_ID in seq:
        seq = seq[: seq.index(PAD_ID)]
    seq_length = len(seq)
    ninetyfive_percent_const = 19.0  # use 19 for 95 percent
    fifty_percent_const = 1.0  # use 1 for 5 percent
    if data_style == "general" or data_style == "locality":
        for t in range(1, seq_length):  # draw next character
            probs = [1.0] * vocab_size
            if (t >= 1) and (seq[t - 1] == "A"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 3) and (seq[t - 3] == "D"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 5) and (seq[t - 5] == "E"):
                probs[alphabet.index("E")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 2) and (seq[t - 2] == "H") and (seq[t - 1] == "I"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 2) and (seq[t - 2] == "I") and (seq[t - 1] == "H"):
                probs[alphabet.index("I")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 3) and (seq[t - 3] == "B") and (seq[t - 2] == "C"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 11) and (seq[t - 1] == "F"):
                probs[alphabet.index("J")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t == 7) and (seq[t - 1] == "F"):
                probs[alphabet.index("G")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t == 8) and (seq[t - 1] == "F"):
                probs[alphabet.index("G")] = fifty_percent_const * (vocab_size - 1)
            if (t == 5) or (t == 10) or (t == 15) or (t == 20):
                probs[alphabet.index("C")] = fifty_percent_const * (vocab_size - 1)
            if (t >= 1) and (seq[t - 1] == "C"):
                probs[alphabet.index("B")] = fifty_percent_const * (vocab_size - 1)
            probs = [x / sum(probs) for x in probs]
            prob *= probs[seq[t]]
        prob = -1 * np.log(prob)
        return prob
    elif data_style == "comparison":
        for t in range(1, seq_length):  # draw next character
            probs = [1.0] * vocab_size
            if (t >= 1) and (seq[t - 1] == "A"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 3) and (seq[t - 3] == "D"):
                probs[alphabet.index("E")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 5) and (seq[t - 5] == "E"):
                probs[alphabet.index("F")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 2) and (seq[t - 2] == "H") and (seq[t - 1] == "I"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 2) and (seq[t - 2] == "I") and (seq[t - 1] == "H"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 3) and (seq[t - 3] == "B") and (seq[t - 2] == "C"):
                probs[alphabet.index("E")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 11) and (seq[t - 1] == "F"):
                probs[alphabet.index("F")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t == 7) and (seq[t - 1] == "F"):
                probs[alphabet.index("F")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t == 8) and (seq[t - 1] == "F"):
                probs[alphabet.index("F")] = fifty_percent_const * (vocab_size - 1)
            if (t == 5) or (t == 10) or (t == 15) or (t == 20):
                probs[alphabet.index("C")] = fifty_percent_const * (vocab_size - 1)
            if (t >= 1) and (seq[t - 1] == "C"):
                probs[alphabet.index("B")] = fifty_percent_const * (vocab_size - 1)
            probs = [x / sum(probs) for x in probs]
            prob *= probs[seq[t]]
        prob = -1 * np.log(prob)
        return prob


def generateSimulationSeq(data_style, alphabet, seq_length, vocab_size):
    seq = ["Z"]
    ninetyfive_percent_const = 19.0  # use 19 for 95 percent
    fifty_percent_const = 1.0  # use 1 for 5 percent
    if data_style == "general" or data_style == "locality":
        for t in range(1, seq_length + 1):  # draw next character
            probs = [1.0] * vocab_size
            if (t >= 1) and (seq[t - 1] == "A"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 3) and (seq[t - 3] == "D"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 5) and (seq[t - 5] == "E"):
                probs[alphabet.index("E")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 2) and (seq[t - 2] == "H") and (seq[t - 1] == "I"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 2) and (seq[t - 2] == "I") and (seq[t - 1] == "H"):
                probs[alphabet.index("I")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 3) and (seq[t - 3] == "B") and (seq[t - 2] == "C"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 11) and (seq[t - 1] == "F"):
                probs[alphabet.index("J")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t == 7) and (seq[t - 1] == "F"):
                probs[alphabet.index("G")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t == 8) and (seq[t - 1] == "F"):
                probs[alphabet.index("G")] = fifty_percent_const * (vocab_size - 1)
            if (t == 5) or (t == 10) or (t == 15) or (t == 20):
                probs[alphabet.index("C")] = fifty_percent_const * (vocab_size - 1)
            if (t >= 1) and (seq[t - 1] == "C"):
                probs[alphabet.index("B")] = fifty_percent_const * (vocab_size - 1)
            next_char = random.choices(
                list(alphabet[:vocab_size]),
                weights=[x / sum(probs) for x in probs],
                k=1,
            )[0]
            seq.append(next_char)
        return seq[1:]
    elif data_style == "comparison":
        for t in range(1, seq_length + 1):  # draw next character
            probs = [1.0] * vocab_size
            if (t >= 1) and (seq[t - 1] == "A"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 3) and (seq[t - 3] == "D"):
                probs[alphabet.index("E")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 5) and (seq[t - 5] == "E"):
                probs[alphabet.index("F")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 2) and (seq[t - 2] == "H") and (seq[t - 1] == "I"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 2) and (seq[t - 2] == "I") and (seq[t - 1] == "H"):
                probs[alphabet.index("A")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 3) and (seq[t - 3] == "B") and (seq[t - 2] == "C"):
                probs[alphabet.index("E")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t >= 11) and (seq[t - 1] == "F"):
                probs[alphabet.index("F")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t == 7) and (seq[t - 1] == "F"):
                probs[alphabet.index("F")] = ninetyfive_percent_const * (vocab_size - 1)
            if (t == 8) and (seq[t - 1] == "F"):
                probs[alphabet.index("F")] = fifty_percent_const * (vocab_size - 1)
            if (t == 5) or (t == 10) or (t == 15) or (t == 20):
                probs[alphabet.index("C")] = fifty_percent_const * (vocab_size - 1)
            if (t >= 1) and (seq[t - 1] == "C"):
                probs[alphabet.index("B")] = fifty_percent_const * (vocab_size - 1)
            next_char = random.choices(
                list(alphabet[:vocab_size]),
                weights=[x / sum(probs) for x in probs],
                k=1,
            )[0]
            seq.append(next_char)
        return seq[1:]


def getSimulationOutcome(seq, data_style, max_seq_length=20.0):
    if data_style == "general":
        return seq.count("A") / max_seq_length
    if data_style == "locality":
        return seq[:10].count("A") / max_seq_length
    if data_style == "comparison":
        if seq.count("F") != 0:
            if seq.count("A") < seq.count("F"):
                score = seq.count("A") / seq.count("F")
            else:
                score = 1.0
        else:
            if seq.count("A") == 0:
                score = 0.0
            else:
                score = 1.0
        return score


def generateSimulationData(data_style, alphabet, seq_num, seq_length_range, vocab_size):
    seqs = []
    outcomes = []
    probs = []
    for _ in range(seq_num):
        seq_length = random.randint(seq_length_range[0], seq_length_range[1])
        sequence = generateSimulationSeq(data_style, alphabet, seq_length, vocab_size)
        outcome = getSimulationOutcome(sequence, data_style, max_seq_length=seq_length)
        prob = compute_probability(
            [alphabet.index(x) for x in sequence], 10, data_style, alphabet, vocab_size
        )
        seqs.append(sequence)
        outcomes.append(outcome)
        probs.append(prob)
    print("the standard deviation of outcome is {}".format(np.std(outcomes)))
    print("the mean of outcome is {}".format(np.mean(outcomes)))
    print("this median of outcome is {}".format(np.median(outcomes)))
    print("***")
    print("the standard deviation of sequence probability is {}".format(np.std(probs)))
    print("the mean of sequence probability is {}".format(np.mean(probs)))
    print("this median of sequence probability is {}".format(np.median(probs)))

    return (seqs, outcomes)


class SimulationData(Dataset):
    def __init__(self, data, vocab, max_length):
        self.data = data
        self.vocab = vocab
        self.PAD_ID = len(self.vocab)
        self.max_length = max_length

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        seq = self.data[0][index]
        outcome = self.data[1][index]
        for char in seq:
            assert char in self.vocab
        seq_index = [self.vocab.index(char) for char in seq]
        if len(seq_index) < self.max_length:
            seq_index += [self.PAD_ID] * (self.max_length - len(seq_index))
        return torch.tensor(seq_index), torch.tensor(outcome)


def compute_simulation_dataloader(
    batch_size,
    data_style,
    n_train=10000,
    n_val=1000,
    n_test=1000,
    seq_length_range=(10, 20),
    vocab_size=10,
):
    alphabet = "ABCDEFGHIJKLMNOZQRSTUVWXYZ"[:vocab_size]
    train_data = generateSimulationData(
        data_style, alphabet, n_train, seq_length_range, vocab_size
    )
    val_data = generateSimulationData(
        data_style, alphabet, n_val, seq_length_range, vocab_size
    )
    test_data = generateSimulationData(
        data_style, alphabet, n_test, seq_length_range, vocab_size
    )

    train_dataset = SimulationData(train_data, alphabet, seq_length_range[1])
    val_dataset = SimulationData(val_data, alphabet, seq_length_range[1])
    test_dataset = SimulationData(test_data, alphabet, seq_length_range[1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Create datasets:
    vocab_size = 10
    max_seq_length = 20
    length_range = (10, max_seq_length)
    n_train = 10
    n_val = 10
    n_test = 10
    data_style = "comparison"

    train_loader, val_loader, test_loader = compute_simulation_dataloader(
        1,
        data_style,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seq_length_range=(10, 20),
        vocab_size=10,
    )

    alphabet = "ABCDEFGHIJKLMNOZQRSTUVWXYZ"
    PAD_ID = 10

    X = generateSimulationData(data_style, alphabet, 100000, (10, 20), vocab_size)

    """
    for t in train_loader:
        seq = t[0][0].tolist()
        prob = compute_probability(seq, PAD_ID, data_style, alphabet, vocab_size)
        print(prob)
    """
