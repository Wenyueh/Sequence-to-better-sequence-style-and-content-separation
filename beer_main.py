import torch
from beer_data import compute_beer_data, Collator, word_embedding
import argparse
import torch
from VAE_beer import VAE, kl_anneal_function
import os
import sys
import numpy as np


def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def load_model(args, vocab, glove, model_dir):
    checkpoint = torch.load(model_dir)

    VAE_model = VAE(
        glove,
        vocab,
        args.rnn_type,
        args.pred_type,
        args.embed_dim,
        args.memory_dim,
        args.latent_dim,
        args.score_controller_dim,
        args.max_seq_length,
        args.experiment_type,
    ).cuda()

    VAE_model.load_state_dict(checkpoint)

    return VAE_model


class Logger:
    def __init__(self, path):
        self.log_path = path

    def log(self, string, newline=True):
        with open(self.log_path, "a") as f:
            f.write(string)
            if newline:
                f.write("\n")

        sys.stdout.write(string)
        if newline:
            sys.stdout.write("\n")
        sys.stdout.flush()


def compute_evaluation(VAE_model):
    all_reconstruction_errors = []
    all_outcomes_errors = []
    for t in train_loader:
        input_seq = t[0].cuda()
        gold_outcome = t[1].cuda()
        z, sigma = VAE_model.encode(input_seq)
        reconstructed_probs = VAE_model.decode(input_seq, z, sigma)
        reconstructed_seqs = torch.argmax(reconstructed_probs, dim=-1)
        predicted_outcomes = VAE_model.outcome_prediction(z, use_sigmoid).squeeze()
        # print(input_seq)
        # print(reconstructed_seqs)
        # break
        # print("this is the pred outcome")
        # print(predicted_outcomes)
        # print("this is the gold outcome")
        # print(gold_outcome)
        reconstruction_errors = 1 - torch.eq(input_seq, reconstructed_seqs).sum() / (
            input_seq.size(0) * input_seq.size(1)
        )
        outcomes_errors = np.mean(
            np.abs(
                gold_outcome.cpu().detach().numpy()
                - predicted_outcomes.cpu().detach().numpy()
            )
        )

        all_reconstruction_errors.append(reconstruction_errors)
        all_outcomes_errors.append(outcomes_errors)

    averaged_reconstruction_errors = torch.mean(torch.tensor(all_reconstruction_errors))
    averaged_outcomes_errors = np.mean(all_outcomes_errors)

    logger.log(
        "this is the train average reconstruction errors:{}".format(
            averaged_reconstruction_errors
        )
    )
    logger.log(
        "this is the train average outcome errors:{}".format(averaged_outcomes_errors)
    )

    all_reconstruction_errors = []
    all_outcomes_errors = []
    for t in test_loader:
        input_seq = t[0].cuda()
        gold_outcome = t[1].cuda()
        z, sigma = VAE_model.encode(input_seq)
        reconstructed_probs = VAE_model.decode(input_seq, z, sigma)
        reconstructed_seqs = torch.argmax(reconstructed_probs, dim=-1)
        predicted_outcomes = VAE_model.outcome_prediction(z, use_sigmoid).squeeze()
        # print(input_seq)
        # print(reconstructed_seqs)
        # break
        reconstruction_errors = 1 - torch.eq(input_seq, reconstructed_seqs).sum() / (
            input_seq.size(0) * input_seq.size(1)
        )
        outcomes_errors = np.mean(
            np.abs(
                gold_outcome.cpu().detach().numpy()
                - predicted_outcomes.cpu().detach().numpy()
            )
        )

        all_reconstruction_errors.append(reconstruction_errors)
        all_outcomes_errors.append(outcomes_errors)

    averaged_reconstruction_errors = torch.mean(torch.tensor(all_reconstruction_errors))
    averaged_outcomes_errors = np.mean(all_outcomes_errors)

    logger.log(
        "this is the test average reconstruction errors:{}".format(
            averaged_reconstruction_errors
        )
    )
    logger.log(
        "this is the test average outcome errors:{}".format(averaged_outcomes_errors)
    )

    return averaged_reconstruction_errors, averaged_outcomes_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--glove_dir", type=str, default="data/glove/")
    parser.add_argument("--review_type", type=str, default="appearance")
    parser.add_argument("--toy", type=bool, default=False)

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=70)

    # model
    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument("--pred_type", type=str, default="non_linear")
    parser.add_argument("--embed_dim", type=int, default=50)
    parser.add_argument("--memory_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--score_controller_dim", type=int, default=64)
    parser.add_argument("--experiment_type", type=str, default="vector_split")

    # training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_epochs", type=int, default=30)
    parser.add_argument("--kl_epochs", type=int, default=10)
    parser.add_argument("--inv_epochs", type=int, default=10)

    # logging
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--gpus", type=str, default="3")

    args = parser.parse_args()
    set_seed(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    logger = Logger(
        "beer" + args.rnn_type + "_{}_prenorm.log".format(args.experiment_type)
    )
    logger.log(str(args))

    glove, vocab, _ = word_embedding(args.glove_dir)

    collator = Collator(vocab)

    train_loader, test_loader = compute_beer_data(
        args.data_dir,
        args.glove_dir,
        args.review_type,
        args.batch_size,
        collator,
        args.toy,
    )

    VAE_model = VAE(
        glove,
        vocab,
        args.rnn_type,
        args.pred_type,
        args.embed_dim,
        args.memory_dim,
        args.latent_dim,
        args.score_controller_dim,
        args.max_seq_length,
        args.experiment_type,
    ).cuda()

    # optimizer
    optimizer = torch.optim.AdamW(VAE_model.parameters(), lr=args.lr)

    # train
    # train reconstruction & mse
    seq2seq_importance = 1
    mse_importance = 1
    kl_importance = 0
    invar_importance = 0
    use_sigmoid = True
    loss_val = 0
    step = 0
    best_result = float("inf")
    VAE_model.zero_grad()
    for e in range(args.seq_epochs):
        for t in train_loader:
            input_seq = torch.tensor(t[0]).cuda()
            gold_outcome = torch.tensor(t[1]).cuda()
            loss = VAE_model.compute_loss(
                input_seq,
                gold_outcome,
                use_sigmoid,
                seq2seq_importance,
                mse_importance,
                kl_importance,
                invar_importance,
            )
            loss.backward()
            step += 1
            loss_val += loss.item()
            optimizer.step()
            VAE_model.zero_grad()

            if step % args.logging_step == 0:
                logger.log(str(loss_val))
                loss_val = 0

        (
            averaged_reconstruction_errors,
            averaged_outcomes_errors,
        ) = compute_evaluation(VAE_model)
        if averaged_reconstruction_errors + averaged_outcomes_errors < best_result:
            best_result = averaged_reconstruction_errors + averaged_outcomes_errors
            torch.save(
                VAE_model.state_dict(),
                "beer_VAE_{}_prenorm.pt".format(args.experiment_type),
            )
    """

    # train prior using logistic kl_importance
    VAE_model = load_model(args, "VAE_locality_vector_pre_norm.pt")
    # VAE_model = load_model(args, "VAE_kl_new.pt")
    optimizer = torch.optim.AdamW(VAE_model.parameters(), lr=args.lr)
    VAE_model.zero_grad()
    seq2seq_importance = 0.95
    mse_importance = 0.05
    invar_importance = 0
    use_sigmoid = True
    step = 0
    loss_val = 0
    for annealing_step in range(args.kl_epochs):
        kl_importance = kl_anneal_function("logistic", annealing_step * 100)
        logger.log("the KL importance is {}".format(kl_importance))
        best_loss = float("inf")
        for e in range(100):
            for t in train_loader:
                input_seq = t[0].cuda()
                gold_outcome = t[1].cuda()
                # compute loss
                loss = VAE_model.compute_loss(
                    input_seq,
                    gold_outcome,
                    use_sigmoid,
                    seq2seq_importance,
                    mse_importance,
                    kl_importance,
                    invar_importance,
                )
                loss.backward()
                step += 1
                loss_val += loss.item()
                optimizer.step()
                VAE_model.zero_grad()

                if step % args.logging_step == 0:
                    logger.log(str(loss_val))
                    if loss_val < best_loss:
                        torch.save(
                            VAE_model.state_dict(),
                            "VAE_{}_{}_kl.pt".format(args.data_style, args.experiment_type),
                        )
                        best_loss = loss_val
                    loss_val = 0


    # train invariance
    VAE_model = load_model(args, "VAE_kl_new.pt")
    optimizer = torch.optim.AdamW(VAE_model.parameters(), lr=args.lr)
    VAE_model.zero_grad()
    seq2seq_importance = 1
    mse_importance = 1
    kl_importance = 1
    use_sigmoid = True
    step = 0
    loss_val = 0
    for invar_step in range(1, args.inv_epochs + 1):
        inv_importance = kl_anneal_function("logistic", invar_step * 100)
        logger.log("the invariance importance is {}".format(inv_importance))
        best_loss = float("inf")
        for e in range(50):
            for t in train_loader:
                input_seq = t[0].cuda()
                gold_outcome = t[1].cuda()
                # compute loss
                loss = VAE_model.compute_loss(
                    input_seq,
                    gold_outcome,
                    use_sigmoid,
                    seq2seq_importance,
                    mse_importance,
                    kl_importance,
                    inv_importance,
                )
                loss.backward()
                step += 1
                loss_val += loss.item()
                optimizer.step()
                VAE_model.zero_grad()

                if step % args.logging_step == 0:
                    logger.log(str(loss_val))
                    if loss_val < best_loss:
                        torch.save(
                            VAE_model.state_dict(), "VAE_perturbed_inv.pt".format(step)
                        )
                        best_loss = loss_val
                    loss_val = 0
    """
