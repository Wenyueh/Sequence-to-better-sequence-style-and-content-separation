import torch
import torch.nn as nn
from simulation_dataset import compute_simulation_dataloader, compute_probability
import numpy as np
import os
import random
import sys


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


def kl_anneal_function(anneal_function, step, k=0.0025, x0=2500):
    if anneal_function == "logistic":
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == "linear":
        return min(1, step / x0)


def levenshtein(seq1, seq2, pad_ID):
    s1 = [value for value in seq1 if value != pad_ID]
    s2 = [value for value in seq2 if value != pad_ID]
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
    return distances[-1]


def seq_difference(s1, s2):
    dis = 0
    for x, y in zip(s1, s2):
        if x != y:
            dis += 1
    return dis


class VAE(nn.Module):
    def __init__(
        self,
        vocab,
        data_style,
        rnn_type,
        pred_type,
        embed_dim,
        hidden_dim,
        latent_dim,
        score_controller_dim,
        max_length,
        experiment_type,
    ):
        super().__init__()
        self.vocab = vocab
        self.data_style = data_style
        self.vocab.append("<PAD>")
        self.PAD_ID = self.vocab.index("<PAD>")
        self.vocab_size = len(vocab)
        self.embedding_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.score_controller_dim = score_controller_dim
        self.max_length = max_length
        self.prediction_type = pred_type
        self.rnn_type = rnn_type
        self.experiment_type = experiment_type

        if self.embedding_dim is None:
            self.embedding_dim = self.vocab_size - 1
        else:
            self.embedding_dim = embed_dim
        # use embedding wrapper to encode the input characters
        # self.embedding_wrapper = Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_wrapper = nn.Embedding(self.vocab_size, self.embedding_dim)
        # embedding_wrapper for transformers ... need a much larger embedding dim
        self.embedding_wrapper_transformers = nn.Embedding(
            self.vocab_size, self.hidden_dim
        )

        ## rnn model
        if self.rnn_type == "RNN":
            self.model = nn.RNN(self.embedding_dim, self.hidden_dim, batch_first=True)
            self.depth = 1
        elif self.rnn_type == "GRU":
            self.model = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
            self.depth = 1
        elif self.rnn_type == "Deep_GRU":
            self.model = nn.GRU(
                self.embedding_dim, self.hidden_dim, 2, batch_first=True
            )
            self.depth = 2
        elif self.rnn_type == "LSTM":
            self.model = nn.LSTM(
                self.embedding_dim, self.hidden_dim, 1, batch_first=True
            )
            self.depth = 1

        ## decoder
        self.weights_pi = nn.Linear(self.hidden_dim, self.vocab_size, bias=True)
        self.softmax_pi = nn.Softmax(dim=-1)

        ## distribution predictor
        if self.experiment_type == "general":
            self.weights_mu = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
            self.weights1_sigma = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.relu_sigma = nn.ReLU()
            self.weights2_sigma = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
            self.latent2hidden = nn.Linear(self.latent_dim, self.hidden_dim, bias=True)
        if self.experiment_type == "vector_split":
            self.weights_style_mu = nn.Linear(
                self.hidden_dim, self.score_controller_dim, bias=True
            )
            self.weights1_style_sigma = nn.Linear(
                self.hidden_dim, self.hidden_dim, bias=True
            )
            self.weights2_style_sigma = nn.Linear(
                self.hidden_dim, self.score_controller_dim, bias=True
            )
            self.weights_content_mu = nn.Linear(
                self.hidden_dim, self.latent_dim - self.score_controller_dim, bias=True
            )
            self.weights1_content_sigma = nn.Linear(
                self.hidden_dim, self.hidden_dim, bias=True
            )
            self.weights2_content_sigma = nn.Linear(
                self.hidden_dim, self.latent_dim - self.score_controller_dim, bias=True
            )
            self.relu_sigma = nn.ReLU()
            self.latent2hidden = nn.Linear(self.latent_dim, self.hidden_dim, bias=True)

        ## outcome predictor
        self.tanh_pred = nn.Tanh()
        if self.experiment_type == "general":
            if self.prediction_type == "linear":
                self.weights_pred = nn.Linear(self.latent_dim, 1, bias=True)
            if self.prediction_type == "non_linear":
                self.weights1_pred = nn.Linear(
                    self.latent_dim, self.latent_dim, bias=True
                )
                self.weights2_pred = nn.Linear(self.latent_dim, 1, bias=True)
        elif self.experiment_type == "vector_split":
            if self.prediction_type == "linear":
                self.weights_pred = nn.Linear(self.score_controller_dim, 1, bias=True)
            if self.prediction_type == "non_linear":
                self.weights1_pred = nn.Linear(
                    self.score_controller_dim, self.score_controller_dim, bias=True
                )
                self.weights2_pred = nn.Linear(self.score_controller_dim, 1, bias=True)

    def compute_variance(self, gold_outcome):
        ys = gold_outcome.tolist()
        variance = np.var(ys)
        return variance

    def encode(self, input_seq):
        # if len(inputs) == 2 and isinstance(inputs, list):
        #    inputs = inputs[0]
        B = input_seq.size(0)
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)
        # use embedding wrapper to encode the input characters
        if not self.rnn_type == "Transformers":
            input_embeddings = self.embedding_wrapper(input_seq)
        else:
            input_embeddings = self.embedding_wrapper_transformers(input_seq)

        # send into the model
        if self.rnn_type == "LSTM":
            _, (hn, _) = self.model(input_embeddings)
        else:
            _, hn = self.model(input_embeddings)
        assert hn.dim() == 3

        # depth * B * memory_dim
        last_hidden_state = hn

        if self.experiment_type == "general":
            # depth * B * latent_dim
            z = self.weights_mu(last_hidden_state)
            # depth * B * latent_dim
            min_sigma = 1e-6
            sigma = torch.clamp(
                torch.exp(
                    -torch.abs(
                        self.weights2_sigma(
                            self.relu_sigma(self.weights1_sigma(last_hidden_state))
                        )
                    )
                ),
                max=1.0,
                min=min_sigma,
            )
        elif self.experiment_type == "vector_split":
            # depth * B * score_controller_dim
            z_style = self.weights_style_mu(last_hidden_state)
            # depth * B * score_controller_dim
            min_sigma = 1e-6
            sigma_style = torch.clamp(
                torch.exp(
                    -torch.abs(
                        self.weights2_style_sigma(
                            self.relu_sigma(
                                self.weights1_style_sigma(last_hidden_state)
                            )
                        )
                    )
                ),
                max=1.0,
                min=min_sigma,
            )
            # depth * B * latent_dim-score_controller_dim
            z_content = self.weights_content_mu(last_hidden_state)
            # depth * B * latent_dim-score_controller_dim
            min_sigma = 1e-6
            sigma_content = torch.clamp(
                torch.exp(
                    -torch.abs(
                        self.weights2_content_sigma(
                            self.relu_sigma(
                                self.weights1_content_sigma(last_hidden_state)
                            )
                        )
                    )
                ),
                max=1.0,
                min=min_sigma,
            )
            z = (z_style, z_content)
            sigma = (sigma_style, sigma_content)
        return z, sigma

    def decode(self, encoder_inputs, z, sigma):
        # if len(inputs) == 2 and isinstance(inputs, list):
        #    encoder_inputs = inputs[0]
        # else:
        #    encoder_inputs = inputs
        B, _ = encoder_inputs.size()
        starter = torch.zeros(B, 1).cuda()
        decoder_inputs = torch.cat([starter, encoder_inputs[:, :-1]], dim=-1).int()
        # compute sampling
        if self.experiment_type == "general":
            depth, B, latent_dim = z.size()
            epsilons = torch.normal(0, 1, size=(depth, B, latent_dim)).cuda()
            sampled_z = z + epsilons * sigma
        elif self.experiment_type == "vector_split":
            (z_style, z_content) = z
            (sigma_style, sigma_content) = sigma
            # style
            depth, B, sdim = z_style.size()
            epsilons_style = torch.normal(0, 1, size=(depth, B, sdim)).cuda()
            sampled_z_style = z_style + epsilons_style * sigma_style
            # content
            depth, B, cdim = z_content.size()
            epsilons_content = torch.normal(0, 1, size=(depth, B, cdim)).cuda()
            sampled_z_content = z_content + epsilons_content * sigma_content
            # depth, B, latent_dim
            sampled_z = torch.cat([sampled_z_style, sampled_z_content], dim=-1)
        # depth * B * hidden_dim
        hidden_z = self.latent2hidden(sampled_z)

        # compute probability
        input_embeddings = self.embedding_wrapper(decoder_inputs)
        # output: B * L * memory_dim
        if self.rnn_type == "LSTM":
            output, _ = self.model(input_embeddings, (hidden_z, hidden_z))
        else:
            output, _ = self.model(input_embeddings, hidden_z)
        assert output.dim() == 3
        # output for training: B * L * vocab_size
        train_probabilities = self.softmax_pi(self.weights_pi(output))
        # print(train_probabilities)
        # print("print min")
        # print(torch.min(output))
        # print("print max")
        # print(torch.max(output))

        return train_probabilities

    def predict_one(self, one_input, hidden_state, c):
        input_embedding = self.embedding_wrapper(one_input)
        if c is None:
            output, hidden = self.model(input_embedding, hidden_state)
            probs = self.softmax_pi(self.weights_pi(output))
            return probs, hidden, None
        else:
            output, (hidden, c0) = self.model(input_embedding, (hidden_state, c))
            probs = self.softmax_pi(self.weights_pi(output))
            return probs, hidden, c0

    def predict(self, z):
        if self.experiment_type == "general":
            _, b, _ = z.size()
        elif self.experiment_type == "vector_split":
            z_style, z_content = z
            _, b, _ = z_style.size()
            z = torch.cat([z_style, z_content], dim=-1)

        initial_token = torch.zeros(b, 1).int().cuda()
        decoder_hidden = self.latent2hidden(z)  # .unsqueeze(0)
        c0 = decoder_hidden
        all_tokens = torch.zeros([b, 0]).cuda()
        all_scores = torch.zeros([b, 0]).cuda()
        for _ in range(self.max_length):
            # Forward pass through decoder
            # decoder_output: B * 1 * vocab_size
            # decoder_hidden: 1 * B * memory_dim
            if not self.rnn_type == "LSTM":
                decoder_output, decoder_hidden, _ = self.predict_one(
                    initial_token, decoder_hidden, None
                )
            else:
                decoder_output, decoder_hidden, c0 = self.predict_one(
                    initial_token, decoder_hidden, c0
                )
            # Obtain most likely word token and its softmax score
            # decoder_scores/input: B * 1
            decoder_scores, decoder_input = torch.max(decoder_output, dim=-1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=-1)
            all_scores = torch.cat((all_scores, decoder_scores), dim=-1)
            # Prepare current token to be next decoder input (add a dimension)
            initial_token = decoder_input

        return all_tokens, all_scores

    def outcome_prediction(self, z, use_sigmoid):
        if self.experiment_type == "vector_split":
            z_style, _ = z
            z = z_style
        if self.prediction_type == "linear":
            outcome = self.weights_pred(z)
        else:
            outcome = self.weights2_pred(self.tanh_pred(self.weights1_pred(z)))
        if use_sigmoid:
            sigmoid = nn.Sigmoid()
            outcome = sigmoid(outcome)
        # B * 1
        return outcome

    def compute_loss(
        self,
        input_seq,
        gold_outcome,
        use_sigmoid,
        seq2seq_importance,
        mse_importance,
        kl_importance,
        invar_importance,
    ):
        z, sigma = self.encode(input_seq)

        if self.experiment_type == "vector_split":
            z_style, z_content = z
            sigma_style, sigma_content = sigma
            _, B, _ = z_style.size()
        elif self.experiment_type == "general":
            _, B, _ = z.size()

        ## original invariance loss
        invar_pred = self.outcome_prediction(z, use_sigmoid)
        decoded, _ = self.predict(z)
        decoded = decoded.int()
        new_z, _ = self.encode(decoded)
        invar_result = self.outcome_prediction(new_z, use_sigmoid)
        non_perturbed_inv_loss = torch.mean(torch.square(invar_pred - invar_result))
        # print("this is inv loss:{}".format(inv_loss))

        """
        ## stochastic new invariance loss
        num_updated = random.randint(0, 100)
        # randomly select one sample
        rand_seq = random.randint(0, B - 1)
        # print("this is num updated:{}".format(num_updated))
        if self.experiment_type == "general":
            updated_z = z.clone()
            updated_z = updated_z[:, rand_seq, :].unsqueeze(0)
        elif self.experiment_type == "vector_split":
            updated_z_style = z_style.clone()
            updated_z_content = z_content.clone()
            updated_z = (
                updated_z_style[:, rand_seq, :].unsqueeze(0),
                updated_z_content[:, rand_seq, :].unsqueeze(0),
            )


        # add perturbation to the sample
        for _ in range(num_updated):
            intermediate_outcome = self.outcome_prediction(updated_z, use_sigmoid)
            if self.experiment_type == "general":
                z_grad = torch.autograd.grad(
                    intermediate_outcome, updated_z, retain_graph=True
                )[0]
                updated_z += z_grad * 0.05
            elif self.experiment_type == "vector_split":
                z_grad_style = torch.autograd.grad(
                    intermediate_outcome, updated_z[0], retain_graph=True
                )[0]
                # z_grad_content = torch.autograd.grad(
                #    intermediate_outcome, updated_z[1], retain_graph=True
                # )[0]
                updated_z = (updated_z[0] + z_grad_style * 0.05, updated_z[1])
                # updated_z[1] += z_grad_content * 0.05

        # compute outcome of the perturbed representation
        if self.experiment_type == "general":
            modified_z = updated_z.clone()
        elif self.experiment_type == "vector_split":
            modified_z = (updated_z[0].clone(), updated_z[1].clone())
        updated_outcome = self.outcome_prediction(modified_z, use_sigmoid)
        # predict a sequence based on the perturbed representation
        updated_sequence, _ = self.predict(modified_z)
        new_z, _ = self.encode(updated_sequence.int().long())
        # compute outcome of this new sequence, the result should be the same
        new_outcome = self.outcome_prediction(new_z, use_sigmoid)
        # print("this is the original outcmoe:{}".format(invar_pred[:, rand_seq, :]))
        # print("this is the updated outcome:{}".format(updated_outcome))
        # print("this is the new outcome:{}".format(new_outcome))
        perturbed_inv_loss = torch.mean(torch.square(updated_outcome - new_outcome))
        """
        # print("this is the perturbed_inv_loss:{}".format(perturbed_inv_loss))
        inv_loss = non_perturbed_inv_loss  # + perturbed_inv_loss

        ## reconstruction loss
        train_probabilities = self.decode(input_seq, z, sigma)
        # result = torch.argmax(train_probabilities, dim=-1)
        # print('the decoded result')
        # print(result)
        # print(train_probabilities)
        log_likelihood = torch.log(train_probabilities).permute(0, 2, 1)
        NLL_loss_fn = nn.NLLLoss()
        # print(log_likelihood)
        # print(input_seq)
        nll_loss = NLL_loss_fn(log_likelihood, input_seq)
        # print("this is the nll loss:{}".format(nll_loss))

        ## KL loss
        if self.experiment_type == "general":
            loss_prior = torch.mean(
                -(0.5 / self.hidden_dim)
                * torch.sum(
                    1.0
                    + torch.log(torch.square(sigma))
                    - torch.square(z)
                    - torch.square(sigma),
                    1,
                )
            )
            priormean_scalingfactor = 0.1
            loss_priormean = torch.mean(
                (0.5 * priormean_scalingfactor / self.hidden_dim)
                * torch.sum(torch.square(z), 1)
            )
        elif self.experiment_type == "vector_split":
            loss_prior_style = torch.mean(
                -(0.5 / self.hidden_dim)
                * torch.sum(
                    1.0
                    + torch.log(torch.square(sigma_style))
                    - torch.square(z_style)
                    - torch.square(sigma_style),
                    1,
                )
            )
            loss_prior_content = torch.mean(
                -(0.5 / self.hidden_dim)
                * torch.sum(
                    1.0
                    + torch.log(torch.square(sigma_content))
                    - torch.square(z_content)
                    - torch.square(sigma_content),
                    1,
                )
            )
            loss_prior = loss_prior_style + loss_prior_content
            priormean_scalingfactor = 0.1
            loss_priormean_style = torch.mean(
                (0.5 * priormean_scalingfactor / self.hidden_dim)
                * torch.sum(torch.square(z_style), 1)
            )
            loss_priormean_content = torch.mean(
                (0.5 * priormean_scalingfactor / self.hidden_dim)
                * torch.sum(torch.square(z_content), 1)
            )
            loss_priormean = loss_priormean_style + loss_priormean_content
        ## mse outcome-prediction loss
        pred_outcome = self.outcome_prediction(z, use_sigmoid).squeeze()  # .squeeze(1)
        # gold_outcome = gold_outcome
        mse_loss_fn = nn.MSELoss()
        # print("this is the pred outcome")
        # print(pred_outcome)
        # print("this is the gold outcome")
        # print(gold_outcome)
        mse_loss = mse_loss_fn(pred_outcome, gold_outcome)
        # print("this is the pred outcome:{}".format(pred_outcome))
        # print("this is the gold outcome:{}".format(gold_outcome))
        # print("this is the mse loss:{}".format(mse_loss))

        outcome_var = self.compute_variance(gold_outcome)
        if outcome_var == 0:
            outcome_var = 1

        loss_joint = (
            seq2seq_importance
            * (
                nll_loss
                + (loss_prior * kl_importance + loss_priormean * (1.0 - kl_importance))
            )
            + (mse_importance / outcome_var) * mse_loss
            + (invar_importance / outcome_var) * inv_loss
        )

        return loss_joint

    def fixed_gradient_optimization(
        self, input_seq, use_sigmoid, lr=0.05, optimization_step=1000
    ):
        edit_distances = []
        all_modifications = []
        correct_modifications = []
        outcome_improvements = []
        sequence_probs = []
        orig_input_seq = input_seq.tolist()[0]

        z, _ = self.encode(input_seq)
        outcome_init = self.outcome_prediction(z, use_sigmoid)
        for _ in range(optimization_step):
            outcome = self.outcome_prediction(z, use_sigmoid)
            if self.experiment_type == "general":
                z_grad = torch.autograd.grad(outcome, z, retain_graph=True)[0]
                z += z_grad * lr
            elif self.experiment_type == "vector_split":
                z_grad_style = torch.autograd.grad(outcome, z[0], retain_graph=True)[0]
                new_z = z[0] + z_grad_style * lr
                z = (new_z, z[1])

            if self.experiment_type == "general":
                modified_z = z.clone()
            elif self.experiment_type == "vector_split":
                modified_z = (z[0].clone(), z[1].clone())
            modified_sequence, _ = self.predict(modified_z)
            sequence_prob = compute_probability(
                modified_sequence[0].int().tolist(),
                self.PAD_ID,
                self.data_style,
                self.vocab,
                self.vocab_size,
            )
            sequence_probs.append(sequence_prob)

            new_actual_z, _ = self.encode(modified_sequence[0].int())
            modified_outcome = self.outcome_prediction(new_actual_z, use_sigmoid)

            outcome_improvement = float(float(modified_outcome) - outcome_init)
            edit_distance = levenshtein(
                input_seq.tolist()[0], modified_sequence.tolist()[0], self.PAD_ID
            )
            outcome_improvements.append(outcome_improvement)
            edit_distances.append(edit_distance / len(orig_input_seq))

            new_seq = modified_sequence.int().tolist()[0]
            all_mod = seq_difference(orig_input_seq, new_seq)
            all_modifications.append(all_mod / len(orig_input_seq))

            one_correct_modification = 0
            if self.data_style == "comparison":
                for i, (old, new) in enumerate(zip(orig_input_seq, new_seq)):
                    if old != new:
                        if old != 5 and new == 0:
                            one_correct_modification += 1
                        if old == 5 and new != 5:
                            one_correct_modification += 1
            elif self.data_style == "general":
                for i, (old, new) in enumerate(zip(orig_input_seq, new_seq)):
                    if old != new:
                        if new == 0:
                            one_correct_modification += 1
            elif self.data_style == "locality":
                for i, (old, new) in enumerate(zip(orig_input_seq, new_seq)):
                    if old != new:
                        if new == 0 and i < 10:
                            one_correct_modification += 1
            correct_modifications.append(one_correct_modification / len(orig_input_seq))

        return (
            modified_sequence,
            all_modifications,
            outcome_improvements,
            edit_distances,
            correct_modifications,
            sequence_probs,
        )

    def constrained_gradient_optimization(
        self, input_seq, log_alpha, use_sigmoid, lr=0.05, optimization_step=1000
    ):
        edit_distances = []
        all_modifications = []
        correct_modifications = []
        outcome_improvements = []
        sequence_probs = []
        orig_input_seq = input_seq.tolist()[0]

        z, sigmas = self.encode(input_seq)
        if self.experiment_type == "vector_split":
            sigmas = sigmas[0]
        sigmas = sigmas.squeeze(0).tolist()
        outcome_init = self.outcome_prediction(z, use_sigmoid)

        # compute the constraint
        min_sigma_threshold = 1e-2
        for i in range(len(sigmas)):
            for j in range(len(sigmas[i])):
                if sigmas[i][j] < min_sigma_threshold:
                    sigmas[i][j] = min_sigma_threshold
        log_alpha = log_alpha - (self.hidden_dim / 2) * np.log(2 * np.pi)
        # 1 * latent_dim
        sigmas_sq = np.square(sigmas)
        # 1 * 1
        Covar = np.expand_dims(np.diag(sigmas_sq), axis=0)
        max_log_alpha = -0.5 * np.sum(np.log(2 * np.pi * np.array(sigmas_sq)))
        if log_alpha > max_log_alpha:
            print(
                "log_alpha = %f is too large (max = %f will return no revision."
                % (log_alpha, max_log_alpha)
            )
            return
        K = -2 * (
            np.log(np.power(2 * np.pi, self.hidden_dim / 2))
            + 0.5 * np.sum(np.log(sigmas_sq))
            + log_alpha
        )
        A = np.linalg.pinv(Covar) / K

        # optimization within the constrains
        convergence_threshold = 1e-8
        for i in range(optimization_step):
            outcome = self.outcome_prediction(z, use_sigmoid)
            if self.experiment_type == "vector_split":
                z_grad = torch.autograd.grad(outcome, z[0], retain_graph=True)[0]
            else:
                z_grad = torch.autograd.grad(outcome, z, retain_graph=True)[0].data
            stepsize = lr * 1000 / (1000 + np.sqrt(i))
            violation = True
            while violation and (stepsize >= convergence_threshold / 100.0):
                if self.experiment_type == "vector_split":
                    new_z = z[0] + z_grad * lr
                    shift = new_z - z[0]
                    z_proposal = (new_z, z[1])
                else:
                    z_proposal = z + z_grad * stepsize
                    # B * latent_dim
                    shift = z_proposal - z
                shift_numpy = shift.squeeze(0).cpu().clone().detach().numpy()
                if (
                    np.dot(shift_numpy, np.dot(A, shift_numpy).transpose()) < 1
                ):  # we are inside constraint-set
                    violation = False
                else:
                    stepsize /= (
                        2.0  # keep dividing by 2 until we remain within constraint
                    )
            if stepsize < convergence_threshold / 100.0:
                break  # break out of for loop.
            z = z_proposal

            if self.experiment_type == "general":
                modified_z = z.clone()
            elif self.experiment_type == "vector_split":
                modified_z = (z[0].clone(), z[1].clone())

            modified_sequence, _ = self.predict(modified_z)
            sequence_prob = compute_probability(
                modified_sequence[0].int(),
                self.PAD_ID,
                self.data_style,
                self.alphabet,
                self.vocab_size,
            )
            sequence_probs.append(sequence_prob)

            new_actual_z, _ = self.encode(modified_sequence[0].int().long())
            modified_outcome = self.outcome_prediction(new_actual_z, use_sigmoid)

            outcome_improvement = float(float(modified_outcome) - outcome_init)
            edit_distance = levenshtein(
                input_seq.tolist()[0], modified_sequence.tolist()[0], self.PAD_ID
            )

            outcome_improvements.append(outcome_improvement)
            edit_distances.append(edit_distance / len(orig_input_seq))

            new_seq = modified_sequence.int().tolist()[0]
            one_correct_modification = 0
            all_mod = seq_difference(orig_input_seq, new_seq)
            all_modifications.append(all_mod / len(orig_input_seq))
            if self.data_style == "comparison":
                for i, (old, new) in enumerate(zip(orig_input_seq, new_seq)):
                    if old != new:
                        if old != 5 and new == 0:
                            one_correct_modification += 1
                        if old == 5 and new != 5:
                            one_correct_modification += 1
            elif self.data_style == "general":
                for i, (old, new) in enumerate(zip(orig_input_seq, new_seq)):
                    if old != new:
                        if new == 0:
                            one_correct_modification += 1
            elif self.data_style == "locality":
                for i, (old, new) in enumerate(zip(orig_input_seq, new_seq)):
                    if old != new:
                        if new == 0 and i < 10:
                            one_correct_modification += 1
            correct_modifications.append(one_correct_modification / len(orig_input_seq))

        return (
            modified_sequence,
            all_modifications,
            outcome_improvements,
            edit_distances,
            correct_modifications,
            sequence_probs,
        )


if __name__ == "__main__":
    random.seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # Create datasets:
    vocab_size = 10
    max_seq_length = 20
    length_range = (10, max_seq_length)
    n_train = 10
    n_val = 10
    n_test = 100
    data_style = "general"

    train_loader, val_loader, test_loader = compute_simulation_dataloader(
        10,
        data_style,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seq_length_range=(10, max_seq_length),
        vocab_size=10,
    )

    # model
    vocab = list("ABCDEFGHIJ")
    rnn_type = "LSTM"
    pred_type = "non_linear"
    embed_dim = 8
    memory_dim = 256
    latent_dim = 128
    score_controller_dim = 16
    experiment_type = "vector_split"
    seq2seq_importance = 1
    mse_importance = 1
    kl_importance = 0
    invar_importance = 0

    use_sigmoid = True
    log_alpha = -10000

    input_seq = torch.tensor(
        [[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 2, 3, 10, 10, 10]]
    ).cuda()
    gold_outcome = torch.tensor(0.2).cuda()

    checkpoint = torch.load(
        "VAE_{}_continuous_{}_prenorm.pt".format(data_style, experiment_type)
    )
    # checkpoint = torch.load("VAE_inv.pt")

    VAE_model = VAE(
        vocab,
        data_style,
        rnn_type,
        pred_type,
        embed_dim,
        memory_dim,
        latent_dim,
        score_controller_dim,
        max_seq_length,
        experiment_type,
    ).cuda()
    VAE_model.load_state_dict(checkpoint)

    """
    for t in test_loader:
        input_seq = t[0].cuda()
        gold_outcome = t[1].tolist()
        z, sigma = VAE_model.encode(input_seq)
        score = VAE_model.outcome_prediction(z, use_sigmoid)
        reconstructed_seq = VAE_model.predict(z)[0]
        for s in range(input_seq.size(0)):
            print(input_seq[s])
            print(reconstructed_seq[s].int())
            print("******")
            # print(gold_outcome)
            # print(score)
    """

    use_sigmoid = True
    log_alpha = -10000

    logger = Logger("{}_{}_result.log".format(data_style, experiment_type))
    logger.log("********")
    logger.log("result after 1000 steps")
    # correct_modification = 0
    # all_modification = 0
    average_edit_distance = []
    average_all_modification = []
    average_correct_modification = []
    average_improvement = []
    average_seq_prob = []
    for t in test_loader:
        input_seq = t[0].cuda()
        gold_outcome = t[1].tolist()
        z, sigma = VAE_model.encode(input_seq)
        score = VAE_model.outcome_prediction(z, use_sigmoid)
        for s in range(input_seq.size(0)):
            seq = input_seq[s].unsqueeze(0).cuda()

            (
                modified_sequence,
                all_modifications,
                outcome_improvements,
                edit_distances,
                correct_modifications,
                sequence_probs,
            ) = VAE_model.fixed_gradient_optimization(
                seq, use_sigmoid, lr=0.05, optimization_step=1000
            )

            """

            (
                modified_sequence,
                all_modifications,
                outcome_improvements,
                edit_distances,
                correct_modifications,
                sequence_probs,
            ) = VAE_model.constrained_gradient_optimization(
                seq, log_alpha, use_sigmoid, lr=0.05, optimization_step=1000
            )
            """

            average_edit_distance.append(edit_distances)
            average_all_modification.append(all_modifications)
            average_correct_modification.append(correct_modifications)
            average_improvement.append(outcome_improvements)
            average_seq_prob.append(sequence_probs)

        # print(new_decode[0].int())
        # print(new_outcome)
        # print("*********")
    average_edit_distance = [
        np.mean([x[i] for x in average_edit_distance])
        for i in range(len(average_edit_distance[0]))
    ]
    average_all_modification = [
        np.mean([x[i] for x in average_all_modification])
        for i in range(len(average_all_modification[0]))
    ]
    average_correct_modification = [
        np.mean([x[i] for x in average_correct_modification])
        for i in range(len(average_correct_modification[0]))
    ]
    correct_proportion = [
        y / x for x, y in zip(average_all_modification, average_correct_modification)
    ]
    average_improvement = [
        np.mean([x[i] for x in average_improvement])
        for i in range(len(average_improvement[0]))
    ]
    average_seq_prob = [
        np.mean([x[i] for x in average_seq_prob])
        for i in range(len(average_seq_prob[0]))
    ]
    print("this is average edit distance")
    print(average_edit_distance)
    print("this is average all modification")
    print(average_all_modification)
    print("this is average correct modification")
    print(average_correct_modification)
    print("this is the correct modification proportion")
    print(correct_proportion)
    print("this is average improvement")
    print(average_improvement)
    print("this is average sequence probability")
    print(average_seq_prob)

    """
    logger.log(
        "average improvement per sentence is {}".format(np.mean(avererage_improvement))
    )

    logger.log(
        "all correct modifications per sentence is {}".format(
            np.mean(average_correct_modification)
        )
    )
    logger.log(
        "all modification per sentence is {}".format(np.mean(average_all_modification))
    )
    logger.log(
        "average correctness is {}".format(correct_modification / all_modification)
    )
    """

    """
    (
        modified_sequence,
        outcome_improvement,
        edit_distance,
    ) = VAE_model.constrained_gradient_optimization(input_seq, log_alpha, use_sigmoid)

    # = VAE_model.fixed_gradient_optimization(
    #    input_seq, use_sigmoid, lr=0.05, optimization_step=1000
    # )

    # VAE_model.constrained_gradient_optimization(input_seq, log_alpha, use_sigmoid)
    print(modified_sequence)
    print(outcome_improvement)
    print(edit_distance)


    VAE_model = VAE(
        vocab,
        rnn_type,
        pred_type,
        embed_dim,
        memory_dim,
        latent_dim,
        score_controller_dim,
        max_seq_length,
        experiment_type,
    ).cuda()
    optimizer = torch.optim.AdamW(VAE_model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(VAE_model.parameters(), lr=0.001, momentum=0.9)

    num_training_steps = n_train * 100 / 2
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 0.3 * num_training_steps, num_training_steps
    )

    loss_val = 0
    step = 0
    VAE_model.zero_grad()
    for epoch in range(100):
        for t in train_loader:
            input_seq = t[0].cuda()
            gold_outcome = t[1].cuda()
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
            optimizer.step()
            VAE_model.zero_grad()
            # scheduler.step()
            step += 1
            loss_val += loss.item()
            if step % 1000 == 0:
                print(loss_val / 1000)
                loss_val = 0

    z, sigma = VAE_model.encode(input_seq)
    out = VAE_model.predict(z)
    print("this is the final prediction")
    print(out)
    score = VAE_model.outcome_prediction(z, True)
    print(score)

    VAE_model = VAE(
        vocab,
        rnn_type,
        pred_type,
        embed_dim,
        memory_dim,
        latent_dim,
        score_controller_dim,
        max_seq_length,
        experiment_type,
    ).cuda()
    optimizer = torch.optim.AdamW(VAE_model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(VAE_model.parameters(), lr=0.001, momentum=0.9)

    loss_val = 0
    step = 0
    VAE_model.zero_grad()
    for epoch in range(100):
        for t in train_loader:
            input_seq = t[0].cuda()
            gold_outcome = t[1].cuda()
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
            optimizer.step()
            VAE_model.zero_grad()
            # scheduler.step()
            step += 1
            loss_val += loss.item()
            if step % 10 == 0:
                print(loss_val)
                loss_val = 0


    z, sigma = VAE_model.encode(input_seq)
    out = VAE_model.predict(z)
    print("this is the final prediction")
    print(out)
    score = VAE_model.outcome_prediction(z, True)
    print(score)
    """
