import torch
import torch.nn as nn
from beer_data import compute_beer_data, word_embedding, Collator
import numpy as np
import os
import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


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

    print(
        "this is the train average reconstruction errors:{}".format(
            averaged_reconstruction_errors
        )
    )
    print(
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

    print(
        "this is the test average reconstruction errors:{}".format(
            averaged_reconstruction_errors
        )
    )
    print("this is the test average outcome errors:{}".format(averaged_outcomes_errors))

    return averaged_reconstruction_errors, averaged_outcomes_errors


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


def compute_embeddings(input_seq, glove):
    input_embeddings = []
    for seq in input_seq:
        seq_embeddings = []
        for s in seq:
            seq_embeddings.append(glove[int(s)])
        input_embeddings.append(seq_embeddings)
    input_embeddings = torch.tensor(input_embeddings).float()
    # print('*******')
    # print(input_embeddings.size())
    return input_embeddings


class VAE(nn.Module):
    def __init__(
        self,
        glove,
        vocab,
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
        self.glove = glove
        self.vocab = vocab
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
                self.embedding_dim, self.hidden_dim, 8, batch_first=True
            )
            self.depth = 8

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
                self.weights_pred = nn.Linear(
                    self.depth * self.latent_dim, 1, bias=True
                )
            if self.prediction_type == "non_linear":
                self.weights1_pred = nn.Linear(
                    self.depth * self.latent_dim,
                    self.depth * self.latent_dim,
                    bias=True,
                )
                self.weights2_pred = nn.Linear(
                    self.depth * self.latent_dim, 1, bias=True
                )
        elif self.experiment_type == "vector_split":
            if self.prediction_type == "linear":
                self.weights_pred = nn.Linear(
                    self.depth * self.score_controller_dim, 1, bias=True
                )
            if self.prediction_type == "non_linear":
                self.weights1_pred = nn.Linear(
                    self.depth * self.score_controller_dim,
                    self.depth * self.score_controller_dim,
                    bias=True,
                )
                self.weights2_pred = nn.Linear(
                    self.depth * self.score_controller_dim, 1, bias=True
                )

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
        # print("this is the input seq size {}".format(input_seq.size()))
        # use embedding wrapper to encode the input characters
        input_embeddings = compute_embeddings(input_seq, self.glove).cuda().float()
        # print("this is the inpue embeddings size {}".format(input_embeddings.size()))

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
        # print("the size of decoder inputs {}".format(decoder_inputs.size()))
        input_embeddings = compute_embeddings(decoder_inputs, self.glove).cuda()
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
        input_embedding = compute_embeddings(one_input, self.glove).cuda()
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
        # depth * b * hidden_dim
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
        B = z.size(1)
        outcomes = []
        for b in range(B):
            one_z = z[:, b, :].contiguous().view(1, -1)
            if self.prediction_type == "linear":
                outcome = self.weights_pred(one_z)
            else:
                outcome = self.weights2_pred(self.tanh_pred(self.weights1_pred(one_z)))
            if use_sigmoid:
                sigmoid = nn.Sigmoid()
                outcome = sigmoid(outcome)
            outcomes.append(outcome)
        outcome = torch.cat(outcomes, dim=0)
        """
        if self.experiment_type == "vector_split":
            z_style, _ = z
            z = z_style
        z = z.permute(1, 0, 2)
        B = z.size(0)
        z = z.contiguous().view(B, -1)
        if self.prediction_type == "linear":
            outcome = self.weights_pred(z)
        else:
            outcome = self.weights2_pred(self.tanh_pred(self.weights1_pred(z)))
        if use_sigmoid:
            sigmoid = nn.Sigmoid()
            outcome = sigmoid(outcome)
        """
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
        # print("this is the size of z {}".format(z.size()))
        decoded, _ = self.predict(z)
        # print("this is the size of the predicted decode seq {}".format(decoded.size()))
        decoded = decoded.int()
        new_z, _ = self.encode(decoded)
        invar_result = self.outcome_prediction(new_z, use_sigmoid)
        non_perturbed_inv_loss = torch.mean(torch.square(invar_pred - invar_result))
        # print("this is inv loss:{}".format(non_perturbed_inv_loss))

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
        # print(
        #    "this is the size of train_probabilities {}".format(
        #        train_probabilities.size()
        #    )
        # )
        log_likelihood = torch.log(train_probabilities).permute(0, 2, 1)
        # print("this is the log likelihood size {}".format(log_likelihood.size()))
        NLL_loss_fn = nn.NLLLoss()
        # print(log_likelihood.size())
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
        # print("******")
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

        # loss_joint = mse_loss + nll_loss

        return loss_joint

    def fixed_gradient_optimization(
        self, input_seq, use_sigmoid, lr=0.05, optimization_step=1000
    ):
        z, _ = self.encode(input_seq)
        outcome_init = self.outcome_prediction(z, use_sigmoid)
        for _ in range(optimization_step):
            outcome = self.outcome_prediction(z, use_sigmoid)
            if self.experiment_type == "general":
                z_grad = torch.autograd.grad(outcome, z, retain_graph=True)[0]
                z += z_grad * lr
            elif self.experiment_type == "vector_split":
                z_grad_style = torch.autograd.grad(outcome, z[0], retain_graph=True)[0]
                z_grad_content = torch.autograd.grad(outcome, z[1], retain_graph=True)[
                    0
                ]
                z[0] += z_grad_style * lr
                z[1] += z_grad_content * lr

        if self.experiment_type == "general":
            modified_z = z.clone()
        elif self.experiment_type == "vector_split":
            modified_z = (z[0].clone(), z[1].clone())
        modified_sequence, _ = self.predict(modified_z)
        modified_outcome = self.outcome_prediction(modified_z, use_sigmoid)

        outcome_improvement = modified_outcome - outcome_init
        edit_distance = levenshtein(
            input_seq.tolist()[0], modified_sequence.tolist()[0], self.PAD_ID
        )

        return modified_sequence, outcome_improvement, edit_distance

    def constrained_gradient_optimization(
        self, input_seq, log_alpha, use_sigmoid, lr=0.05, optimization_step=1000
    ):
        z, sigmas = self.encode(input_seq)
        sigmas = sigmas.tolist()
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
            z_grad = torch.autograd.grad(outcome, z, retain_graph=True)[0].data
            stepsize = lr * 1000 / (1000 + np.sqrt(i))
            violation = True
            while violation and (stepsize >= convergence_threshold / 100.0):
                z_proposal = z + z_grad * stepsize
                # B * latent_dim
                shift = z_proposal - z
                shift_numpy = shift.clone().detach().numpy()
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
        modified_z = z_proposal.clone()

        modified_sequence, _ = self.predict(modified_z)
        modified_outcome = self.outcome_prediction(modified_z, use_sigmoid)

        outcome_improvement = modified_outcome - outcome_init
        edit_distance = levenshtein(
            input_seq.tolist()[0], modified_sequence.tolist()[0], self.PAD_ID
        )

        return modified_sequence, outcome_improvement, edit_distance


if __name__ == "__main__":
    set_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # create dataset
    data_dir = "data/"
    glove_dir = "data/glove/"
    review_type = "appearance"
    batch_size = 10
    toy = True

    _, vocab, _ = word_embedding(glove_dir)
    collator = Collator(vocab)
    train_loader, test_loader = compute_beer_data(
        data_dir, glove_dir, review_type, batch_size, collator, toy
    )

    # model
    rnn_type = "LSTM"
    pred_type = "non_linear"
    embed_dim = 50
    memory_dim = 512
    latent_dim = 256
    score_controller_dim = 16
    max_seq_length = 70
    experiment_type = "general"
    seq2seq_importance = 1
    mse_importance = 1
    kl_importance = 0
    invar_importance = 0

    use_sigmoid = True
    log_alpha = -10000

    glove, vocab, _ = word_embedding(glove_dir)

    # checkpoint = torch.load("VAE_inv.pt")
    VAE_model = VAE(
        glove,
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
    # VAE_model.load_state_dict(checkpoint)

    """
    input_seq = torch.tensor(
        [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10]]
    ).cuda()
    gold_outcome = torch.tensor(0.2).cuda()

    use_sigmoid = True
    log_alpha = -10000

    (
        modified_sequence,
        outcome_improvement,
        edit_distance,
    ) = VAE_model.fixed_gradient_optimization(
        input_seq, use_sigmoid, lr=0.05, optimization_step=1000
    )

    print(modified_sequence)
    print(outcome_improvement)
    print(edit_distance)

    z, _ = VAE_model.encode(modified_sequence.int().long())
    new_decode = VAE_model.predict(z)
    new_outcome = VAE_model.outcome_prediction(z, use_sigmoid)
    print(new_decode)
    print(new_outcome)

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
    """

    optimizer = torch.optim.AdamW(VAE_model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(VAE_model.parameters(), lr=0.001, momentum=0.9)

    loss_val = 0
    step = 0
    VAE_model.zero_grad()
    for epoch in range(100):
        for i, t in enumerate(train_loader):
            input_seq = torch.tensor(t[0]).cuda()
            gold_outcome = torch.tensor(t[1]).cuda()
            # print(gold_outcome)
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
        (
            averaged_reconstruction_errors,
            averaged_outcomes_errors,
        ) = compute_evaluation(VAE_model)

    """
    z, sigma = VAE_model.encode(input_seq)
    out = VAE_model.predict(z)
    print("this is the final prediction")
    print(out)
    score = VAE_model.outcome_prediction(z, True)
    print(score)
    """
