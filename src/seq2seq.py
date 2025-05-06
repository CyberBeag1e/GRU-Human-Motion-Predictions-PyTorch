from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import List

use_cuda = True

class Attention(nn.Module):

    def __init__(self, enc_output_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(enc_output_size, dec_hidden_size, bias = False)
        self.Ua = nn.Linear(dec_hidden_size, dec_hidden_size, bias = False)
        self.Va = nn.Linear(dec_hidden_size, 1, bias = False)
    
    def forward(self, decoder_hidden: torch.Tensor, 
                      encoder_outputs: torch.Tensor):
        """
        Args:
            `decoder_hidden` (`torch.Tensor`): Current decoder hidden state (batch_size, hidden_size)
            `encoder_outputs` (`torch.Tensor`): Outputs from encoder (seq_len, batch_size, hidden_size)
        Returns:
            `context_vector` (`torch.Tensor`): Context vector (batch_size, hidden_size)
            `attn_weights` (`torch.Tensor`): Attention weights (batch_size, 1, seq_len)
        """

        batch_size, enc_seq_len, _ = encoder_outputs.shape
        dec_hidden_size = decoder_hidden.shape[-1]

        # Shape: (batch_size, 1, dec_hidden_size)
        decoder_hidden_proj = self.Ua(decoder_hidden).unsqueeze(1)

        # Shape: (batch_size, enc_seq_len, dec_hidden_size)
        encoder_outputs_proj = self.Wa(encoder_outputs)

        # Calculate energy (alignment scores) using broadcasting
        # Shape: (batch_size, enc_seq_len, dec_hidden_size) -> (batch_size, enc_seq_len, 1)
        energy = torch.tanh(decoder_hidden_proj + encoder_outputs_proj)

        # Shape: (batch_size, enc_seq_len)
        alignment_scores = self.Va(energy).squeeze(-1) 

        # Calculate attention weights
        attn_weights = F.softmax(alignment_scores, dim = 1) # Shape: (batch_size, enc_seq_len)

        # Calculate context vector (weighted sum of encoder outputs)
        # Shape: (batch_size, 1, enc_output_size) -> (batch_size, enc_output_size)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context_vector, attn_weights


class Seq2SeqModel(nn.Module):

    def __init__(self,
                 source_seq_len: int, 
                 target_seq_len: int,
                 number_of_actions: int,
                 rnn_size: int = 1024,
                 num_layers: int = 1,
                 max_gradient_norm: float = 5.0,
                 batch_size: int = 16,
                 learning_rate: float = 0.005,
                 learning_rate_decay_factor: float = 0.95,
                 loss_to_use: str = "sampling_based",
                 one_hot: bool = True,
                 bidirectional: bool = False,
                 attention: bool = False,
                 tied: bool = True,
                 residual_velocities: bool = False,
                 dropout: float = 0.0,
                 dtype = torch.float32):
        
        """
        reate the model.

        Args:
            `source_seq_len`: lenght of the input sequence.
            `target_seq_len`: lenght of the target sequence.
            `number_of_actions`: number of classes we have.
            `rnn_size`: number of units in the rnn.
            `num_layers`: number of rnns to stack.
            `max_gradient_norm`: gradients will be clipped to maximally this norm.
            `batch_size`: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            `learning_rate`: learning rate to start with.
            `learning_rate_decay_factor`: decay learning rate by this much when needed.
            `loss_to_use`: [supervised, sampling_based]. Whether to use ground truth in
                           each timestep to compute the loss after decoding, or to feed back the
                           prediction from the previous time-step.
            `one_hot`: whether to use one_hot encoding during train/test (sup models).
            `bidirectional`: whether to use bidirectional encoder.
            `attention`: whether to introduce attention mechanism.
            `tied`: whether to tie encoder and decoder (share weights between encoding and decoding processes).
            `residual_velocities`: whether to use a residual connection that models velocities.
            `dtype`: the data type to use to store internal variables.
        """

        super(Seq2SeqModel, self).__init__()
        self.MOTION_SIZE = 54
        self.input_size = self.MOTION_SIZE + number_of_actions if one_hot else self.MOTION_SIZE
        self.lr = learning_rate
        self.gradient_clip = max_gradient_norm
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

        ## Architecture

        if bidirectional and tied:
            raise ValueError("Cannot use shared GRU (tied=True) with a bidirectional encoder (bidirectional=True). "
                             "Set tied=False if using bidirectional=True.")
        
        self.tied = tied

        if loss_to_use == "sampling_based":
            self.teacher_forcing = False
        elif loss_to_use == "supervised":
            self.teacher_forcing = True
        else:
            raise ValueError(f"Unknown loss type: {loss_to_use}")
        
        self.bidirectional = bidirectional
        self.n_direction = 2 if self.bidirectional else 1
        self.residual_velocities = residual_velocities
        self.use_attention = attention

        print(f"One hot is {one_hot}")
        print(f"Input size is {self.input_size}")
        print(f"Tied: {self.tied}\nBidirectional: {self.bidirectional}\nFeed Ground Truth: {self.teacher_forcing}\nResidual: {self.residual_velocities}")

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout

        if self.tied:
            self.cell = nn.GRU(
                self.input_size, 
                self.rnn_size, 
                num_layers = self.num_layers,
                batch_first = True,
                # dropout = self.dropout if self.num_layers > 1 else 0,
                bidirectional = False
            )

            self.enc_GRU = self.cell
            self.dec_GRU = self.cell
        else:
            self.enc_GRU = nn.GRU(
                self.input_size, 
                self.rnn_size, 
                num_layers = self.num_layers, 
                batch_first = True, 
                # dropout = self.dropout if self.num_layers > 1 else 0,
                bidirectional = self.bidirectional
            )
            
            self.dec_GRU = nn.GRU(
                self.input_size, 
                self.rnn_size, 
                num_layers = self.num_layers,
                batch_first = True,
                # dropout = self.dropout if self.num_layers > 1 else 0, 
                bidirectional = False
            )
        
        enc_output_size = self.rnn_size * self.n_direction
        if self.bidirectional:
            self.fc_hidden = nn.Linear(enc_output_size, self.rnn_size)
        else:
            self.fc_hidden = nn.Identity()

        if self.use_attention:
            self.attention = Attention(enc_output_size, self.rnn_size)
            fc1_input_size = self.rnn_size + enc_output_size
        else:
            fc1_input_size = self.rnn_size

        self.fc1 = nn.Linear(fc1_input_size, self.input_size)
    

    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor):
        batch_size = encoder_inputs.shape[0]
        enc_seq_len = encoder_inputs.shape[1]
        h0 = torch.zeros(self.num_layers * self.n_direction, batch_size, self.rnn_size).to(self.device)

        enc_outputs, enc_hidden = self.enc_GRU(encoder_inputs, h0)
        enc_hidden = enc_hidden.view(self.num_layers, self.n_direction, batch_size, self.rnn_size)

        if self.bidirectional:
            enc_last = torch.cat((enc_hidden[-1, 0, :, :], enc_hidden[-1, 1, :, :]), dim = 1)
            dec_hidden = torch.tanh(self.fc_hidden(enc_last))
        else:
            dec_hidden = enc_hidden[-1, 0, :, :]
        
        dec_hidden = dec_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)

        predictions = []
        curr_inp = decoder_inputs[:, 0, :].unsqueeze(1).detach()

        for t in range(self.target_seq_len):

            if self.teacher_forcing and self.training:
                curr_inp = decoder_inputs[:, t, :].unsqueeze(1).detach()
            
            dec_output, dec_hidden = self.dec_GRU(curr_inp, dec_hidden)
            dec_output = F.dropout(dec_output, self.dropout, training = self.training)

            if self.use_attention:
                # Use the last layer's current hidden state as the query
                # Shape: (num_layers, batch, rnn_size) -> (batch, rnn_size)
                query_state = dec_hidden[-1, :, :]
                # encoder_outputs shape: (batch, enc_seq_len, encoder_output_size)
                context_vector, attn_weights = self.attention(query_state, enc_outputs)
                # context_vector: (batch, encoder_output_size)
                # attn_weights: (batch, enc_seq_len)

                combined_output = torch.cat((dec_output.squeeze(1), context_vector), dim = 1)
            else:
                combined_output = dec_output.squeeze(1)
            
            fc_output = self.fc1(combined_output)
            prev_pose = curr_inp.squeeze(1)

            if self.residual_velocities:
                pred = prev_pose + fc_output
            else:
                pred = fc_output
            
            predictions.append(pred)
            curr_inp = pred.unsqueeze(1)
        
        outputs = torch.stack(predictions, dim = 1)

        return outputs


    def get_batch(self, data: dict, actions: str | List[str]):
        all_keys = list(data.keys())
        chosen_keys = np.random.choice(len(all_keys), self.batch_size)

        total_frames = self.source_seq_len + self.target_seq_len

        encoder_inputs = np.zeros((self.batch_size, self.source_seq_len - 1, self.input_size), dtype = np.float32)
        decoder_inputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype = np.float32)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype = np.float32)

        for i in range(self.batch_size):
            the_key = all_keys[chosen_keys[i]]
            n, _ = data[the_key].shape
            idx = np.random.randint(16, n - total_frames)
            data_sel = data[the_key][idx: idx + total_frames, :]

            encoder_inputs[i, :, 0: self.input_size] = data_sel[0: self.source_seq_len - 1, :]
            decoder_inputs[i, :, 0: self.input_size] = data_sel[self.source_seq_len - 1: self.source_seq_len + self.target_seq_len - 1, :]
            decoder_outputs[i, :, 0: self.input_size] = data_sel[self.source_seq_len:, 0: self.input_size]
        
        return encoder_inputs, decoder_inputs, decoder_outputs
    
    def find_indices_srnn(self, 
                          data: dict, 
                          action: List[str]):
        """
        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """

        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState(SEED)

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[(subject, action, subaction1, 'even')].shape[0]
        T2 = data[(subject, action, subaction2, 'even')].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))

        return idx

    def get_batch_srnn(self, 
                       data: dict, 
                       action: List[str]):
        """
        Get a random batch of data from the specified bucket, prepare for step.

        Args
            `data`: dictionary with k:v, k=((subject, action, subsequence, 'even')),
                  v=nxd matrix with a sequence of poses
            `action`: the action to load data from
        Returns
            The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
            the constructed batches have the proper format to call step(...) later.
        """

        actions = ["directions", "discussion", "eating", "greeting", "phoning", 
                   "posing", "purchases", "sitting", "sittingdown", "smoking",
                   "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

        if not action in actions:
            raise ValueError(f"Unrecognized action {action}")

        frames = {}
        frames[action] = self.find_indices_srnn(data, action)

        batch_size = 8                                                                                          # We always evaluate 8 seeds
        subject = 5                                                                                             # We always evaluate on subject 5
        source_seq_len = self.source_seq_len
        target_seq_len = self.target_seq_len

        seeds = [(action, (i % 2) + 1, frames[action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, self.input_size), dtype = float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype = float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype = float)

        # Reproducing SRNN's sequence subsequence selection as done in
        # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
        for i in range(batch_size):

            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[(subject, action, subsequence, 'even')]

            data_sel = data_sel[(idx - source_seq_len): (idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0: source_seq_len - 1, :]
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1: (source_seq_len + target_seq_len - 1), :]
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]

        return encoder_inputs, decoder_inputs, decoder_outputs
