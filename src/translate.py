from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import h5py
import numpy as np
import os
import sys
import time

import textwrap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from typing import List

import data_utils
from seq2seq import Seq2SeqModel

########## -------------------- Argument Parsing -------------------- ##########
# Learning
parser = argparse.ArgumentParser(description='Train RNN for human pose estimation')
parser.add_argument('--learning_rate', 
                    dest = 'learning_rate', 
                    help = 'Learning rate',
                    default = 0.005, 
                    type = float)

parser.add_argument('--learning_rate_decay_factor', 
                    dest = 'learning_rate_decay_factor',
                    help = 'Learning rate is multiplied by this much. 1 means no decay.',
                    default = 0.95, 
                    type = float)

parser.add_argument('--learning_rate_step', 
                    dest = 'learning_rate_step', 
                    help = 'Every this many steps, do decay.', 
                    default = 10000, 
                    type = int)

parser.add_argument('--batch_size', 
                    dest = 'batch_size', 
                    help = 'Batch size to use during training.',
                    default = 16, 
                    type = int)

parser.add_argument('--max_gradient_norm', 
                    dest = 'max_gradient_norm',
                    help = 'Clip gradients to this norm.',
                    default = 5.0, 
                    type = float)

parser.add_argument('--iterations', 
                    dest = 'iterations',
                    help = 'Iterations to train for.',
                    default = int(1e4), 
                    type = int)

parser.add_argument('--test_every', 
                    dest = 'test_every', 
                    help='',
                    default = 1000, 
                    type = int)

# Architecture
parser.add_argument('--loss_to_use', 
                    dest = 'loss_to_use', 
                    help = 'The type of loss to use, supervised or sampling_based',
                    default = 'sampling_based', 
                    type = str)

parser.add_argument('--bidirectional', 
                    dest = "bidirectional", 
                    help = "Whether to use a bidirectional encoder or not.", 
                    action = "store_true",
                    default = False)

parser.add_argument('--attention', 
                    dest = 'attention', 
                    help = 'Use attention mechanism or not.', 
                    action = "store_true",
                    default = False)

parser.add_argument('--untied', 
                    dest = 'untied', 
                    help = "Whether to tie the encoder and decoder.", 
                    action = "store_true", 
                    default = False)

parser.add_argument('--residual_velocities', 
                    dest = 'residual_velocities', 
                    help = 'Add a residual connection that effectively models velocities',
                    action = 'store_true', 
                    default = False)

parser.add_argument('--size', 
                    dest = 'size', 
                    help = 'Size of each model layer.',
                    default = 1024, 
                    type = int)

parser.add_argument('--num_layers', 
                    dest = 'num_layers',
                    help = 'Number of layers in the model.', 
                    default = 1, 
                    type = int)

parser.add_argument('--seq_length_in', 
                    dest = 'seq_length_in',
                    help = 'Number of frames to feed into the encoder. 25 fp',
                    default = 50, 
                    type = int)

parser.add_argument('--seq_length_out', 
                    dest = 'seq_length_out', 
                    help = 'Number of frames that the decoder has to predict. 25fps', 
                    default = 10, 
                    type = int)

parser.add_argument('--omit_one_hot', 
                    dest = 'omit_one_hot', 
                    action = 'store_true', 
                    default = False)

# Directories
parser.add_argument('--data_dir', 
                    dest = 'data_dir',
                    help = 'Data directory', 
                    default = os.path.normpath("../Data/H36M_Expmap"), 
                    type = str)

parser.add_argument('--train_dir', 
                    dest = 'train_dir', 
                    help = 'Training directory', 
                    default = os.path.normpath("./experiments/"), 
                    type = str)

parser.add_argument('--action', 
                    dest = 'action', 
                    help = 'The action to train on. all means all the actions, all_periodic means walking, eating and smoking', 
                    default = 'all', 
                    type = str)

parser.add_argument('--use_cpu', 
                    dest = 'use_cpu', 
                    help = '', 
                    action = 'store_true', 
                    default = False)

parser.add_argument('--load', 
                    dest = 'load', 
                    help = 'Try to load a previous checkpoint.', 
                    default = 0, 
                    type = int)

parser.add_argument('--sample', 
                    dest = 'sample', 
                    help = 'Set to True for sampling.', 
                    action = 'store_true', 
                    default = False)

args = parser.parse_args()

train_dir = os.path.normpath(os.path.join(
    args.train_dir, 
    args.action,
    f'out_{args.seq_length_out}',
    f'iterations_{args.iterations}',
    'untied' if args.untied else 'tied',
    args.loss_to_use,
    'bidirectional' if args.bidirectional else 'unidirectional',
    'attention' if args.attention else 'no_attention',
    'omit_one_hot' if args.omit_one_hot else 'one_hot',
    f'depth_{args.num_layers}',
    f'size_{args.size}',
    f'lr_{args.learning_rate}',
    'residual_vel' if args.residual_velocities else 'not_residual_vel'))

print(train_dir)
os.makedirs(train_dir, exist_ok = True)

def create_model(actions: str | List[str], sampling = False):
    model = Seq2SeqModel(
        source_seq_len = args.seq_length_in if not sampling else 50,
        target_seq_len = args.seq_length_out if not sampling else 100,
        number_of_actions = len(actions),
        rnn_size = args.size,
        num_layers = args.num_layers,
        max_gradient_norm = args.max_gradient_norm,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        learning_rate_decay_factor = args.learning_rate_decay_factor,
        loss_to_use = args.loss_to_use if not sampling else "sampling_based",
        one_hot = not args.omit_one_hot,
        bidirectional = args.bidirectional,
        attention = args.attention,
        tied = not args.untied,
        residual_velocities = args.residual_velocities,
        dtype = torch.float32
    )

    if args.load <= 0:
        return model
    
    print("Loading Model.")
    model = torch.load(os.path.join(train_dir, f"model_{args.load}"), weights_only = False)             ## New PyTorch version set weight_only from default False to default True

    if sampling:
        model.source_seq_len = 50
        model.target_seq_len = 100

    return model

def train():
    actions = define_actions(args.action)
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(actions, 
                                                                                        args.seq_length_in, 
                                                                                        args.seq_length_out, 
                                                                                        args.data_dir, 
                                                                                        not args.omit_one_hot)
    
    model = create_model(actions, args.sample)

    if not args.use_cpu:
        model = model.cuda()
    
    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_euler = get_srnn_gts(actions, 
                                  model, 
                                  test_set, 
                                  data_mean,
                                  data_std, 
                                  dim_to_ignore, 
                                  not args.omit_one_hot)
    
    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if args.load <= 0 else args.load + 1
    previous_losses = []

    optimizer = optim.SGD(model.parameters(), lr = model.lr)

    for epoch in range(args.iterations):
        optimizer.zero_grad()
        model.train()

        start_time = time.time()
        
        # Actual training

        # === Training step ===
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(train_set, not args.omit_one_hot)
        encoder_inputs = torch.from_numpy(encoder_inputs).float()
        decoder_inputs = torch.from_numpy(decoder_inputs).float()
        decoder_outputs = torch.from_numpy(decoder_outputs).float()

        if not args.use_cpu:
            encoder_inputs = encoder_inputs.cuda()
            decoder_inputs = decoder_inputs.cuda()
            decoder_outputs = decoder_outputs.cuda()

        encoder_inputs = Variable(encoder_inputs)
        decoder_inputs = Variable(decoder_inputs)
        decoder_outputs = Variable(decoder_outputs)

        preds = model(encoder_inputs, decoder_inputs)
        # print(f"preds shape: {preds.shape}; decoder_output shape: {decoder_outputs.shape}")

        step_loss = (preds - decoder_outputs) ** 2
        step_loss = step_loss.mean()

        # Actual backpropagation
        step_loss.backward()
        optimizer.step()

        step_loss = step_loss.cpu().data.numpy()

        if current_step % 10 == 0:
            print(f"step {current_step:04d}; step_loss: {step_loss:.4f}")

        step_time += (time.time() - start_time) / args.test_every
        loss += step_loss / args.test_every
        current_step += 1
        # === step decay ===
        if current_step % args.learning_rate_step == 0:
            model.lr = model.lr * args.learning_rate_decay_factor
            optimizer = optim.Adam(model.parameters(), lr = model.lr, betas = (0.9, 0.999))
            print(f"Decay learning rate. New value at {model.lr}")
        
        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % args.test_every == 0:
            model.eval()

            # === Validation with randomly chosen seeds ===
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(test_set, not args.omit_one_hot)
            encoder_inputs = torch.from_numpy(encoder_inputs).float()
            decoder_inputs = torch.from_numpy(decoder_inputs).float()
            decoder_outputs = torch.from_numpy(decoder_outputs).float()

            if not args.use_cpu:
                encoder_inputs = encoder_inputs.cuda()
                decoder_inputs = decoder_inputs.cuda()
                decoder_outputs = decoder_outputs.cuda()

            encoder_inputs = Variable(encoder_inputs)
            decoder_inputs = Variable(decoder_inputs)
            decoder_outputs = Variable(decoder_outputs)

            preds = model(encoder_inputs, decoder_inputs)

            step_loss = (preds - decoder_outputs) ** 2
            step_loss = step_loss.mean()

            val_loss = step_loss # Loss book-keeping

            print()
            print(f"{"milliseconds": <16} |", end="")
            for ms in [80, 160, 320, 400, 560, 1000]:
                print(f" {ms:5d} |", end="")
            print()

            # === Validation with srnn's seeds ===
            for action in actions:

            # Evaluate the model on the test batches
                encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(test_set, action)

            #### Evaluate model on action
                encoder_inputs = torch.from_numpy(encoder_inputs).float()
                decoder_inputs = torch.from_numpy(decoder_inputs).float()
                decoder_outputs = torch.from_numpy(decoder_outputs).float()

                if not args.use_cpu:
                    encoder_inputs = encoder_inputs.cuda()
                    decoder_inputs = decoder_inputs.cuda()
                    decoder_outputs = decoder_outputs.cuda()
                    
                encoder_inputs = Variable(encoder_inputs)
                decoder_inputs = Variable(decoder_inputs)
                decoder_outputs = Variable(decoder_outputs)

                srnn_poses = model(encoder_inputs, decoder_inputs)


                srnn_loss = (srnn_poses - decoder_outputs) ** 2
                srnn_loss.cpu().data.numpy()
                srnn_loss = srnn_loss.mean()

                srnn_poses = srnn_poses.cpu().data.numpy()
                srnn_poses = srnn_poses.transpose([1, 0, 2])

                srnn_loss = srnn_loss.cpu().data.numpy()

                # Denormalize the output
                srnn_pred_expmap = data_utils.revert_output_format(srnn_poses, 
                                                                   data_mean, 
                                                                   data_std, 
                                                                   dim_to_ignore, 
                                                                   actions, not args.omit_one_hot)

                # Save the errors here
                mean_errors = np.zeros((len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]))

                # Training is done in exponential map, but the error is reported in
                # Euler angles, as in previous work.
                # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
                N_SEQUENCE_TEST = 8
                for i in np.arange(N_SEQUENCE_TEST):
                    eulerchannels_pred = srnn_pred_expmap[i]

                    # Convert from exponential map to Euler angles
                    for j in np.arange(eulerchannels_pred.shape[0]):
                        for k in np.arange(3, 97, 3):
                            eulerchannels_pred[j, k: k + 3] = data_utils.rotmat2euler(data_utils.expmap2rotmat(eulerchannels_pred[j, k: k + 3]))

                    # The global translation (first 3 entries) and global rotation
                    # (next 3 entries) are also not considered in the error, so the_key
                    # are set to zero.
                    # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
                    gt_i = np.copy(srnn_gts_euler[action][i])
                    gt_i[:, 0: 6] = 0

                    # Now compute the l2 error. The following is numpy port of the error
                    # function provided by Ashesh Jain (in matlab), available at
                    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
                    idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]

                    euc_error = np.power(gt_i[:, idx_to_use] - eulerchannels_pred[:, idx_to_use], 2)
                    euc_error = np.sum(euc_error, 1)
                    euc_error = np.sqrt(euc_error)
                    mean_errors[i, :] = euc_error

                # This is simply the mean error over the N_SEQUENCE_TEST examples
                mean_mean_errors = np.mean( mean_errors, 0 )

                # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                print(f"{action: <16} |", end="")
                for ms in [1, 3, 7, 9, 13, 24]:
                    if args.seq_length_out >= ms + 1:
                        print(f" {mean_mean_errors[ms]:.3f} |", end="")
                    else:
                        print("   n/a |", end="")
                print()
            
            print()
            print(textwrap.dedent(f"""\
                                  ============================
                                  Global step:         {current_step: d}
                                  Learning rate:       {model.lr: .4f}
                                  Step-time (ms):     {step_time * 1000: .4f}
                                  Train loss avg:      {loss:.4f}
                                  ----------------------------
                                  Val loss:            {val_loss:.4f}
                                  srnn loss:           {srnn_loss:.4f}
                                  ============================
            """))

            torch.save(model, os.path.join(train_dir, f"model_{current_step}"))

            print()
            previous_losses.append(loss)

            # Reset global time and loss
            step_time, loss = 0.0, 0.0

            sys.stdout.flush()


def get_srnn_gts(actions: List[str], 
                 model: Seq2SeqModel, 
                 test_set: dict, 
                 data_mean: np.ndarray | torch.Tensor, 
                 data_std: np.ndarray | torch.Tensor, 
                 dim_to_ignore: np.ndarray | torch.Tensor | List[int], 
                 one_hot: bool, 
                 to_euler = True):
    
    """
    Get the ground truths for srnn's sequences, and convert to Euler angles.
    (the error is always computed in Euler angles).

    Args
        `actions`: a list of actions to get ground truths for.
        `model`: training model we are using (we only use the "get_batch" method).
        `test_set`: dictionary with normalized training data.
        `data_mean`: d-long vector with the mean of the training data.
        `data_std`: d-long vector with the standard deviation of the training data.
        `dim_to_ignore`: dimensions that we are not using to train/predict.
        `one_hot`: whether the data comes with one-hot encoding indicating action.
        `to_euler`: whether to convert the angles to Euler format or keep thm in exponential map

    Returns
        `srnn_gts_euler`: a dictionary where the keys are actions, and the values
            are the ground_truth, denormalized expected outputs of srnns's seeds.
    """
    srnn_gts_euler = {}

    for action in actions:

        srnn_gt_euler = []
        _, _, srnn_expmap = model.get_batch_srnn(test_set, action)

        # expmap -> rotmat -> euler
        for i in np.arange(srnn_expmap.shape[0]):
            denormed = data_utils.unNormalizeData(srnn_expmap[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot)

            if to_euler:
                for j in np.arange(denormed.shape[0]):
                    for k in np.arange(3, 97, 3):
                        denormed[j, k: k + 3] = data_utils.rotmat2euler(data_utils.expmap2rotmat(denormed[j, k: k + 3]))

            srnn_gt_euler.append(denormed)

        # Put back in the dictionary
        srnn_gts_euler[action] = srnn_gt_euler

    return srnn_gts_euler

def sample():
    """Sample predictions for srnn's seeds"""
    actions = define_actions(args.action)

    # === Create the model ===
    print(f"Creating {args.num_layers:d} layers of {args.size:d} units.")
    sampling = True
    model = create_model(actions, sampling)
    if not args.use_cpu:
        model = model.cuda()
    print("Model created")

    # Load all the data
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(actions, 
                                                                                        args.seq_length_in, 
                                                                                        args.seq_length_out, 
                                                                                        args.data_dir, 
                                                                                        not args.omit_one_hot)

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts(actions, 
                                   model, 
                                   test_set, 
                                   data_mean,
                                   data_std, 
                                   dim_to_ignore, 
                                   not args.omit_one_hot, 
                                   to_euler = False)
    
    srnn_gts_euler = get_srnn_gts(actions, 
                                  model, 
                                  test_set, 
                                  data_mean, 
                                  data_std, 
                                  dim_to_ignore, 
                                  not args.omit_one_hot)

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'
    try:
        os.remove(SAMPLES_FNAME)
    except OSError:
        pass

    # Predict and save for each action
    for action in actions:

        # Make prediction with srnn' seeds
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(test_set, action)

        encoder_inputs = torch.from_numpy(encoder_inputs).float()
        decoder_inputs = torch.from_numpy(decoder_inputs).float()
        decoder_outputs = torch.from_numpy(decoder_outputs).float()

        if not args.use_cpu:
            encoder_inputs = encoder_inputs.cuda()
            decoder_inputs = decoder_inputs.cuda()
            decoder_outputs = decoder_outputs.cuda()

        encoder_inputs = Variable(encoder_inputs)
        decoder_inputs = Variable(decoder_inputs)
        decoder_outputs = Variable(decoder_outputs)

        srnn_poses = model(encoder_inputs, decoder_inputs)

        srnn_loss = (srnn_poses - decoder_outputs) ** 2
        srnn_loss.cpu().data.numpy()
        srnn_loss = srnn_loss.mean()

        srnn_poses = srnn_poses.cpu().data.numpy()
        srnn_poses = srnn_poses.transpose([1, 0, 2])

        srnn_loss = srnn_loss.cpu().data.numpy()

        # denormalizes too
        srnn_pred_expmap = data_utils.revert_output_format(srnn_poses, 
                                                           data_mean, 
                                                           data_std, 
                                                           dim_to_ignore, 
                                                           actions, 
                                                           not args.omit_one_hot)

        # Save the samples
        with h5py.File(SAMPLES_FNAME, 'a') as hf:
            for i in np.arange(8):
                # Save conditioning ground truth
                node_name = f'expmap/gt/{action}_{i}'
                hf.create_dataset(node_name, data = srnn_gts_expmap[action][i])
                # Save prediction
                node_name = f'expmap/preds/{action}_{i}'
                hf.create_dataset(node_name, data = srnn_pred_expmap[i])

        # Compute and save the errors here
        mean_errors = np.zeros((len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]))

        for i in np.arange(8):
            eulerchannels_pred = srnn_pred_expmap[i]

            for j in np.arange(eulerchannels_pred.shape[0]):
                for k in np.arange(3, 97, 3):
                    eulerchannels_pred[j, k: k + 3] = data_utils.rotmat2euler(data_utils.expmap2rotmat(eulerchannels_pred[j, k: k + 3]))

            eulerchannels_pred[:, 0: 6] = 0

            # Pick only the dimensions with sufficient standard deviation. Others are ignored.
            idx_to_use = np.where(np.std(eulerchannels_pred, 0) > 1e-4)[0]

            euc_error = np.power(srnn_gts_euler[action][i][:, idx_to_use] - eulerchannels_pred[:, idx_to_use], 2)
            euc_error = np.sum(euc_error, 1)
            euc_error = np.sqrt(euc_error)
            mean_errors[i, :] = euc_error

        mean_mean_errors = np.mean( mean_errors, 0 )
        print(action)
        print(','.join(map(str, mean_mean_errors.tolist())))

        with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
            node_name = f'mean_{action}_error'
            hf.create_dataset(node_name, data = mean_mean_errors)

    return

def define_actions(action):
    """
    Define the list of actions we are using.

    Args
    action: String with the passed action. Could be "all"
    Returns
    actions: List of strings of actions
    Raises
    ValueError if the action is not included in H3.6M
    """

    actions = ["walking", "eating", "smoking", "discussion",  "directions",
                "greeting", "phoning", "posing", "purchases", "sitting",
                "sittingdown", "takingphoto", "waiting", "walkingdog",
                "walkingtogether"]

    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

    raise ValueError(f"Unrecognized action: {action}")

def read_all_data(actions: List[str], 
                  seq_length_in: int, 
                  seq_length_out: int, 
                  data_dir: str, 
                  one_hot: bool):
    """
    Loads data for training/testing and normalizes it.

    Args
        `actions`: list of strings (actions) to load
        `seq_length_in`: number of frames to use in the burn-in sequence
        `seq_length_out`: number of frames to use in the output sequence
        `data_dir`: directory to load the data from
        `one_hot`: whether to use one-hot encoding per action
    Returns
        `train_set`: dictionary with normalized training data
        `test_set`: dictionary with test data
        `data_mean`: d-long vector with the mean of the training data
        `data_std`: d-long vector with the standard dev of the training data
        `dim_to_ignore`: dimensions that are not used becaused stdev is too small
        `dim_to_use`: dimensions that we are actually using in the model
    """

    # === Read training data ===
    print (f"Reading training data (seq_len_in: {seq_length_in}, seq_len_out {seq_length_out}).")

    train_subject_ids = [1, 6, 7, 8, 9, 11]
    test_subject_ids = [5]

    train_set, complete_train = data_utils.load_data(data_dir, train_subject_ids, actions, one_hot)
    test_set, complete_test  = data_utils.load_data(data_dir, test_subject_ids, actions, one_hot)

    # Compute normalization stats
    data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

    # Normalize -- subtract mean, divide by stdev
    train_set = data_utils.normalize_data(train_set, data_mean, data_std, dim_to_use, actions, one_hot)
    test_set = data_utils.normalize_data(test_set, data_mean, data_std, dim_to_use, actions, one_hot)
    print("Done reading data.")

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

# Main Function
def main():
    if args.sample:
        sample()
    else:
        train()

if __name__ == "__main__":
    main()