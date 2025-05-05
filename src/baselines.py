from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from typing import List

import translate
import data_utils
from seq2seq import Seq2SeqModel

class Object(object):
    pass

def running_average(actions_dict: dict, 
                    actions: List[str], 
                    k: int):
    """
    Compute the error if we simply take the average of the last k frames.

    Args
        `actions_dict`: Dictionary where keys are the actions, and each entry has a
                      tuple of (enc_in, dec_in, dec_out) poses.
        `actions`: List of strings. The keys of actions_dict.
        `k`:Integer. Number of frames to use for running average.

    Returns
        `errs`: a dictionary where, for each action, we have a 100-long list with the
              error at each point in time.
    """

    # Get how many batches we have
    enc_in, dec_in, dec_out = actions_dict[actions[0]]

    n_sequences = len(enc_in)
    seq_length_out = dec_out[0].shape[0]

    errs = dict()

    for action in actions:

        # Make space for the error
        errs[action] = np.zeros((n_sequences, seq_length_out))

        # Get the lists for this action
        enc_in, dec_in, dec_out = actions_dict[action]

        for i in np.arange(n_sequences):

            # The last frame
            last_frame = dec_in[i][0, :]
            last_frame[0: 6] = 0

            if k > 1:
                # Get the last k-1 frames
                last_k = enc_in[i][(-k + 1):, :]
                assert(last_k.shape[0] == (k - 1))

                # Merge and average them
                avg = np.mean(np.vstack((last_k, last_frame)), 0)
            else:
                avg = last_frame

            dec_out[i][:, 0: 6] = 0
            idx_to_use = np.where(np.std(dec_out[i], 0) > 1e-4)[0]

            ee = np.power(dec_out[i][:, idx_to_use] - avg[idx_to_use], 2)
            ee = np.sum(ee, 1)
            ee = np.sqrt(ee)
            errs[action][i, :] = ee

        errs[action] = np.mean( errs[action], 0 )

    return errs

def denormalize_and_convert_to_euler(data: dict, 
                                     data_mean: np.ndarray, 
                                     data_std: np.ndarray, 
                                     dim_to_ignore: np.ndarray | List[int], 
                                     actions: List[str], 
                                     one_hot: bool):
    """
    Denormalizes data and converts to Euler angles
    (all losses are computed on Euler angles).

    Args
        `data`: dictionary with human poses.
        `data_mean`: d-long vector with the mean of the training data.
        `data_std`: d-long vector with the standard deviation of the training data.
        `dim_to_ignore`: dimensions to ignore because the std is too small or for other reasons.
        `actions`: list of strings with the actions in the data dictionary.
        `one_hot`: whether the data comes with one-hot encoding.

    Returns
        `all_denormed`: a list with nbatch entries. Each entry is an n-by-d matrix
                        that corresponds to a denormalized sequence in Euler angles
    """

    all_denormed = []

    # expmap -> rotmat -> euler
    for i in np.arange(data.shape[0]):
        denormed = data_utils.unNormalizeData(data[i, :, :], 
                                              data_mean, 
                                              data_std, 
                                              dim_to_ignore, 
                                              actions, 
                                              one_hot)

        for j in np.arange(denormed.shape[0]):
            for k in np.arange(3, 97, 3):
                denormed[j, k: k + 3] = data_utils.rotmat2euler(data_utils.expmap2rotmat(denormed[j, k: k + 3]))

        all_denormed.append(denormed)

    return all_denormed

def replicate_zero_velocity():
    actions = ["walking", "eating", "smoking", "discussion", 
               "directions", "greeting", "phoning", "posing", "purchases", 
               "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]
    
    # actions = ["walking", "eating", "smoking", "discussion"]

    # Parameters for dummy model. We only build the model to load the data.
    one_hot = False
    FLAGS = Object()
    FLAGS.data_dir = "../Data/H36M_Expmap"
    FLAGS.seq_length_in = 50
    FLAGS.seq_length_out = 100
    FLAGS.num_layers = 1
    FLAGS.size = 128
    FLAGS.max_gradient_norm = 5
    FLAGS.batch_size = 8
    FLAGS.learning_rate = 0.005
    FLAGS.learning_rate_decay_factor = 1
    FLAGS.loss_to_use = "sampling_based"
    FLAGS.bidirectional = False
    FLAGS.attention = False
    FLAGS.tied = True
    FLAGS.omit_one_hot = not one_hot
    FLAGS.residual_velocities = False
    dtype = torch.float32

    # Baselines are very simple. No need to use the GPU.
    model = Seq2SeqModel(
        source_seq_len = FLAGS.seq_length_in,
        target_seq_len = FLAGS.seq_length_out,
        number_of_actions = len(actions),
        rnn_size = FLAGS.size, # hidden layer size
        num_layers = FLAGS.num_layers,
        max_gradient_norm = FLAGS.max_gradient_norm,
        batch_size = FLAGS.batch_size,
        learning_rate = FLAGS.learning_rate,
        learning_rate_decay_factor = FLAGS.learning_rate_decay_factor,
        loss_to_use = FLAGS.loss_to_use,
        one_hot = not FLAGS.omit_one_hot,
        bidirectional = FLAGS.bidirectional,
        attention = FLAGS.attention,
        tied = FLAGS.tied,
        residual_velocities = FLAGS.residual_velocities,
        dtype = dtype)

    # Load the data
    _, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = translate.read_all_data(
        actions, 
        FLAGS.seq_length_in, 
        FLAGS.seq_length_out, 
        FLAGS.data_dir, 
        not FLAGS.omit_one_hot)

    # Get all the data, denormalize and convert it to euler angles
    poses_data = {}
    for action in actions:
        enc_in, dec_in, dec_out = model.get_batch_srnn(test_set, action)

        enc_in = denormalize_and_convert_to_euler(enc_in, 
                                                  data_mean, 
                                                  data_std, 
                                                  dim_to_ignore, 
                                                  actions, 
                                                  not FLAGS.omit_one_hot)
        
        dec_in = denormalize_and_convert_to_euler(dec_in, 
                                                  data_mean, 
                                                  data_std, 
                                                  dim_to_ignore, 
                                                  actions, 
                                                  not FLAGS.omit_one_hot)
        
        dec_out = denormalize_and_convert_to_euler(dec_out, 
                                                   data_mean, 
                                                   data_std, 
                                                   dim_to_ignore, 
                                                   actions, 
                                                   not FLAGS.omit_one_hot)

        poses_data[action] = (enc_in, dec_in, dec_out)

    # Compute baseline errors
    const = running_average(poses_data, actions, 1)
    ave_2 = running_average(poses_data, actions, 2)
    ave_4 = running_average(poses_data, actions, 4)

    print()
    print("=== Zero-velocity ===")
    print(f"{"milliseconds": <16} | {80:4d} | {160:4d} | {320:4d} | {400:4d}")
    for action in actions:
        print(f"{action: <16} | {const[action][1]:.2f} | {const[action][3]:.2f} | {const[action][7]:.2f} | {const[action][9]:.2f}")

    print()
    print("=== Runnning avg. 2 ===")
    print(f"{"milliseconds": <16} | {80:4d} | {160:4d} | {320:4d} | {400:4d}")
    for action in actions:
        print(f"{action: <16} | {ave_2[action][1]:.2f} | {ave_2[action][3]:.2f} | {ave_2[action][7]:.2f} | {ave_2[action][9]:.2f}")

    print()
    print("=== Runnning avg. 4 ===")
    print(f"{"milliseconds": <16} | {80:4d} | {160:4d} | {320:4d} | {400:4d}")
    for action in actions:
        print(f"{action: <16} | {ave_4[action][1]:.2f} | {ave_4[action][3]:.2f} | {ave_4[action][7]:.2f} | {ave_4[action][9]:.2f}")

def replicate_residuals():
    pass

def main():
    pass

if __name__ == "__main__":
    replicate_zero_velocity()