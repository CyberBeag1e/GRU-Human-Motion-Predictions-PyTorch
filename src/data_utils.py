from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
from typing import List

def rotmat2euler(R: np.ndarray):
    """
    Converts a rotation matrix to Euler angles. 
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

    Args
        `R`: a 3x3 rotation matrix
    Returns
        `euler`: a 3x1 Euler angle representation of R
    """
    if R[0, 2] == 1 or R[0, 2] == -1:
        E3 = 0
        delta = np.arctan2(R[0, 1], R[0, 2])

        if R[0, 2] == -1:
            E2 = np.pi / 2
            E1 = E3 + delta
        else:
            E2 = -np.pi / 2
            E1 = -E3 + delta
    
    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))
    
    euler = np.array([E1, E2, E3])

    return euler


def quat2expmap(q: np.ndarray):
    """
    Converts a quaternion to an exponential map. 
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Args
        `q`: 1x4 quaternion
    Returns
        `r`: 1x3 exponential map
    Raises
        `ValueError` if the l2 norm of the quaternion is not close to 1
    """
    
    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise ValueError("quatexpmap: input quaternion is not norm 1.")

    sin_half_theta = np.linalg.norm(q[1: ])
    cos_half_theta = q[0]

    r0 = np.divide(q[1: ], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sin_half_theta, cos_half_theta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0
    
    r = r0 * theta
    return r

def rotmat2quat(R: np.ndarray):
    """
    Converts a rotation matrix to a quaternion. 
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Args
        `R`: 3x3 rotation matrix
    Returns
        `q`: 1x4 quaternion
    """
    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] =  rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sin_theta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    cos_theta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sin_theta, cos_theta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)

    return q

def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))

def expmap2rotmat(r: np.ndarray):
    """
    Converts an exponential map angle to a rotation matrix. 
    Matlab port to python for evaluation purposes. 
    I believe this is also called Rodrigues' formula: 
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
        `r`: 1x3 exponential map
    Returns
        `R`: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
    r0x = r0x - r0x.T
    R = np.eye(3,3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x)

    return R


def unNormalizeData(normalizedData: np.ndarray, 
                    data_mean: np.ndarray, 
                    data_std: np.ndarray, 
                    dimensions_to_ignore: np.ndarray, 
                    actions: List[str], 
                    one_hot: bool):
    """
    Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

    Args
        `normalizedData`: nxd matrix with normalized data
        `data_mean`: vector of mean used to normalize the data
        `data_std`: vector of standard deviation used to normalize the data
        `dimensions_to_ignore`: vector with dimensions not used by the model
        `actions`: list of strings with the encoded actions
        `one_hot`: whether the data comes with one-hot encoding
    Returns
        `origData`: data originally used to
    """
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype = np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if one_hot:
        origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
    else:
        origData[:, dimensions_to_use] = normalizedData

    # potentially inefficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis = 0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis = 0)
    origData = np.multiply(origData, stdMat) + meanMat

    return origData

def revert_output_format(poses: np.ndarray, 
                         data_mean: np.ndarray, 
                         data_std: np.ndarray, 
                         dim_to_ignore: np.ndarray, 
                         actions: List[str], 
                         one_hot: bool):
    """
    Converts the output of the neural network to a format that is more easy to
    manipulate for, e.g. conversion to other format or visualization

    Args
        `poses`: The output from the TF model. A list with (seq_length) entries,
        each with a (batch_size, dim) output
    Returns
        `poses_out`: A tensor of size (batch_size, seq_length, dim) output. Each
        batch is an n-by-d sequence of poses.
    """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in range(poses_out.shape[0]):
        poses_out_list.append(unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list

def readCSVasFloat(filename: str) -> np.ndarray:
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
        `filename`: string. Path to the csv file
    Returns
        `returnArray`: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def load_data(path_to_dataset: str, 
              subjects: List[int], 
              actions: List[str], 
              one_hot: bool):
    """
    Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files: 
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

    Args
        `path_to_dataset`: string. directory where the data resides
        `subjects`: list of numbers. The subjects to load
        `actions`: list of string. The actions to load
        `one_hot`: Whether to add a one-hot encoding to the data
    Returns
        `trainData`: dictionary with k:v
            k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
        `completeData`: nxd matrix with all the data. Used to normlization stats
    """
    nactions = len(actions)

    trainData = {}
    complete_data = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]

            for subact in [1, 2]:  # subactions
                print(f"Reading subject {subj}, action {action}, subaction {subact}")
                filename = f'{path_to_dataset}/S{subj}/{action}_{subact}.txt'
                action_sequence = readCSVasFloat(filename)

                n, d = action_sequence.shape
                even_list = range(0, n, 2)

                if one_hot:
                    # Add a one-hot encoding at the end of the representation
                    the_sequence = np.zeros((len(even_list), d + nactions), dtype = float)
                    the_sequence[:, 0: d] = action_sequence[even_list, :]
                    the_sequence[:, d + action_idx] = 1
                    trainData[(subj, action, subact, 'even')] = the_sequence
                else:
                    trainData[(subj, action, subact, 'even')] = action_sequence[even_list, :]

                complete_data.append(action_sequence)
    
    complete_data = np.vstack(complete_data)

    return trainData, complete_data


def normalize_data(data, 
                   data_mean: np.ndarray, 
                   data_std: np.ndarray, 
                   dim_to_use: np.ndarray, 
                   actions: List[str], 
                   one_hot: bool):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and dividing by the standard deviation

    Args
        `data`: nx99 matrix with data to normalize
        `data_mean`: vector of mean used to normalize the data
        `data_std`: vector of standard deviation used to normalize the data
        `dim_to_use`: vector with dimensions used by the model
        `actions`: list of strings with the encoded actions
        `one_hot`: whether the data comes with one-hot encoding
    Returns
        `data_out`: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
    # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
    # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[key] = np.divide((data[key][:, 0: 99] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:,-nactions:]))

    return data_out


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
        `completeData`: nx99 matrix with data to normalize
    Returns
        `data_mean`: vector of mean used to normalize the data
        `data_std`: vector of standard deviation used to normalize the data
        `dimensions_to_ignore`: vector with dimensions not used by the model
        `dimensions_to_use`: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis = 0)
    data_std = np.std(completeData, axis = 0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use