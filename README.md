# GRU-Human-Motion-Predictions-PyTorch
This is a modified PyTorch implementation of the paper (modified from [@enriccorona](https://github.com/enriccorona/human-motion-prediction-pytorch/tree/master))

Julieta Martinez, Michael J. Black, Javier Romero.
_On human motion prediction using recurrent neural networks_. In CVPR 17.

It can be found on arxiv as well: https://arxiv.org/pdf/1705.02445.pdf

The code in the original repository was written by [Julieta Martinez](https://github.com/una-dinosauria/) and [Javier Romero](https://github.com/libicocco/) and is accessible [here](/blob/master/src/translate.py).

### Dependencies

* [h5py](https://github.com/h5py/h5py) -- to save samples
* [Pytorch](https://pytorch.org/)

### Get this code and the data

Note that the following human3.6m dataset on exponential map format is not accessible yet.

```bash
git clone https://github.com/enriccorona/human-motion-prediction-pytorch.git
cd human-motion-prediction-pytorch
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```

In order to obtain the dataset, you could apply from [Human 3.6M Dataset](http://vision.imar.ro/human3.6m/description.php), or refer to [MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation](https://github.com/Vegetebird/MHFormer) for preprocessed data and use functions in `data_utils.py` to get exponential maps.

### Quick demo and visualization

The code in this fork should work exactly as in the original repo:

For a quick demo, you can train for a few iterations and visualize the outputs of your model.

To train, run
```bash
python src/translate.py --action walking --seq_length_out 25 --iterations 10000
```

To save some samples of the model, run
```bash
python src/translate.py --action walking --seq_length_out 25 --iterations 10000 --sample --load 10000
```

Finally, to visualize the samples run
```bash
python src/forward_kinematics.py
```

This should create a visualization similar to this one

<p align="center">
  <img src="https://raw.githubusercontent.com/una-dinosauria/human-motion-prediction/master/imgs/walking.gif"><br><br>
</p>


### Running average baselines

To reproduce the running average baseline results from our paper, run

`python src/baselines.py`

### RNN models

To train and reproduce the results of our models, use the following commands

| model      | arguments | training time (gtx 1080) | notes |
| ---        | ---       | ---   | --- |
| Sampling-based loss (SA) | `python src/translate.py --action walking --seq_length_out 25` | 45s / 1000 iters | Realistic long-term motion, loss computed over 1 second. |
| Residual (SA)            | `python src/translate.py --residual_velocities --action walking` | 35s / 1000 iters |  |
| Residual unsup. (MA)     | `python src/translate.py --residual_velocities --omit_one_hot` | 65s / 1000 iters | |
| Residual sup. (MA)       | `python src/translate.py --residual_velocities` | 65s / 1000 iters | best quantitative.|
| Untied       | `python src/translate.py --residual_velocities --untied` | 70s / 1000 iters | |


You can substitute the `--action walking` parameter for any action in

```
["directions", "discussion", "eating", "greeting", "phoning",
 "posing", "purchases", "sitting", "sittingdown", "smoking",
 "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
```

or `--action all` (default) to train on all actions.

### More Achitectures

More flags are introduced to enable more choices for achitectures

|model                                            |flags                       |
|-------------------------------------------------|----------------------------|
|teacher forcing (instead of sampling-based loss) | `--loss_to_use supervised` |
|untied (not share weights)                       | `--untied`                 |
|bidirectional encoder                            | `--untied --bidirectional` |
|stacked RNNs (for any number > 1)                | `--num_layers 2`           |
|attention                                        | `--attention`              |

Note that 
- You can substitute the `--num_layer` parameter for any integer that is greater than 1.
- If you need a bidirectional encoder, be sure to also add flag `untied` since the model is by default sharing weights between the encoder and the decoder.
### Citing

If you use our code, please cite our work

```
@inproceedings{julieta2017motion,
  title={On human motion prediction using recurrent neural networks},
  author={Martinez, Julieta and Black, Michael J. and Romero, Javier},
  booktitle={CVPR},
  year={2017}
}
```

### Acknowledgments

The pre-processed human 3.6m dataset and some of our evaluation code (specially under `src/data_utils.py`) was ported/adapted from [SRNN](https://github.com/asheshjain399/RNNexp/tree/srnn/structural_rnn) by [@asheshjain399](https://github.com/asheshjain399).

### Licence
MIT
