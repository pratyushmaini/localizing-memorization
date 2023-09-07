# Can Neural Network Memorization Be Localized?

This is the code repository for the ICML 2023 paper "Can Neural Network Memorization Be Localized?" by Pratyush Maini, Michael C. Mozer, Hanie Sedghi, Zachary C. Lipton, J. Zico Kolter and Chiyuan Zhang.

`This repository is still under construction. We will be adding more code and documentation soon.`

## Abstract

Recent efforts at explaining the interplay of memorization and generalization in deep overparametrized networks have posited that neural networks memorize “hard” examples in the final few layers of the model. In this work, we show that rather than being confined to individual layers, memorization is a phenomenon confined to a small set of neurons in various layers of the model. 

First, via three experimental sources of converging evidence, we find that most layers are redundant for the memorization of examples and the layers that contribute to example memorization are, in general, not the final layers. 

Second, we ask a more generic question: can memorization be localized anywhere in a model? We discover that memorization is often confined to a small number of neurons or channels (around 5) of the model. 

Based on these insights we propose a new form of dropout—example-tied dropout that enables us to direct the memorization of examples to an a priori determined set of neurons. By dropping out these neurons, we are able to reduce the accuracy on memorized examples from 100% → 3%, while also reducing the generalization gap.

## Usage

The code requires Python 3.10 and PyTorch1.8+. 

The main scripts to run are:

- `experiments/gradient_accounting.py` - Computes gradient norms for clean and noisy examples
- `experiments/layer_rewinding.py` - Evaluates model accuracy when rewinding layers 
- `experiments/layer_retraining.py` - Retrains individual layers on clean examples
- `experiments/neuron_flipping` - Finds the minimum neurons to flip an example's prediction
- `models/dropout.py` - Implements the proposed dropout method



## Citation

If you find this repository useful, please cite the paper:

```
@inproceedings{maini2023memorization,
  title={Can Neural Network Memorization Be Localized?},
  author={Maini, Pratyush and Mozer, Michael C and Sedghi, Hanie and Lipton, Zachary C and Kolter, J Zico and Zhang, Chiyuan},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## Contact

For any questions, please contact Pratyush Maini at pratyushmaini@cmu.edu.