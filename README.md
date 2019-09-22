# CIFAR10 Classifier using Pytorch

CNN's are used to build a classifier for CIFAR10 dataset.

For Training:
```
python train.py --epochs=number_of_epochs_required --learning_rate=learning_rate_required
```
Model is now saved to the current working directory.

Default number of epochs is 5 and default learning rate is 0.003. Adam Optimizer is used.

For Testing:
```
python test.py
```
