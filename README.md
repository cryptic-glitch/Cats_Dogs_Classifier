Binary Classification
===

# Introduction
The project uses pretrained Deep Learning neural networks in TF2 for the binary classification of cats and dogs. Metrics(accuracy) are obtained as aan average of 5 fold cross validation.  


# Usage
**training**

With the given training script, we can train the several models along with several learning rates and batch sizes as a pandas dataframe. 

**Batch sizes, Model architectures & learning rates should be a list of int, string and float respectively**

```gherkin=
# Examplary CLI instructions
python train.py --imdir "path/to/the/imagedir" --lr [0.001, 0.0001] --m ["MobileNetV3Small", "EfficientNetV2B2"] --bs [18, 24] --epoch 5
```
**where**
```
rimdir = PATH TO IMAGEDIR. The folder contains combined images of both Cats and Dogs.
However, there are some erronous files which are not images and should be fileterd
out before running the training script.

lr = Learning rate
m = Model architecture
bs = Batch size
epoch = No. of epochs to train
```
# Inference
Model is released here: https://github.com/cryptic-glitch/Cats_Dogs_Classifier/releases/download/ckpt/checkpoint_InceptionV3.zip
After downloading the model, follow the instructions.
CLI instructions 
```gherkin=
python inference.py path/to/image path/to/model
```

# Augmentations
I used the following augmentations:
* random horizontal flip
* rescale
* random rotation upto 10 degrees.

While optical augmentations could also be used, I decided to limit to just the ones above.

# Results
To save training time from all possible combinations, all the results below were computed for the learning rate of 0.0001. Also, they were trained just for 4 epochs with fixed image sizes of (256 x 256).
As a rule of thumb, it is always preferable to use shallower networks unless absolutely necessary since bigger neural networks are harder to train.

| Architecture     | bs | Accuracy |
|------------------|----|----------|
| InceptionV3      | 18 |   0.9859 |
| MobileNetV3Small | 18 |   0.8658 |
| EfficientNetV2B2 | 18 |   0.9298 |
| VGG16            | 18 |   0.9677 |

![](https://i.imgur.com/b8A3hbh.png)



Along with them, with a batch size of 24, the metrics are -

| Architecture     | bs | Accuracy |
|------------------|----|----------|
| InceptionV3      | 24 |   0.9862 |
| MobileNetV3Small | 24 |   0.8687 |
| EfficientNetV2B2 | 24 |   0.9219 |
| VGG16            | 24 |   0.9508 |

![](https://i.imgur.com/6nFZJLD.png)

**The criterion for saving the checkpoints has been val-loss**

# Conclusion and further outlooks
1) Since the inter class variance is quite high(due to distintice features of the binary classes), shallow models were able to train themselves with high accuracy. 
2) The model can be further improved with higher batch sizes, high number of epochs along with the inclusion of more robust augmenentations and finally the addition of scheduler. Since, I trained it for just 4 epochs, scheduler was a bit unwanted, however, I would incorporate if I train the model for more epochs. 


