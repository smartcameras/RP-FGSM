# Robust Private FGSM (RP-FGSM)

## Introduction
This is the official repository of Robust Private FGSM (RP-FGSM), a work published as [*Exploiting Vulnerabilities of Deep Neural Networks for Privacy Protection*](https://ieeexplore.ieee.org/document/9069287) in IEEE Transactions on Multimedia.

| Original Image | Adversarial image |
|---|---|
| ![Original Image](https://github.com/RiSaMa/RP-FGSM/blob/master/example/clean.png) | ![Adversarial Image](https://github.com/RiSaMa/RP-FGSM/blob/master/example/adv.png) |
|Church|Swimming pool|

## Requirements
 - Conda
 - Python 3.7
 - Numpy
 - PyTorch
 - Torchvision
 - Opencv-python
 - Tqdm

The code has been tested on Ubuntu 18.04 and MacOs 10.15.5.

## Setup

Install miniconda: https://docs.conda.io/en/latest/miniconda.html
Create conda environment for Python 3.7
```
conda create -n rpfgsm python=3.7
```
Activate conda environment:
```
source activate rpfgsm
```
Install requirements
```
pip install -r requirements.txt
```
**Only if using MacOs**:
```
export PATH="<pathToMiniconda>/bin:$PATH"
brew install wget
```

## Download models
The pre-trained models will download automatically on the first execution of the code.

 - [ResNet50](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar) 
 - [ResNet18](http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar)
 - [AlexNet](http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar)
 - [DenseNet161](http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar)


## Generate adversarial images
Generate adversarial images executing:
```
python rp-fgsm.py --model=<modelName> --eps=<epsilonValue> --gamma=<gammaValue>
```
For example:
```
python rp-fgsm.py --model=resnet50 --eps=16/255 --gamma=0.99
```
\<modelName\> can be one of: resnet50, resnet18, alexnet, densenet161.
 
\<epsilonValue\> should be in fraction as x/255, x = 1,2,...,255.
 
\<gammaValue\> should be in decimal, between 0 and 1.

## Output and format
The adversarial images and the log file are stored in 'results/' folder.

The image is in 'results/adv_<modelName>_eps<epsilonValue>_gamma<gammaValue>' folder.

Classification results are in log_<modelName>_eps<epsilonValue>_gamma<gammaValue>.txt, in the following order of columns:
* image name
* original class
* original class probability
* target class
* target class probability
* final class
* final class probability

Runtime results are in logTimes_<modelName>_eps<epsilonValue>_gamma<gammaValue>.txt, in the following order of columns:
* image name
* runtime (millisecond) 

## Authors
* [Ricardo Sanchez-Matilla](mailto:ricardo.sanchezmatilla@qmul.ac.uk),
* [Chau Yi Li](mailto:chauyi.li@qmul.ac.uk), 
* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk),
* [Riccardo Mazzon](mailto:r.mazzon@qmul.ac.uk), and
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk).

## References
If you use our code, please cite the following publication:

R. Sanchez-Matilla, C. Y. Li, A. S. Shamsabadi, R. Mazzon and A. Cavallaro, "Exploiting Vulnerabilities of Deep Neural Networks for Privacy Protection," in IEEE Transactions on Multimedia, vol. 22, no. 7, pp. 1862-1873, July 2020.

    @ARTICLE{SanchezMatilla2020_RPFGSM,
      author={R. Sanchez-Matilla and C. Y. Li and A. Shahin Shamsabadi and R. Mazzon and A. Cavallaro},
      journal={IEEE Transactions on Multimedia}, 
      title={Exploiting vulnerabilities of deep neural networks for privacy protection}, 
      volume={22},
      number={7},
      month={July},
      year={2020},
      pages={1862-1873},
      }

## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).
