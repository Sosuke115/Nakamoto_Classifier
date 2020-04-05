
# Nakamoto_Classifier
 
A system that can classify Nakamoto ramen using a deep neural network model created from scratch

In this system, you can choose "Alexnet","VGG11" or "simplenet" as a CNN model.
 
# DEMO
 
![demo](https://github.com/Sosuke115/Nakamoto_Classifier/blob/master/image.jpg "サンプル")


 
# Requirement
 
 
* Pillow 6.0.0
* numpy 1.16.2
 

# Usage


 
 
```bash
git clone https://github.com/Sosuke115/Nakamoto_Classifier.git
cd Nakamoto_Classifier
cp hogehoge/your_nakamoto.jpg judge_data
python judge.py
```
 
# Note
 
if you train the model by yourself, please augment data with reference to data_augmentation.ipynb.

```bash
cd Nakamoto_Classifier
python train.py --file_path dataset --model alexnet
```

# Author
 
* Sosuke
* website : https://sosuke115.github.io
* Twitter : https://twitter.com/ponyo_ponyo115


 

 
Thank you!
