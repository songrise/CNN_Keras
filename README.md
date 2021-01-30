# CNN_Keras
**(still updating)**  
My Keras implementation of famous CNN models, in chronological order   

Some of the implementation may contains minor difference (both intentionally and unintentionally) from original structure proposed by their designers. Please contact me if it cause any confusion.  


## CNN
- [LeNet-5(1998)](src/LeNet-5.py)  
- [AlexNet(2012)](src/AlexNet.py)  
- [GoogLeNet(2014)](src/GoogLeNet.py)  
- [VGG-16(2014)](src/VGG-16.py)
- [ResNet-18(2015)](src/ResNet-18.py)

## FCN
- [FCN-8s](src/FCN-8s.py)

Environment:
Keras 2.4.3
Tensorflow 2.2.0


Todo list:  
- shift to Keras subclass api (currently, only complex model such as ResNet is using subclass api)
- refactor for better readability
