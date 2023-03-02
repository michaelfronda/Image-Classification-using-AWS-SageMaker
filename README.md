# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
<img width="638" alt="image" src="https://user-images.githubusercontent.com/46351476/222314463-800a27a0-3157-4f85-a886-efe7fb35f928.png">

## Model Selection
The repo contains a notebook "train_and_deploy.ipynb" that implements a multi class image classification DL model to classify between 133 different species of dogs. 

The model itself adds a input layer, a fully connected layer and output layer to a pretrained Resnet50 model from the pytorch vision library. This utilizes transfer learning by freezing the convultional layers in the Resnet50 model. 


We will be using a pretrained Resnet50 model from pytorch vision library.
We will be adding in two Fully connected Neural network Layers on top of the above Resnet50 model.
Note: We will be using concepts of Transfer learning and so we will be freezing all the exisiting Convolutional layers in the pretrained resnet50 model and only changing gradients for the tow fully connected layers that we have added.
Then we will perform Hyperparameter tuning, to help figure out the best hyperparameters to be used for our model.
Next we will be using the best hyperparameters and fine-tuning our Resent50 model.
We will also be adding in configuration for Profiling and Debugging our training mode by adding in relevant hooks in the Training and Testing( Evaluation) phases.
Next we will be deploying our model. While deploying we will create our custom inference script. The custom inference script will be overriding a few functions that will be used by our deployed endpoint for making inferences/predictions.
Finally we will be testing out our model with some test images of dogs, to verfiy if the model is working as per our expectations.


## Hyperparameter Tuning
I needed up tuning the parameters of learning rate, epochs and batch size. Learning rate determines how fast or slow we will move towards the optimal weights, batch size controls the ability to generalize based on the samples propogated through the syste and finally epochs to control the number of times the model works through the entire training set. Here we retrieve the best results and use to train

<img width="885" alt="image" src="https://user-images.githubusercontent.com/46351476/222315966-23d0b526-7594-4283-8996-6dfee1aba5ac.png">


## Debugging and Profiling
I set Debugger hook to essentially keep track of records of Loss Criterion metrics in the model training as well as the validation/testing phases. The Plot of the Cross entropy loss is shown here :

<img width="839" alt="image" src="https://user-images.githubusercontent.com/46351476/222316097-caa1a05f-ed70-4076-9e87-794454de6972.png">

We can see anomalous behavior in the graph, we can fix this: 
- Adjust the pretrained model or experiment with additional FC layers which ideally would help smooth out the curve and reduce variance

## Cloudwatch Metrics for Endpoint
<img width="760" alt="image" src="https://user-images.githubusercontent.com/46351476/222316784-6deacc95-3aac-4724-aec6-8501c03c869f.png">

## Model Deployment
In the end a "ml.m5.large" instance type was used to deploy the endpoint along with "inference.py" script for setup. We can see the endpoint here:

<img width="1580" alt="image" src="https://user-images.githubusercontent.com/46351476/222317373-e9002a67-3d69-4fd1-b857-668adcef2540.png">

For testing I tested the model against three images of the Alaskan Malamute or class 5 breed of dog. 

<img width="356" alt="image" src="https://user-images.githubusercontent.com/46351476/222317130-c19186c5-15d3-479a-a219-5c4ec354bec5.png">

We can see the model was trained correctly and correctly classifies the dog breed! 
