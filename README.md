# Yoga Pose Recognition using Convolutional Neural Networks

## Introduction
This project aims to recognize yoga poses using Convolutional Neural Networks (CNNs). It utilizes image data of various yoga poses and trains a CNN model to classify them accurately.

## Dataset
The dataset consists of images of different yoga poses, including Adho Mukha Svanasana, Adho Mukha Vrksasana, Alanasana, Anjaneyasana, and Ardha Chandrasana. These images are preprocessed and resized to a standard size before being used for training.

## Model Architecture
The CNN model architecture comprises several convolutional and pooling layers followed by fully connected layers. Data augmentation techniques such as random flipping, rotation, and zooming are applied to improve model generalization.

## Training
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. The training data is split into training and validation sets, and the model is trained for a specified number of epochs.

## Evaluation
The trained model is evaluated using a separate test set to assess its performance in recognizing yoga poses. Metrics such as loss and accuracy are computed to evaluate the model's effectiveness.

## Results
The model achieves close to 80% accuracy on the test set, demonstrating its ability to classify yoga poses accurately.

## Usage
To use the model for inference, simply provide an image of a yoga pose to the trained model, and it will predict the corresponding pose.

## Conclusion
This project demonstrates the effectiveness of CNNs in recognizing yoga poses. The trained model can be used as a tool to assist yoga practitioners in improving their posture and form.
