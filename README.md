# Concrete_Crack_Image_Classification_TransferLearning

## Project Description

This project aims to leverage artificial intelligence to revolutionize the process of identifying and classifying cracks in concrete structures, a critical aspect of civil engineering. As an AI engineer, you will develop an image classification model that can accurately detect and categorize various types of cracks, such as hairline, shrinkage, settlement, and structural cracks.

The impact of this project is significant. By enabling rapid and precise identification of concrete cracks, we can ensure timely intervention, enhancing the safety and durability of buildings. This project is not just about technological innovation; it’s about creating a safer future. Your work could potentially save thousands of lives

## Data Description

The dataset utilized for this project is sourced from the comprehensive collection titled [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

Data Example:
<img src="/repository/resources/data_example.png" alt="Concrete Crack" title="Concrete Crack Dataset Example">

## Results

<img src="/repository/results/output_evaluation.png" alt="Output Model Evaluation" title="Output Model Evaluation">
<img src="/repository/results/output_graph.png" alt="Output Model Graph" title="Output Model Performance Graph">
<img src="/repository/results/output_deployment.png" alt="Output Model Deployment" title="Model Deployment">


## Workflow

### 1. Setup
> First and foremost, set up our development environment. This involves the installation and importation of necessary libraries that provide the tools and functionalities required for our tasks.

### 2. Data Loading
> The datasets are downloaded and pasted into the same directory as the main Python file. Dataset is loaded to the project using tensorflow keras `utils.image_dataset_from_directory`

### 3. Data Visualization
> Generate visual representations of a selection of images from the dataset, providing us with tangible examples and a deeper understanding of the data we are working with.

### 4. Validation-Test Splits
> We will proceed to partition the validation dataset into distinct subsets for `validation` and `testing`. This step is crucial as it allows us to evaluate the performance of our model on unseen data, ensuring its robustness and reliability.

### 5. Convert the dataset type
>  The primary reason for this conversion is to ensure that the GPU is kept busy and used to its full potential during training.

### 6. Data Augmentation (SKIPPED)
> In my cases, I would exclude or skip the data augmentation step because I'm using Tensorflow 2.10. In TensorFlow 2.10, there are known issues with data augmentation layers that result in warnings during execution. These warnings do not necessarily mean that the code will not work, but they indicate that certain operations are not optimized. I've commented on this step and you can refer to it in the `Main.py`.

### 7. Data Normalization
> Here, we instantiate a preprocessing layer to normalize input data for compatibility with the MobileNetV2 model

### 8. Transfer Learning
> We're using transfer learning which is using a pre-trained model for feature extraction, approach for this project. The model selected is MobileNetV2. After we load the pre-trained model, we will set the base_model to become non-trainable. You can also refer to the [MobileNetV2 Architecture](https://iq.opengenus.org/mobilenetv2-architecture/) to look more.

### 9. Classification Head
> To derive predictions from the feature block, we employ a `tf.keras.layers.GlobalAveragePooling2D` layer. This layer operates by averaging the features across each 5x5 spatial location. The result of this operation is a transformation of the features into a single vector per image.

### 10. Pipeline
> Build the entire model pipeline chaining together all the layers as in the pictures below:

<img src="/repository/resources/model_architecture.png" alt="Pipeline Architecture" title="Pipeline Architecture">

### 11. Model Compilation
> Normal drill, we will prepare all the callbacks and optimizer to compile the model.

### 12. Evaluation Before Training
> We would like to see the performance of our pre-trained model before training:

<img src="/repository/resources/evaluation_before_training.png" alt="Evaluation Before Training" title="Evaluation Before Training">

### 13. Model Training
> As usual, we will train the model by calling the `model.fit()`. The training report is as per below:

<img src="/repository/resources/base_model_training_report.png" alt="Training Report" title="Training Report">


### 14. Fine Tuning
> While the model is already achieving the desired performance with an accuracy exceeding 90%, there exists an opportunity to enhance this performance further. This can be accomplished by training or “fine-tuning” the layers of the model. It is done by unfreezing the `base_model` and setting the bottom layers to be un-trainable, followed by recompiling the model.

### 15. Fine Tune Model Compilation
> This will be repeated step as model compilation, the only difference is we will be using `RMSprop` optimizer

### 16. Fine Tune Model Training
> As usual, we will be training the fine tune model

### 17. Fine Tune Model Evaluation
> Print out the `test_loss` and `test_accuracy` for the fine-tune model and plot the performance graph of the base model and fine-tune model, as on the Results section on top


### 19. Model Deployment
> We will print and display the image with the predictions and labels as in Results section on top












