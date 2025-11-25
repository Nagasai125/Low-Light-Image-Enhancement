

# Zero-IG Framework

<img src="Figs/Fig3.png" width="900px"/> 
<p style="text-align:justify">Note that the provided model in this code are not the model for generating results reported in the paper.

## Model Training Configuration
* To train a new model, specify the dataset path in "train.py" and execute it. The trained model will be stored in the 'weights' folder, while intermediate visualization outputs will be saved in the 'results' folder.
* We have provided some model parameters, but we recommend training with a single image for better result.

## Requirements
* Python 3.7
* PyTorch 1.13.0
* CUDA 11.7
* Torchvision 0.14.1

## Testing
* Ensure the data is prepared and placed in the designated folder.
* Select the appropriate model for testing, which could be a model trained by yourself.
* Execute "test.py" to perform the testing.





## License

This implementation is provided under the MIT License. See LICENSE file for details.

