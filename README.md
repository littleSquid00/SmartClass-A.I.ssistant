# COMP-472-Artificial Intelligence

Team Name - OB_14

Team Members - Chit Chit Myet Cheal Zaw (40110140)
               Mehdi Chitsaz (40132819)
               Nadav Friedman (40091001)


### Project Structure
1. `OriginalityForm` folder:
    Contains Expectations of Originality forms, signed by each student, which attest to the originality of the work.
2. `data` folder:
    The Dataset containing all the images. The images are split into different classes (Angry, Happy, Neutral, and Focused), each of which is further split into training and test folders. More information regarding the Dataset can be found in the `DatasetExplanation.txt` file.
3. `Report.pdf` file:
    Report detailing our findings, methodology, and analysis for project part 1.
4. `DatasetExplanation.txt` file:
    File detailing the provenance of each dataset/image.
5. `DataVisualization.py`:
    Python code which handles all data visualization. It displays a bar graph showing the number of images in each class, tt plots the aggregated pixel intensity distribution for each of the four classes, and it displays a sample of 15 images with their respective pixel intensity histograms.
   
### Executing the code

#### Clean the data

Starting with an image (i.e: a `source.jpg` for demonstration purposes), do the following:

1. Crop the image: using Photoshop for instance
2. Turn the image into grayscale: by running the following on Linux (ImageMagick is required) `mogrify -colorspace Gray source.jpg`
3. Resize the image (to 48x48px for example): by running the following on Linux (ImageMagick is required) `mogrify -resize 48x48! source.jpg`

Note that the dataset provided in this repository has already been cleaned.

#### Data Visualization

When in the root directory of the project, run the following in the terminal (python >= 3.10 may be required):

1. `python -m venv venv`
2. `. venv/bin/activate` on linux or `.\venv\Scripts\activate` on windows
3. `pip install -r requirements.txt`
4. `python DataVisualization.py`

### Training

When in the root directory of the project, run the following in the terminal (python >= 3.10 may be required) to train all 3 models:

`python code/train_model.py`

This script trains each model for a maximum of 20 epochs and uses early stopping if the validation loss does not improve for 5 consecutive epochs. It saves the best model found in a .pth file in the root directory.

### Evaluation

#### Evaluation on Test Dataset
To evaluate the performance of each of the three models on the test dataset, run the following in the terminal:

`python code/evaluate_model.py`

Note that a confusion matrix plot is created for each model in this process and to move onto the evaluation of the next model, you need to close the generated plot.

#### Evaluation of Specific Image using A Model

To predict the class corresponding to a given image, edit the predict_image.py file in the following 2 ways:
- Line 8: Enter the path to the saved model parameters of the model you want to use (i.e. <model_name>_best_model.pth)
- Line 52: Change the image_path variable to the path of the image you want to predict the class of.

Then run the following in the terminal: 
`python code/predict_image.py`





