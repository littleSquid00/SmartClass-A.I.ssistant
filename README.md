# COMP-472-Artificial Intelligence

Team Name - OB_14

Team Members - Chit Chit Myet Cheal Zaw (40110140)
               Name (ID)
               Name (ID)


### Project Structure
1. `OriginalityForm` folder:
    Contains Expectations of Originality forms, signed by each student, which attest to the originality of the work.
2. `Emotions` folder:
    The Dataset containing all the images. The images are split into different classes (Angry, Happy, Neutral, and Engaged), each of which is further split into training and test folders. More information regarding the Dataset can be found in the `DatasetExplanation.txt` file.
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
