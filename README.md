{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf600
{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl280\partightenfactor0

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # Cell Migration Mode Prediction for High Magnification Images\
\
Prediction toolbox is a python package to facilitate [Deciphering Cancer Cell Migration by Systems Microscopy](https://sysmic.ki.se) project, aiming at predicting cell migration modes in low magnification time-lapse confocal microscopy image movies.\
Currently, this toolbox employs a CNN regression model for predicting (or classifying) the probability of each cell observation. This package provides functionalities:\
\
<h6>Tracking</h6>\
\
```\
Tracking of multiple cells in a high resolution time-lapse movie \
```\
<h6>Classification</h6>\
\
```\
Classification of each cell observation as 'continuous', 'discontinuous', or 'excluded' (i.e., mitosis). \
```\
\
## Installation\
\
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Prediction Toolbox.\
\
```bash\
pip install Prediction_Toolbox\
```\
### Prerequisites\
```\
python 3.6 or above\
matplotlib\
opencv\
tiffile\
skimage\
tensorflow\
keras\
tqdm\
collections\
prettytable\
pandas\
*if platform =='linux' or 'mac os'\
python bioformats\
pims\
pims_nd2\
```\
\
## Getting Started\
\
These instructions will get you a copy of the project up and running on your \
local machine for development and testing purposes. See Usage for \
notes on how to use the project on a live system.\
\
<h5>Download the CNN model</h5>\
\
[model](https://uppsala.box.com/s/83u06sagixbuxaukq78ymjxb9l7n07sy)\
\
<h5>run the following command from the bash to install required dependencies</h5>\
\
```bash\
\
user@user-pc:~path to cell_mig_lm_tbox_v01/ python package_test.py\
```\
\
#### Example Usage \
<h5>Case 1: Tracking only (see example_tracking.py)</h5>\
\
```python\
\
from helping_util.image_helping_util import display_directory\
from tracking_util.image_tracking_util import _tracker_lm,_utilities_for_tracking_lm\
\
'''\
    To track objects in the image series\
    \
    input\
    \
        nd2_folder: folder location where nd2 files are stored.\
        nd2_filename: name of nd2 movie for tracking objects.\
        no_of_frames: maximum number of frames to read for tracking objects and for saving \
        in the xlsx file. Maximum length is restricted to number of frames in a movie.\
        Option: 'full_movie' to track objects in full movie or intiger (e.g., 20).\
        Default is 40. If 'full_movie' is opted then processing time increases upto 4 times.\
        \
        fov_to_process: any specifc field of view to priorities. Only valid when multiple FOV exist.\
        \
        minimum_dis: Minimum distance tolerance of an object in two consequtive frames \
        to assign new object id.\
        \
        bord_radius: pixels distance of object from the border to remove the border \
        touching objects. Default is 2 pixels\
        \
        segmentation_method: option to select different methods for segmentation.\
        Default is otsu. Optional ['triangle', 'minimum','otsu','li', 'isodata']. \
        \
        lower_threshold_value: size (area) threshold to remove small objects (in pixels). \
            \
        eccentricity_threshold: threshold to remove non-circular debris.\
        upper_threshold_value: size (area) threshold to remove large objects (in pixels).\
        convert_to_binary_method: Default is otsu. Optional ['triangle', 'minimum','otsu',\
                                                             'li', 'isodata']\
        minimum_tracklet_length: track_length_tolerance to remove small tracks.\
        back_tracking_length: maximum length of tracks for computing min_dispalcement_to_reject \
        non-moveable objects. Higher range might affect the computation speed.\
    \
        back_tracking_length: maximum length of tracks for computing min_dispalcement_to_reject \
        \
        non-moveable objects. Higher range might affect the computation speed.\
        \
        min_dispalcement_to_reject: optional to remove debris or small tracks from the annotation file\
    \
    output\
    \
        a dictionary with detected objects with their object id\
        xlsx file will be saved in the subdirectory named as 'image_filename under the folder 'annotated_image/'\
'''\
\
input_image_folder='image_data'\
file_directory=display_directory(input_image_folder, extension='.tif',display=False)\
\
\
\
_,curated_objects= _tracker_lm(image_folder=input_image_folder,\
                                    image_filename=file_directory[0],\
                                    no_of_frames=10,\
                                    fov_to_process=None,\
                                    minimum_dis=35,\
                                    bord_radius=2,\
                                    segmentation_method='otsu',\
                                    threshold_value=2.32,\
                                    lower_size_threshold=240,\
                                    eccentricity_threshold=0.99,\
                                    upper_size_threshold=3500,\
                                    minimum_tracklet_length=5,\
                                    back_tracking_length=5,\
                                    min_dispalcement_to_reject=0.9)._lm_cell_tracker() \
```\
\
<h5>Case 2: Classification only (see example_prediction.py)</h5>\
\
```python\
\
import os\
import tifffile\
from helping_util.image_helping_util import display_directory\
from prediction_util.image_prediction_util import _prediction_lm\
\
"""\
    To perform prediction using CNN regression model\
    \
    input\
    \
        image_file: name of nd2 or tiff file.\
        \
        xlsx_file: name ofxlsx file saved after tracking.\
        \
        regression: flag to perform regression or classificaiton. Dafult is True for regression\
        \
        patch_w_and_h: size of patches to extract. Default is 60 pixels.\
        \
        global_context: Context to consider while patch extraction deafult is zero pixels.\
        \
        local_context: local contex while extracting final patch for prediction default is ten pixels.\
        \
    output\
    \
        dataframe with corresponding output predictions for each cell observations \
"""\
\
input_image_folder = "image_data"\
file_directory = display_directory(input_image_folder, extension=".tif", display=False)\
\
images = tifffile.imread(\
    os.getcwd() + "/" + input_image_folder + "/" + file_directory[0]\
)\
xlsx_file_path = "/annotated_image/" + str(file_directory[0].split(".tif")[0]) + "/"\
xlsx_file = display_directory(xlsx_file_path, extension=".xlsx", display=False)[0]\
xlsx_filename = os.getcwd() + "/" + xlsx_file_path + "/" + xlsx_file\
\
ext = "tif"\
patch_w_and_h = 60\
global_context = 0\
sub_movie_length = 3\
local_context = 10\
\
\
results = _prediction_lm._make_prediction(\
    image_file=images,\
    xlsx_file=xlsx_filename,\
    ext=ext,\
    regression=True,\
    patch_w_and_h=patch_w_and_h,\
    global_context=global_context,\
    local_context=local_context,\
)\
```\
\
<h5>Case 3: Complete pipeline (see main.py) </h5>\
\
```python\
\
from helping_util.image_helping_util import display_directory\
from cell_prediction_pipeline_util.cell_predict_pipeline import (\
    _pipeline_lm_single_frame,\
)\
\
"""\
    To track objects in the image series\
    \
    input\
    \
        image_folder: folder location where nd2 files are stored.\
        \
        image_filename: name of time-lapse movie for tracking objects.\
        \
        no_of_frames: maximum number of frames to read for tracking objects and for saving \
        in the xlsx file. Maximum length is restricted to number of frames in a movie.\
        Option: 'full_movie' to track objects in full movie or intiger (e.g., 20).\
        Default is 30. If 'full_movie' is opted then processing time increases upto 4 times.\
        \
        fov_to_process: any specifc field of view to priorities. Only valid when multiple FOV exist.\
        \
        minimum_dis: Minimum distance tolerance of an object in two consequtive frames \
        to assign new object id.\
        \
        bord_radius: pixels distance of object from the border to remove the border \
        touching objects. Default is 2 pixels\
        \
        segmentation_method: option to select different methods for segmentation.\
        Default is otsu. Optional ['triangle', 'minimum','otsu','li', 'isodata']. \
        \
        threshold_value: Range to segment the binary images for nuclei detection.\
        \
        lower_size_threshold: size (area) threshold to remove small objects (in pixels). \
            \
        eccentricity_threshold: threshold to remove non-circular debris.\
        \
        upper_size_threshold: size (area) threshold to remove large objects (in pixels).\
    \
        minimum_tracklet_length: track_length_tolerance to remove small tracks.\
        \
        back_tracking_length: maximum length of tracks for computing min_dispalcement_to_reject \
        non-moveable objects. Higher range might affect the computation speed.\
        \
        min_dispalcement_to_reject: optional to remove debris or small tracks from the annotation file\
        \
        patch_w_and_h: size of patches to extract. Default is 60 pixels.\
        \
        global_context: Context to consider while patch extraction deafult is zero pixels.\
        \
        local_context: local contex while extracting final patch for prediction default is ten pixels.\
        \
        regression: flag to perform regression or classificaiton. Dafult is True for regression\
    \
    tested with follwing conditions:\
        \
        ###Default no_of_frames=30:\
            minimum_tracklet_length=13,\
            back_tracking_length=5 or less,\
            min_dispalcement_to_reject= 0.9 or less,\
            minimum_dis=35,\
            bord_radius=2,\
            segmentation_method='otsu',\
            threshold_value=2.12,\
            lower_size_threshold=240,\
            eccentricity_threshold=0.99,\
            upper_size_threshold=3500,\
    \
        when no_of_frames= 40:\
            minimum_tracklet_length=13,\
            back_tracking_length=5 or less,\
            min_dispalcement_to_reject= 0.9 or less,\
            minimum_dis=35,\
            bord_radius=2,\
            segmentation_method='otsu',\
            threshold_value=2.12,\
            lower_size_threshold=240,\
            eccentricity_threshold=0.99,\
            upper_size_threshold=3500,\
    \
        when no_of_frames= 60:\
            minimum_tracklet_length=13,\
            back_tracking_length=5 or less,\
            min_dispalcement_to_reject=0.9 or less,\
            minimum_dis=35,\
            bord_radius=2,\
            segmentation_method='otsu',\
            threshold_value=2.12,\
            lower_size_threshold=240,\
            eccentricity_threshold=0.99,\
            upper_size_threshold=3500,\
    \
        when no_of_frames= 80:\
            minimum_tracklet_length=10,\
            back_tracking_length=5 or less,\
            min_dispalcement_to_reject= 1 or less,\
            minimum_dis=35,\
            bord_radius=2,\
            segmentation_method='otsu',\
            threshold_value=2.32,\
            lower_size_threshold=240,\
            eccentricity_threshold=0.99,\
            upper_size_threshold=3500,\
"""\
\
input_image_folder = "image_data"\
file_directory = display_directory(input_image_folder, extension=".tif", display=False)\
\
\
_pipeline_lm_single_frame(\
    image_folder=input_image_folder,\
    image_filename=file_directory[0],\
    no_of_frames=30,\
    fov_to_process=None,\
    minimum_dis=35,\
    bord_radius=2,\
    segmentation_method="otsu",\
    threshold_value=2.12,\
    lower_size_threshold=240,\
    eccentricity_threshold=0.99,\
    upper_size_threshold=3500,\
    minimum_tracklet_length=13,\
    back_tracking_length=5,\
    min_dispalcement_to_reject=0.9,\
    patch_w_and_h=60,\
    global_context=0,\
    local_context=10,\
    regression=True,\
)\
```\
\
## Contributing\
Pull requests are welcome. For major changes, please open an issue first to \
discuss what you would like to change. Please make sure to update tests as appropriate.\
\
## License\
[MIT](https://choosealicense.com/licenses/mit/) \
\
## Authors\
\
* [W\'e4hlby Lab](http://user.it.uu.se/~cli05194/research_n_support.html)\
* [Anindya Gupta](https://www.it.uu.se/katalog/anigu165)\
\
## Acknowledgments\
\
* [Str\'f6mblad Group](https://ki.se/en/bionut/cell-biology-of-cancer-staffan-stromblad-0)\
}