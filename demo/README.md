# OMG-Seg Demo

We provide a single-file demo in this folder to facilitate getting started. Supposing that you are in the root directory of this project.

## Embedding Generation
To use the demo, you need first generate a class name list to tell the OMG-Seg model all possible categories as the vocabulary dictionary. We have already provided a sample vocabulary list in `demo/configs/names/th139_st101.py`.

Then, we need to generate the class embeddings based on the names. You can do this by the following command:
```commandline
PYTHONPATH=. python tools/gen_cls.py demo/configs/m2_convl.py
```
The script will automatically read the class list, which is imported in `demo/configs/m2_convl.py` (please refer to `CLASSES` and `DATASET_NAME`), and generate the embeddings.

## Run the Demo
After generating the embeddings, you can run the demo by:
```commandline
PYTHONPATH=. python demo/image_demo.py
```
for image; and
```commandline
PYTHONPATH=. python demo/video_demo.py
```
for video.

Please refer to `test_image` and `test_video` for the visualization of the outputs.

## Customization
If you want to try your own images or videos, please change the `IMG_PATH`, `VID_PATH`, and `MODEL_PATH`.

If you want to customize our model, please refer to the config scripts (`demo/configs/m2_convl.py` and `demo/configs/m2_convl_vid.py`) for details. 

Note that all the model-related code have been imported in the config file. You need to find the corresponding path to find the model implementation details. 