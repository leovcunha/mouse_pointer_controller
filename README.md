# Computer Pointer Controller

This projects uses Intel OpenVino Toolkit plus its pretrained AI models to build a Gaze Estimation model that controls the mouser pointer position based on the user's eyes gaze.

## Project Set Up and Installation

Setup procedures to run the project.

**1. Have Python 3.6 and [OpenVino](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html) installed**

**2. Install requirements**

```bash
pip install -r requirements.txt
```

**3. Download models**

```bash
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-retail-0004 -o ./models
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o ./models
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o ./models
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o ./models
```

## Demo

_TODO:_ Explain how to run a basic demo of your model.

## Documentation

_TODO:_ Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks

_TODO:_ Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results

_TODO:_ Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.
