# Computer Pointer Controller

This projects uses Intel OpenVino Toolkit plus its pretrained AI models to build a Gaze Estimation model that controls the mouser pointer position based on the user's eyes gaze.

## Project Set Up and Installation

Setup procedures to run the project.

**1. Have Python 3.6 and [OpenVino](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html) installed following instruction**

**1.5. If openvino variables are not preconfigured run:**

```
source /out/intel/openvino/bin/setupvars.sh -pyver 3.6
```

**2. Install requirements**

```bash
pip3 install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt #inside folder
```

**3. Download models**

```bash
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-retail-0004 -o ./models
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o ./models
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o ./models
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o ./models
```

## Demo

how to run a basic demo of the model:

`python3 src/app.py -t video -i ./bin/demo.mp4`

## Documentation

_TODO:_ Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks

#### CPU (Macbook Air 2013 - Dual-Core Intel Core i5)

| Loading Times        | FP32    | FP16    | FP16-INT8 |     |
| -------------------- | ------- | ------- | --------- | --- |
| Face Detection       | 0.37740 | -       | -         | sec |
| Facial Landmarks     | 0.15526 | 0.24160 | 0.29069   | sec |
| Head Pose Estimation | 0.14334 | 0.17567 | 0.36981   | sec |
| Gaze Estimation      | 0.17259 | 0.22204 | 0.38439   | sec |

 <br>

| Inference Times / Frame | FP32    | FP16    | FP16-INT8 |     |
| ----------------------- | ------- | ------- | --------- | --- |
| Face Detection          | 0.04233 | 0.04365 | 0.04395   | sec |
| Facial Landmarks        | 0.00162 | 0.00169 | 0.00176   | sec |
| Head Pose Estimation    | 0.00287 | 0.00363 | 0.00301   | sec |
| Gaze Estimation         | 0.00271 | 0.00300 | 0.00243   | sec |

Unfortunately only CPU is available for use with OpenVino in my computer

```
>>> ie = IECore()
>>> print(ie.available_devices)
['CPU']
```

## Results

It can be seen that face detection is the most time consuming model.  
All the models had very similar results for the processor used for inference.
The loading times were better for the FP32 model.
