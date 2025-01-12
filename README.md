# Ascend-om-infer
Description: Inference of the YOLO11 model on Ascend devices.
## Environment
|softwares|versions|
| :-------|:-------|
|CANN|8.0.RC2|
|Python|3.9.18|

## PT_to_Onnx

## ONNX_to_OM
```shell
atc --framework=5 \
    --model=./yolo11n.onnx \
    --input_format=NCHW \
    --input_shape='images:1,3,640,640'\
    --output=./yolov11n_om \
    --log=error \
    --soc_version=Ascend310P3
```
## Inference

### PyACL(Python)
pyACL (Python Ascend Computing Language) is a Python API library obtained by encapsulating AscendCL with CPython. It provides APIs for runtime management, model inference, media data processing, and single-operator execution, enabling users to conduct runtime management, resource management, etc. for Ascend AI processors through the Python language.

### C++
We will continue to supplement later