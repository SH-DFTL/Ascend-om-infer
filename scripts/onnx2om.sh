atc --framework=5 \
    --model=./yolo11n.onnx \
    --input_format=NCHW \
    --input_shape='images:1,3,640,640'\
    --output=./yolov11n_om \
    --log=error \
    --soc_version=Ascend310P3
