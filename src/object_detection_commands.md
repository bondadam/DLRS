# Object Detection Commands

If you have tensorflow version != 1.5

    pip uninstall tensorflow

Then install the following

    pip install lxml protobuf Cython contextlib2 pillow tensorflow==1.15 tensorflow-gpu==1.15

In DLRS
    
    git clone https://github.com/tensorflow/models.git

Install pycocotools
    
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

*To use GPU:*

Donwload CUDA Toolkit (version 10.0): https://developer.nvidia.com/cuda-toolkit-archive (The version is important)

Donwload cuDNN v7.6.5 for CUDA 10.0: https://developer.nvidia.com/cudnn

**Reboot the computer.**

Run `src/simulator.py`

Run `src/Malfunction Detection.ipynb`

In `models/research` directory:


    ..\..\protoc\bin\protoc.exe object_detection\protos\*.proto --python_out=.


In `models/research/object_detection` directory:

Download http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz and extract it. (This needs to be in a folder named `ssd_mobilenet_v1_coco_2018_01_28`)

Move `models/research/object_detection/legacy/train.py` to `models/research/object_detection`.

* **For training:**
```bash
python train.py --logtostderr --train_dir=malfunction_detection_model/training/ --pipeline_config_path=malfunction_detection_model/training/ssd_mobilenet_v1_coco.config
```
* **For real-time visualization of the training:**

```bash
tensorboard --logdir=malfunction_detection_model/training --host=127.0.0.1
```

* After training finishes:

replace `{last_checkpoint_number}` with the highest numbered checkpoint saved in `models/research/object_detection/malfunction_detection_model/`

```bash
python .\export_inference_graph.py --input_type image_tensor --pipeline_config_path malfunction_detection_model/training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix malfunction_detection_model/training/model.ckpt-{last_checkpoint_number} --output_directory malfunction_detection_model
```

Depending on the GPU you use, you might get Resource Allocation Errors while running the training. If that is the case, you might want to lower the `batch_size` & `num_steps` parameters in `malfunction_detection_model/training/ssd_mobilenet_v1_coco.config`

**Override `models/research/object_detection/object_detection_tutorial.ipynb` with `models/research/object_detection/malfunction_detection_model/object_detection_tutorial.ipynb`** and run `models/research/object_detection/object_detection_tutorial.ipynb`