# About:
Quantization on weed detection with PyTorch on deepweeds dataset for performance improvement. Tested on RPi, Mobile processors and PC.

# Code and model details:
1. Clone this repository locally move into the project folder
2. In the local env run `pip install -r requirements.txt`


## Training the model:
Download and place the image and lables from the deepweeds dataset into `prepare_data/deepweeds/` and execute the **train_model.py** file. [dataset-github-repo](https://github.com/AlexOlsen/DeepWeeds)

## Inference on the model:
Download the trained models from the Google-Drive [link](https://drive.google.com/drive/folders/1QDujacI3uPdzVHRBGf9bZKWJjA_3DlnF?usp=share_link) [request access with purpose of access] and place them in `models/` and execute the **inference.py** file.

# Hardware testing
1. Android:
Tested on android device using Android Debug Bridge which enable the use of mobile processor (present on your debug device) without the need to build an actual application. However, few extra steps are needed to convert the current trained models [ref](https://pytorch.org/tutorials/recipes/mobile_perf.html) .

Steps:
  First, clone the pytorch official repository
  ```
  git clone https://github.com/pytorch/pytorch.git

  cd pytorch

  git submodule update --init --recursive

  export ANDROID_ABI=arm64-v8a

  # Change accordingly
  export ANDROID_NDK=/path/to/Android/Sdk/ndk/21.0.6113669/

  ./scripts/build_android.sh \
  -DBUILD_BINARY=ON \
  -DBUILD_CAFFE2_MOBILE=OFF \
  -DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
  -DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.executable)') \
  ```
  pass the downloaded models through `save_for_RPI_inference` present in **inference.py**

  ```
  adb shell mkdir /data/local/tmp/pt
  adb push build_android/install/bin/speed_benchmark_torch /data/local/tmp/pt
  adb push <model.pt> /data/local/tmp/pt

  adb shell  /data/local/tmp/pt/speed_benchmark_torch \
  --model  /data/local/tmp/pt/<model.pt> --input_dims="1,3,224,224" \
  --input_type=float --warmup=5 --iter 100

  ```
  Sample Result:
  ```
  Starting benchmark.
  Running warmup runs.

  Main runs.
  Main run finished. Microseconds per iter: xxx. Iters per second: xxx
  ```

2. Raspberry Pi:
Since RPi is ARM based, few instrutions are specific for this family of architectures which are different from the x86 ones. PyTorch handels most of them, just add `torch.backends.quantized.engine = 'qnnpack'` in inference file and run on RPi hardware.

3. PC:
No extra updates needed, run the code present in inference on your hardware.
