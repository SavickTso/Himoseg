ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

# ARG PYTORCH="1.13.1"
# ARG CUDA="11.6"
# ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update && apt install -y git 
RUN git clone https://github.com/SavickTso/HisRepItself.git
RUN pip install h5py scipy matplotlib pandas
COPY datasets/.   /workspace/HisRepItself/datasets/.


