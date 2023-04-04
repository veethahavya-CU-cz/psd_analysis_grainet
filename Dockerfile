FROM nvidia/cuda:10.0-runtime-ubuntu18.04

# update system and install system dependencies including python 3.7
RUN apt update && apt upgrade -y && apt clean && apt autoremove -y
RUN apt install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py
RUN apt install -y build-essential gcc libgdal-dev gdal-bin libproj-dev proj-data proj-bin libgeos-dev python3-dev

# install python dependencies
RUN pip install --upgrade pip
RUN pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
RUN pip install jupyter matplotlib tensorflow-gpu==1.13.1 h5py==2.9.0 keras==2.2.4 scikit-image setuptools==57.4.0


# set system environment variables
RUN export LD_PRELOAD=$THEPREFIXPATH/lib/libgdal.so.1
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal