# Model and Data Archive
This is an archive for model scripts and example data for a submitted manuscript titled "Automated Mapping of Particle Size Distribution from UAV-imagery using the CNN-based GrainNet Model".
Authors: Theodora Lendzioch, Jakub Langhammer, and Veethahavya K S

## Instructions to run the model with example dataset provided via a docker container
Firstly, move into the directory `src/GRAINet`, clone the GRAINet repository, and unpack the source files into that directory, using the commands:
```
cd src/GRAINet
git clone https://github.com/langnico/GRAINet.git
mv GRAINet/* .
rm -rf GRAINet
```

Move back to the root directory of this archive and build the docker image using the command:
```
cd ../..
docker build -t grain_cuda .
```

After the successful build, run the docker container using the command:
```
docker run -it --name grain_env --mount type=bind,source=[CWD],target=/home/grain/ grain_cuda /bin/bash
```
where [CWD] is the path to the root of this archive.

Once the container is running, run the following commands to preprocess the data and run the model:
```
cd /home/grain
python src/preprocessor.py
python src/run_GRAINet.py
```