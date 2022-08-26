# Great OpenCL Examples
xxx

## Requeriments

```bash
docker run -it -v /home/jhuang:/jhuang my_opencl_docker /bin/bash
cd /jhuang/repo/opencl_ws
```

## Building

Initially run the following command in a terminal:

```bash
mkdir build
cd build
cmake ..
make -j2
```

## Running
Run binaries from `build` directory.
```bash
cd build
./your_binary
```
