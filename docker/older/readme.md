# Running
```bash
docker run -it -v /home/jhuang:/home/jhuang my_opencl_docker /bin/bash
```
```bash
docker run -it -v /home/jhuang:/jhuang cwpearson/opencl2.0-intel-cpu:latest /bin/bash
apt-get update && apt-get install -y cmake build-essential lsb-release libboost-all-dev libglu1-mesa-dev mesa-common-dev && apt-get clean all
```

# Building
```bash
/bin/sh -c apt-get update -q && apt-get install --no-install-recommends -yq alien wget unzip clinfo     && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
/bin/sh -c export DEVEL_URL="https://software.intel.com/file/531197/download"     && wget ${DEVEL_URL} -q -O download.zip --no-check-certificate     && unzip download.zip     && rm -f download.zip *.tar.xz*     && alien --to-deb *dev*.rpm     && dpkg -i *dev*.deb     && rm *.rpm *.deb
/bin/sh -c export RUNTIME_URL="http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz"     && export TAR=$(basename ${RUNTIME_URL})     && export DIR="opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25"     && wget -q ${RUNTIME_URL}     && tar -xf ${TAR}     && for i in ${DIR}/rpm/*.rpm; do alien --to-deb $i; done     && rm -rf ${DIR} ${TAR}     && dpkg -i *.deb     && rm *.deb
/bin/sh -c mkdir -p /etc/OpenCL/vendors/     && echo "/opt/intel/opencl-1.2-6.4.0.25/lib64/libintelocl.so" > /etc/OpenCL/vendors/intel.icd
ENV OCL_INC=/opt/intel/opencl/include
ENV OCL_LIB=/opt/intel/opencl-1.2-6.4.0.25/lib64
ENV LD_LIBRARY_PATH=/opt/intel/opencl-1.2-6.4.0.25/lib64:
```

```bash
/bin/sh -c apt-get update && apt-get install -y --no-install-recommends build-essential clinfo cpio libhwloc5 ocl-icd-opencl-dev opencl-headers wget && rm -rf /var/lib/apt/lists/*
PACKAGE_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/12556/opencl_runtime_16.1.2_x64_rh_6.4.0.37.tgz
PACKAGE_NAME=opencl_runtime_16.1.2_x64_rh_6.4.0.37
wget -q ${PACKAGE_URL} -O /tmp/opencl_runtime.tgz && tar -xzf /tmp/opencl_runtime.tgz -C /tmp && sed 's/decline/accept/g' -i /tmp/${PACKAGE_NAME}/silent.cfg && /tmp/${PACKAGE_NAME}/install.sh -s /tmp/${PACKAGE_NAME}/silent.cfg --cli-mode && rm -rf /tmp/opencl_runtime.tgz
/bin/sh -c apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
```
