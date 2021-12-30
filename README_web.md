# Great OpenCL Examples
This README is based on [Great OpenCL Examples](https://github.com/michel-meneses/great-opencl-examples)

## Requeriments

The examples in this repository require a valid implementation of OpenCL in your system. To install it, follow the next instructions according to your machine OS:

### Debian/Ubuntu
 
 Install OpenCl headers: 
 
    sudo apt-get install opencl-headers

 Install OpenCL drivers according to your parallel computing device vendor:

 - Intel: `sudo apt-get install beignet-dev`
 - AMD: `sudo apt-get install amd-opencl-dev`
 - Nvidia: `sudo apt-get install nvidia-opencl-dev`

### Windows

The following instructions must be performed to install OpenCL on a Windows/OSX device:

 1. Download the OpenCL SDK: 
	-  Check out the website of your device vendor (*e.g.* Intel, AMD, Nvidia, etc).
 2. Set up OpenCL on your IDE:
	- Add header file (.h) directory to includes;
	- Add OpenCL.lib file to linker settings.

For a visual demonstration of how to set up OpenCL on a Windows platform with Visual Studio, watch this video tutorial: https://youtu.be/mtA94WAxkPM (*credits*: Wesley Shillingford).

## Usage

To compile a code with OpenCL, run the following command in a terminal:

    g++ -std=c++0x -o output src.cpp -lOpenCL

## Bonus: OpenCL + CImg

This repository also provides the OpenCL source code of an image filtering application based on the [CImg](http://cimg.eu/) library. This entire library has the form of a single header file, which is already included in this repository. To compile that source code with GCC, run the following command on a terminal:

    g++ -std=c++0x -o output src.cpp -lOpenCL -lm -lpthread -lX11

## References

 1. K. O. W. Group. *The OpenCL Specification*. The Khronos Group, 2.2-10 edition, feb 2019. URL: https://www.khronos.org/registry/OpenCL/specs/2.2/pdf/OpenCL_API.pdf
 2. K. O. W. Group. *The OpenCL C++ 1.0 Specification*. The Khronos Group, 2.2-10 edition, feb 2019. URL: https://www.khronos.org/registry/OpenCL/specs/2.2/pdf/OpenCL_Cxx.pdf
 3. Tschumperlé, D. *Introduction to The CImg Library*. CNRS UMR 6072 (GREYC) - Image Team. URL: http://cimg.eu/CImg_slides.pdf

## License

This repository is *free* and distributed under the MIT license.

