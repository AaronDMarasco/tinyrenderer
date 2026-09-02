# Software rendering in 500 lines of bare C++

The code itself is of little interest. Check the course notes:
1. [Introduction](https://haqr.eu/tinyrenderer/)
2. [Bresenham’s line drawing algorithm](https://haqr.eu/tinyrenderer/bresenham/)
3. [Triangle rasterization](https://haqr.eu/tinyrenderer/rasterization/)
4. [Primer on barycentric coordinates](https://haqr.eu/tinyrenderer/barycentric/)
5. [Hidden faces removal](https://haqr.eu/tinyrenderer/z-buffer/)
6. [A crude (but simple) approach to camera handling](https://haqr.eu/tinyrenderer/camera-naive/)
7. [Better camera handling](https://haqr.eu/tinyrenderer/camera/)
8. [Shading](https://haqr.eu/tinyrenderer/shading/)
9. [More data!](https://haqr.eu/tinyrenderer/textures/)
10. [Tangent space normal mapping](https://haqr.eu/tinyrenderer/tangent/)
11. [Shadow mapping](https://haqr.eu/tinyrenderer/shadow/)
12. [Indirect lighting](https://haqr.eu/tinyrenderer/ssao/)
13. [Bonus: toon shading](https://haqr.eu/tinyrenderer/toon/)
14. [Afterword](https://haqr.eu/tinyrenderer/afterword/)

---

_Notes for Python users:_ This Python fork/branch is unsupported by [Dmitry V. Sokolov](https://github.com/ssloy) but instead is by [Aaron D. Marasco](https://github.com/AaronDMarasco) on github. In the `python` branch you will find all of the code including solutions, with the naming convention that the "line drawing algorithm" is considered "Lesson 1." You will also find a branch `python_start_here` that just has the core libraries to start with (TGAImage, etc) similar to the original C++.

_Starting for Python users:_ Make sure `uv` is installed on your machine. Check out the repo, switch to either `python` or `python_start_here` branch, go to the `python` subdirectory, and run `uv sync --dev`. From there, `./check` will lint your code and run any tests it can find.

Misc Python notes:
1. **Warning**: The biggest Python hurdle is `numpy` operations on matrices may use different notation, _e.g._ `@`. The Python implementation of vectors uses `*` for dot-product, use `vector.cross()` for cross-product.
1. I encourage you to look at the files in `lib` and get familiar with the types and utilities that are available.
  * Limited documentation available by running the `make_doc` script
1. When possible, the more "pythonic" property interfaces are used, _e.g._ `vec4.xyz` to return a `vec3` vs. the C++ method call `vec4.xyz()`.
1. Use `uv run yourscript.py` to run and test your code.
1. Editing the `check` script gives options like running a profiler or only linting.
1. Installing the Linux Steam client will get some TGA files that are used by some self-tests; if you do not, they will be safely skipped but not all functionality will be tested.
1. All Python code is properly typed wherever possible; it gets a little looser when interfacing with `numpy`.
1. Things outside of `python_start_here` are definitely less documented.


_(end of Python notes)_

---

In this series of articles, I aim to demonstrate how OpenGL, Vulkan, Metal, and DirectX work by writing a simplified clone from scratch.
Surprisingly, many people struggle with the initial hurdle of learning a 3D graphics API.
To help with this, I have prepared a short series of lectures, after which my students are able to produce quite capable renderers.

The task is as follows: using no third-party libraries (especially graphics-related ones), we will generate an image like this:

![](https://haqr.eu/tinyrenderer/home/africanhead.png)

_Warning: This is a training material that loosely follows the structure of modern 3D graphics libraries.
It is a **software renderer**.
**I do not intend to show how to write GPU applications — I want to show how they work.**
I firmly believe that understanding this is essential for writing efficient applications using 3D libraries._

## The starting point

The final code consists of about 500 lines.
My students typically require 10 to 20 hours of programming to start producing such renderers.
The input is a 3D model composed of a triangulated mesh and textures.
The output is a rendering.
There is no graphical interface, the program simply generates an image.

To minimize external dependencies, I provide my students with a single class for handling [TGA](http://en.wikipedia.org/wiki/Truevision_TGA) files —
one of the simplest formats supporting RGB, RGBA, and grayscale images.
This serves as our foundation for image manipulation.
At the beginning, the only available functionality (besides loading and saving images) is the ability to set the color of a single pixel.

There are no built-in functions for drawing line segments or triangles — we will implement all of this manually.
While I provide my own source code, written alongside my students, I do not recommend using it directly, as doing the work yourself is essential to understanding the concepts.
The complete code is available on [github](https://github.com/ssloy/tinyrenderer), and you can find the initial source code I provide to my students [here](https://github.com/ssloy/tinyrenderer/tree/706b2dfecff65daeb93de568ee2c2bd87f277860).
Behold, here is the starting point:

```cpp
#include "tgaimage.h"

constexpr TGAColor white   = {255, 255, 255, 255}; // attention, BGRA order
constexpr TGAColor green   = {  0, 255,   0, 255};
constexpr TGAColor red     = {  0,   0, 255, 255};
constexpr TGAColor blue    = {255, 128,  64, 255};
constexpr TGAColor yellow  = {  0, 200, 255, 255};

int main(int argc, char** argv) {
    constexpr int width  = 64; 
    constexpr int height = 64; 
    TGAImage framebuffer(width, height, TGAImage::RGB);

    int ax =  7, ay =  3;  
    int bx = 12, by = 37; 
    int cx = 62, cy = 53; 

    framebuffer.set(ax, ay, white);
    framebuffer.set(bx, by, white);
    framebuffer.set(cx, cy, white);

    framebuffer.write_tga_file("framebuffer.tga");
    return 0;
}
```

It produces the 64x64 image `framebuffer.tga`, here I scaled it for better readability:

![](https://haqr.eu/tinyrenderer/bresenham/bresenham0.png)


## Teaser: few examples made with the renderer

![](https://haqr.eu/tinyrenderer/home/demon.png)

![](https://haqr.eu/tinyrenderer/home/diablo-glow.png)

![](https://haqr.eu/tinyrenderer/home/boggie.png)

![](https://haqr.eu/tinyrenderer/home/diablo-ssao.png)

## Compilation

```sh
git clone https://github.com/ssloy/tinyrenderer.git &&
cd tinyrenderer &&
cmake -Bbuild &&
cmake --build build -j &&
build/tinyrenderer obj/diablo3_pose/diablo3_pose.obj obj/floor.obj
```
The rendered image is saved to `framebuffer.tga`.
