__kernel void renderGradient(__global uchar *image, int width, int height,
                             int sequential) {
  if (sequential == 1) {
    if (get_global_id(0) == 0) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int index = (y * width + x) * 3;

          // Complex computation: add an iterative loop and trigonometric
          // functions
          float r = 0.0f;
          for (int i = 0; i < 1000; ++i) {
            r += sin(x * 0.001f) * cos(y * 0.001f);
          }

          image[index + 0] = (uchar)(255 * (r / 1000.0f)); // Red channel
          image[index + 1] =
              (uchar)(255 * ((float)y / height)); // Green channel
          image[index + 2] = (uchar)(255 * ((float)(x + y) /
                                            (width + height))); // Blue channel
        }
      }
    }
  } else {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
      return;
    
    int index = (y * width + x) * 3;

    // Complex computation: add an iterative loop and trigonometric functions
    float r = 0.0f;
    for (int i = 0; i < 1000; ++i) {
      r += sin(x * 0.001f) * cos(y * 0.001f);
    }

    image[index + 0] = (uchar)(255 * (r / 1000.0f));       // Red channel
    image[index + 1] = (uchar)(255 * ((float)y / height)); // Green channel
    image[index + 2] =
        (uchar)(255 * ((float)(x + y) / (width + height))); // Blue channel
  }
}
