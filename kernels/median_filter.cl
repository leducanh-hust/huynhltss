__kernel void medianFilter(__global uchar *input, __global uchar *output,
                           const int width, const int height, const int channel,
                           const int filterSize) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int halfSize = filterSize / 2;

  if (x >= width || y >= height)
    return;

  for (int c = 0; c < channel; c++) {
    int values[225]; // Assume a max 15x15 filter size
    int count = 0;

    for (int ky = -halfSize; ky <= halfSize; ky++) {
      for (int kx = -halfSize; kx <= halfSize; kx++) {
        int nx = clamp(x + kx, 0, width - 1);
        int ny = clamp(y + ky, 0, height - 1);
        int idx = (ny * width + nx) * channel + c;
        values[count++] = input[idx];
      }
    }

    // Sort values to find median
    for (int i = 0; i < count - 1; i++) {
      for (int j = i + 1; j < count; j++) {
        if (values[i] > values[j]) {
          int temp = values[i];
          values[i] = values[j];
          values[j] = temp;
        }
      }
    }

    int outIdx = (y * width + x) * channel + c;
    output[outIdx] = values[count / 2];
  }
}
