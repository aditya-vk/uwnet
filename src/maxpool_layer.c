#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    int sizesq = l.size * l.size;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (int i = 0; i < in.rows; ++i) {
      image im = float_to_image(
        in.data + i*in.cols, l.width, l.height, l.channels
      );
      matrix col = im2col(im, l.size, l.stride);
      for (int j = 0; j < out.cols; ++j) {
        int c_im = j / (outw * outh);
        int j_col = j % (outw * outh);
        float maxv = FLT_MIN;
        for (int k = 0; k < sizesq; ++k) {
          int col_index = c_im * (outw * outh * sizesq) +
                          k * (outw * outh) + j_col;
          if (col.data[col_index] > maxv) {
            maxv = col.data[col_index];
          }
        }
        out.data[i*out.cols + j] = maxv;
      }
      free_matrix(col);
    }
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    int pad = 0;
    if (l.size % 2 == 1) {
      pad = (l.size - 1) / 2;
    } else {
      pad = ((outw - 1) * l.stride + 1) + (l.size - 1) - l.width;
      if (pad < 0) {
        pad = 0;
      }
    }

    int sizesq = l.size * l.size;
    for (int i = 0; i < in.rows; ++i) {
      image im = float_to_image(
        in.data + i*in.cols, l.width, l.height, l.channels
      );
      matrix col = im2col(im, l.size, l.stride);
      for (int j = 0; j < dy.cols; ++j) {
        int c_im = j / (outw * outh);
        int j_col = j % (outw * outh);
        float maxv = FLT_MIN;
        int maxk = 0;
        for (int k = 0; k < sizesq; ++k) {
          int col_index = c_im * (outw * outh * sizesq) +
                          k * (outw * outh) + j_col;
          if (col.data[col_index] > maxv) {
            maxv = col.data[col_index];
            maxk = k;
          }
        }

        int w = j_col % outw;
        int h = (j_col / outw) % outh;

        int w_kernel = maxk % l.size;
        int h_kernel = maxk / l.size;

        int x = w * l.stride + w_kernel - pad;
        int y = h * l.stride + h_kernel - pad;
        if (x >= 0 && y >= 0 && x < l.width && y < l.height) {
        dx.data[c_im*(l.height * l.width) + y*(l.width) + x] = 
            dy.data[i*dy.cols + j];
        }
      }
      free_matrix(col);
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

