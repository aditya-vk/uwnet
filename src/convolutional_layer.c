#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix xw: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
// returns: y = wx + b
matrix forward_convolutional_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols % b.cols == 0);

    matrix y = copy_matrix(xw);
    int spatial = xw.cols / b.cols;
    int i,j;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            y.data[i*y.cols + j] += b.data[j/spatial];
        }
    }
    return y;
}

// Calculate dL/db from a dL/dy
// matrix dy: derivative of loss wrt xw+b, dL/d(xw+b)
// returns: derivative of loss wrt b, dL/db
matrix backward_convolutional_bias(matrix dy, int n)
{
    assert(dy.cols % n == 0);
    matrix db = make_matrix(1, n);
    int spatial = dy.cols / n;
    int i,j;
    for(i = 0; i < dy.rows; ++i){
        for(j = 0; j < dy.cols; ++j){
            db.data[j/spatial] += dy.data[i*dy.cols + j];
        }
    }
    return db;
}

/// compute necessary padding for a kernel layer
int compute_padding(int width, int height, int size, int stride)
{
    int outw = (width  - 1) / stride + 1;
    int outh = (height - 1) / stride + 1;

    // odd size = reference point in center of kernel
    if (size % 2 == 1) {
      return (size - 1) / 2;
    }

    // even size = reference from top left
    int padw = ((outw - 1) * stride + 1) + (size - 1) - width;
    int padh = ((outh - 1) * stride + 1) + (size - 1) - height;
    int pad = (padw > padh) ? padw : padh;
    return (pad > 0) ? pad : 0;
}

/// image::get_pixel, with padding uncommented
float get_padded_pixel(image im, int x, int y, int c)
{
    if(x >= im.w) return 0;
    if(y >= im.h) return 0;
    if(x < 0) return 0;
    if(y < 0) return 0;

    assert(c >= 0);
    assert(c < im.c);
    return get_pixel(im, x, y, c);
}

/// image::get_pixel, with padding uncommented
void set_padded_pixel(image im, int x, int y, int c, float v)
{
    if(x >= im.w) return;
    if(y >= im.h) return;
    if(x < 0) return;
    if(y < 0) return;

    assert(c >= 0);
    assert(c < im.c);
    return set_pixel(im, x, y, c, v);
}

// Make a column matrix out of an image
// image im: image to process
// int size: kernel size for convolution operation
// int stride: stride for convolution
// returns: column matrix
matrix im2col(image im, int size, int stride)
{
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;
    int cols = outw * outh;
    int pad = compute_padding(im.w, im.h, size, stride);
    int size_sq = size * size;

    matrix col = make_matrix(rows, cols);
    for (int c = 0; c < rows; ++c) {
      int c_im = c / size_sq;
      int w_kernel = (c % size_sq) % size;
      int h_kernel = (c % size_sq) / size;
      for (int j_col = 0; j_col < cols; ++j_col) {
        int w = j_col % outw;
        int h = (j_col / outw) % outh;
        int x = w * stride + w_kernel - pad;
        int y = h * stride + h_kernel - pad;
        col.data[c * cols + j_col] = get_padded_pixel(im, x, y, c_im);
      }
    }
    return col;
}

// The reverse of im2col, add elements back into image
// matrix col: column matrix to put back into image
// int size: kernel size
// int stride: convolution stride
// image im: image to add elements back into
image col2im(int width, int height, int channels, matrix col, int size, int stride)
{
    image im = make_image(width, height, channels);

    // TODO: 5.2
    // Add values into image im from the column matrix
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;
    int cols = outw * outh;
    int pad = compute_padding(im.w, im.h, size, stride);
    int size_sq = size * size;

    for (int c = 0; c < rows; ++c) {
      int c_im = c / size_sq;
      int w_kernel = (c % size_sq) % size;
      int h_kernel = (c % size_sq) / size;
      for (int j_col = 0; j_col < cols; ++j_col) {
        int w = j_col % outw;
        int h = (j_col / outw) % outh;
        int x = w_kernel + w * stride - pad;
        int y = h_kernel + h * stride - pad;
        float value =
            get_padded_pixel(im, x, y, c_im) +
            col.data[c * cols + j_col];
        set_padded_pixel(im, x, y, c_im, value);
      }
    }
    return im;
}

// Run a convolutional layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_convolutional_layer(layer l, matrix in)
{
    assert(in.cols == l.width*l.height*l.channels);
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int i, j;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.filters);
    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(
          in.data + i*in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        matrix wx = matmul(l.w, x);
        for(j = 0; j < wx.rows*wx.cols; ++j){
            out.data[i*out.cols + j] = wx.data[j];
        }
        free_matrix(x);
        free_matrix(wx);
    }
    matrix y = forward_convolutional_bias(out, l.b);
    free_matrix(out);

    return y;
}

// Run a convolutional layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
matrix backward_convolutional_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    assert(in.cols == l.width*l.height*l.channels);

    int i;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;


    matrix db = backward_convolutional_bias(dy, l.db.cols);
    axpy_matrix(1, db, l.db);
    free_matrix(db);


    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);
    matrix wt = transpose_matrix(l.w);

    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);

        dy.rows = l.filters;
        dy.cols = outw*outh;

        matrix x = im2col(example, l.size, l.stride);
        matrix xt = transpose_matrix(x);
        matrix dw = matmul(dy, xt);
        axpy_matrix(1, dw, l.dw);

        matrix col = matmul(wt, dy);
        image dxi = col2im(l.width, l.height, l.channels, col, l.size, l.stride);
        memcpy(dx.data + i*dx.cols, dxi.data, dx.cols * sizeof(float));
        free_matrix(col);

        free_matrix(x);
        free_matrix(xt);
        free_matrix(dw);
        free_image(dxi);

        dy.data = dy.data + dy.rows*dy.cols;
    }
    free_matrix(wt);
    return dx;

}

// Update convolutional layer
// layer l: layer to update
// float rate: learning rate
// float momentum: momentum term
// float decay: l2 regularization term
void update_convolutional_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 5.3
    axpy_matrix(decay, l.w, l.dw);
    axpy_matrix(-rate, l.dw, l.w);
    scal_matrix(momentum, l.dw);
    axpy_matrix(-rate, l.db, l.b);
    scal_matrix(momentum, l.db);
}

// Make a new convolutional layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of convolutional filter to apply
// int stride: stride of operation
layer make_convolutional_layer(int w, int h, int c, int filters, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.filters = filters;
    l.size = size;
    l.stride = stride;
    l.w  = random_matrix(filters, size*size*c, sqrtf(2.f/(size*size*c)));
    l.dw = make_matrix(filters, size*size*c);
    l.b  = make_matrix(1, filters);
    l.db = make_matrix(1, filters);
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update   = update_convolutional_layer;
    return l;
}
