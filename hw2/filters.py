import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for m in range(Hi):
        for n in range(Wi):
            s=0.0
            #kernel width and height
            for i in range(Hk):
                for j in range(Wk):
                    if m+1-i < 0 or n+1-j < 0 or m+1-i >= Hi or n+1-j >= Wi:
                        s+= 0
                    else:
                        s += kernel[i][j]*image[m+1-i][n+1-j]
            out[m][n] = s
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((2*pad_height+H, 2*pad_width+W))
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    #we have to flip the kernel because of how convolution is defined
    ker = np.flip(kernel, axis=(0,1))
    img = zero_pad(image, Hk //2, Wk //2)
    out = np.zeros((Hi, Wi))
    for m in range(Hi):
        for n in range(Wi):
            # our slice is the same size as the kernel, starting w/
            # the zero-padded beginning and ending with the zero-padded end
            out[m][n] = np.sum(img[m:m+Hk, n:n+Wk] * ker)
    
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation_control(f,g):
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for m in range(Hi):
        for n in range(Wi):
            s = 0.00
            for i in range(Hk):
                if i+m >= Hi:
                    break
                for j in range(Wk):
                    if  j+n >= Wi:
                        break
                    s+= f[i +m][j + n]*g[i][j] 
            out[m][n] = s
    ### END YOUR CODE

    return out
def cross_correlation_smart(f,g):
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    #we have to flip the kernel because of how convolution is defined
    ker = g
    img = zero_pad(f, Hk, Wk)
    out = np.zeros((Hi, Wi))
    for m in range(Hi):
        for n in range(Wi):
            # our slice is the same size as the kernel, starting w/
            # the zero-padded beginning and ending with the zero-padded end
            out[m][n] = np.sum(img[(Hk+m):m+2*Hk,  (Wk+n):n+ 2*Wk] * ker)
    
    ### END YOUR CODE

    return out



def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_ = np.flip(g, axis=(0,1))
    out = conv_fast(f,g_)
    
    # out = cross_correlation_smart(f,g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    f_mean = np.mean(f, axis=(0,1))
    f_zero_mean = f - f_mean
    g_mean = np.mean(g, axis=(0,1))
    g_zero_mean = g - g_mean
    
    out = cross_correlation(f, g_zero_mean)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    
    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    #we have to flip the kernel because of how convolution is defined
    g_mean = np.mean(g, axis=(0,1))
    g_var = np.var(g, axis=(0,1))

    ker =( g - g_mean) / g_var

    img = zero_pad(f, Hk //2, Wk //2)
    out = np.zeros((Hi, Wi))
    for m in range(Hi):
        for n in range(Wi):
            mean_fmn = np.mean(img[m:m+Hk, n:n+Wk], axis=(0,1))
            var_fmn = np.var(img[m:m+Hk, n:n+Wk], axis=(0,1))
            f_patch = (img[m:m+Hk, n:n+Wk] - mean_fmn) / var_fmn
            # our slice is the same size as the kernel, starting w/
            # the zero-padded beginning and ending with the zero-padded end
            out[m][n] = np.sum(f_patch * ker)
    
    ### END YOUR CODE

    return out



    g_mean = np.mean(g, axis=(0,1))
    f_mean = np.mean(f, axis=(0,1))
    g_var = np.var(g, axis=(0,1))
    f_var = np.var(g, axis=(0,1))
    f_ = (f-f_mean)/f_var
    g_ = (g-g_mean)/g_var
    out= cross_correlation(f_, g_)
    ### END YOUR CODE

    return out
