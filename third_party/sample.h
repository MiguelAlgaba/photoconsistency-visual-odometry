// Copyright (c) 2012 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// Author: mierle@google.com (Keir Mierle)
//

#ifndef SAMPLE_H_
#define SAMPLE_H_

#include "opencv2/imgproc/imgproc.hpp"
#include "ceres/ceres.h"
#include "jet_extras.h"

void LinearInitAxis(double x, int size,
                    int *x1, int *x2,
                    double *dx)
{
  const int ix = static_cast<int>(x);
  if (ix < 0) {
    *x1 = 0;
    *x2 = 0;
    *dx = 1.0;
  } else if (ix > size - 2) {
    *x1 = size - 1;
    *x2 = size - 1;
    *dx = 1.0;
  } else {
    *x1 = ix;
    *x2 = ix + 1;
    *dx = *x2 - x;
  }
}

/// Linear interpolation.
template<typename T>
void SampleLinear(const cv::Mat & intensityImage,
                  const cv::Mat & intensityGradientX,
                  const cv::Mat & intensityGradientY,
                  double y, double x, T* sample) {

  int x1, y1, x2, y2;
  double dx, dy;

  // Take the upper left corner as integer pixel positions.
  x -= 0.5;
  y -= 0.5;

  LinearInitAxis(y, intensityImage.rows, &y1, &y2, &dy);
  LinearInitAxis(x, intensityImage.cols,  &x1, &x2, &dx);

  //Sample intensity
  const T im11 = T(intensityImage.at<float>(y1, x1));
  const T im12 = T(intensityImage.at<float>(y1, x2));
  const T im21 = T(intensityImage.at<float>(y2, x1));
  const T im22 = T(intensityImage.at<float>(y2, x2));

  sample[0] =(     dy  * ( dx * im11 + (1.0 - dx) * im12 ) +
           (1 - dy) * ( dx * im21 + (1.0 - dx) * im22 ));

  //Sample gradient x
  const T gradx11 = T(intensityGradientX.at<float>(y1, x1));
  const T gradx12 = T(intensityGradientX.at<float>(y1, x2));
  const T gradx21 = T(intensityGradientX.at<float>(y2, x1));
  const T gradx22 = T(intensityGradientX.at<float>(y2, x2));

  sample[1] =(     dy  * ( dx * gradx11 + (1.0 - dx) * gradx12 ) +
           (1 - dy) * ( dx * gradx21 + (1.0 - dx) * gradx22 ));

  //Sample gradient y
  const T grady11 = T(intensityGradientY.at<float>(y1, x1));
  const T grady12 = T(intensityGradientY.at<float>(y1, x2));
  const T grady21 = T(intensityGradientY.at<float>(y2, x1));
  const T grady22 = T(intensityGradientY.at<float>(y2, x2));

  sample[2] =(     dy  * ( dx * grady11 + (1.0 - dx) * grady12 ) +
           (1 - dy) * ( dx * grady21 + (1.0 - dx) * grady22 ));
}

// Sample the image at position (x, y) but use the gradient to
// propagate derivatives from x and y. This is needed to integrate the numeric
// image gradients with Ceres's autodiff framework.
template<typename T>
T SampleWithDerivative(const cv::Mat & intensityImage,
                       const cv::Mat & intensityGradientX,
                       const cv::Mat & intensityGradientY,
                       const T & x,
                       const T & y)
{
  double scalar_x = ceres::JetOps<T>::GetScalar(x);
  double scalar_y = ceres::JetOps<T>::GetScalar(y);

  double sample[3];
  // Sample intensity image and gradients
  SampleLinear(intensityImage,intensityGradientX,intensityGradientY, scalar_y, scalar_x, sample);
  T xy[2] = { x, y };
  return ceres::Chain<double, 2, T>::Rule(sample[0], sample + 1, xy);
}

#endif  // SAMPLE_H_
