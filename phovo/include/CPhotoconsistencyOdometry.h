/*
 *  Photoconsistency-Visual-Odometry
 *  Multiscale Photoconsistency Visual Odometry from RGBD Images
 *  Copyright (c) 2012-2013, Miguel Algaba Borrego
 *
 *  http://code.google.com/p/photoconsistency-visual-odometry/
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of the holder(s) nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef _CPHOTOCONSISTENCY_ODOMETRY_
#define _CPHOTOCONSISTENCY_ODOMETRY_

#define ENABLE_OPENMP_MULTITHREADING_WARP_IMAGE 0

#include "opencv2/imgproc/imgproc.hpp"
#include "eigen3/Eigen/Dense"

#include "Matrix.h"

namespace phovo
{

template< class T >
Numeric::Matrix44RowMajor< T >
PoseTranslationAndEulerAngles( const T x, const T y, const T z,
                               const T yaw, const T pitch, const T roll )
{
  typedef T CoordinateType;
  typedef Numeric::Matrix44RowMajor< CoordinateType > Matrix44Type;
  Matrix44Type pose;
  pose(0,0) = cos(yaw) * cos(pitch);
  pose(0,1) = cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll);
  pose(0,2) = cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll);
  pose(0,3) = x;

  pose(1,0) = sin(yaw) * cos(pitch);
  pose(1,1) = sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll);
  pose(1,2) = sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll);
  pose(1,3) = y;

  pose(2,0) = -sin(pitch);
  pose(2,1) = cos(pitch) * sin(roll);
  pose(2,2) = cos(pitch) * cos(roll);
  pose(2,3) = z;

  pose(3,0) = 0;
  pose(3,1) = 0;
  pose(3,2) = 0;
  pose(3,3) = 1;

  return pose;
}

template< class T >
Numeric::Matrix33RowMajor< T >
Hat( const Numeric::VectorCol3< T > & v )
{
  typedef T CoordinateType;
  typedef Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  Matrix33Type m = Matrix33Type::Zero();
  m( 0, 1 ) = -v( 2 );
  m( 1, 0 ) = v( 2 );
  m( 0, 2 ) = v( 1 );
  m( 2, 0 ) = -v( 1 );
  m( 1, 2 ) = -v( 0 );
  m( 2, 1 ) = v( 0 );

  return m;
}

template< class T >
Numeric::Matrix33RowMajor< T >
Rodrigues( const Numeric::VectorCol3< T > & w )
{
  typedef T CoordinateType;
  typedef Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  Matrix33Type R = Matrix33Type::Identity();
  CoordinateType phi = w.norm();
  if( phi > 100. * std::numeric_limits< CoordinateType >::epsilon() )
  {
    CoordinateType inv_phi = 1. / phi;
    Matrix33Type w_hat = Hat( w );
    R += inv_phi * w_hat * sin( phi ) +
      ( inv_phi * inv_phi ) * w_hat * w_hat * ( 1. - cos( phi ) );
  }

  return R;
}

template< class T >
Numeric::Matrix44RowMajor< T >
PoseExponentialMap( const T t0, const T t1, const T t2,
                    const T w0, const T w1, const T w2 )
{
  typedef T CoordinateType;
  typedef Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  typedef Numeric::Matrix44RowMajor< CoordinateType > Matrix44Type;
  typedef Numeric::VectorCol3< CoordinateType >       Vector3Type;
  Matrix44Type pose = Matrix44Type::Identity();
  Vector3Type t; t << t0, t1, t2;
  Vector3Type w; w << w0, w1, w2;
  CoordinateType phi = w.norm();
  if( phi > 100. * std::numeric_limits< CoordinateType >::epsilon() )
  {
    Matrix33Type w_hat = Hat( w );
    Matrix33Type R = Rodrigues( w );
    pose.block( 0, 0, 3, 3 ) = R;
    pose.block( 0, 3, 3, 1 ) = ( 1. / phi ) * ( ( Matrix33Type::Identity() - R ) * w_hat * t +
      w * w.transpose() * t );
  }
  else
  {
    pose.block( 0, 3, 3, 1 ) = t;
  }

  return pose;
}

template< class TPixel, class TCoordinate >
void WarpImage( const cv::Mat_< TPixel > & intensityImage,
                const cv::Mat_< TCoordinate > & depthImage,
                cv::Mat_< TPixel > & warpedIntensityImage,
                const Numeric::Matrix44RowMajor< TCoordinate > & Rt,
                const Numeric::Matrix33RowMajor< TCoordinate > & intrinsicMatrix,
                const int level = 0 )
{
  typedef TPixel                PixelType;
  typedef cv::Mat_< PixelType > IntensityImageType;

  typedef TCoordinate                           CoordinateType;
  typedef Numeric::VectorCol4< CoordinateType > Vector4Type;

  CoordinateType fx = intrinsicMatrix(0,0)/pow(2,level);
  CoordinateType fy = intrinsicMatrix(1,1)/pow(2,level);
  CoordinateType inv_fx = 1.f/fx;
  CoordinateType inv_fy = 1.f/fy;
  CoordinateType ox = intrinsicMatrix(0,2)/pow(2,level);
  CoordinateType oy = intrinsicMatrix(1,2)/pow(2,level);

  Vector4Type point3D;
  Vector4Type transformedPoint3D;
  int transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1

  warpedIntensityImage = cv::Mat_< PixelType >::zeros( intensityImage.rows, intensityImage.cols );

  #if ENABLE_OPENMP_MULTITHREADING_WARP_IMAGE
  #pragma omp parallel for private( point3D, transformedPoint3D, transformed_r, transformed_c )
  #endif
  for( int r=0; r<intensityImage.rows; r++)
  {
    for( int c=0; c<intensityImage.cols; c++)
    {
      if( depthImage(r,c)>0 ) //If has valid depth value
      {
        //Compute the local 3D coordinates of pixel(r,c) of frame 1
        point3D(2) = depthImage(r,c);              //z
        point3D(0) = (c-ox) * point3D(2) * inv_fx; //x
        point3D(1) = (r-oy) * point3D(2) * inv_fy; //y
        point3D(3) = 1.0;                          //homogeneous coordinate

        //Transform the 3D point using the transformation matrix Rt
        transformedPoint3D = Rt * point3D;

        //Project the 3D point to the 2D plane
        transformed_c = static_cast< int >( ( ( transformedPoint3D(0) * fx ) /
                                              transformedPoint3D(2) ) + ox ); //transformed x (2D)
        transformed_r = static_cast< int >( ( ( transformedPoint3D(1) * fy ) /
                                              transformedPoint3D(2) ) + oy ); //transformed y (2D)

        //Asign the intensity value to the warped image and compute the difference between the transformed
        //pixel of frame 1 and the corresponding pixel of frame 2. Compute the error function
        if( transformed_r >= 0 && transformed_r < intensityImage.rows &
            transformed_c >= 0 && transformed_c < intensityImage.cols)
        {
          warpedIntensityImage( transformed_r, transformed_c ) = intensityImage( r, c );
        }
      }
    }
  }
}

/*!This abstract class defines the mandatory methods that any derived class must implement to compute the rigid (6DoF) transformation that best aligns a pair of RGBD frames using a photoconsistency maximization approach.*/
template< class TPixel, class TCoordinate >
class CPhotoconsistencyOdometry
{
public:
  typedef TPixel                PixelType;
  typedef cv::Mat_< PixelType > IntensityImageType;

  typedef TCoordinate                CoordinateType;
  typedef cv::Mat_< CoordinateType > DepthImageType;

  typedef Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  typedef Numeric::Matrix44RowMajor< CoordinateType > Matrix44Type;
  typedef Numeric::VectorCol6< CoordinateType >       Vector6Type;
  typedef Numeric::VectorCol4< CoordinateType >       Vector4Type;
  typedef Numeric::VectorCol3< CoordinateType >       Vector3Type;

  /*!Sets the 3x3 intrinsic pinhole matrix.*/
  virtual void SetIntrinsicMatrix( const Matrix33Type & intrinsicMatrix ) = 0;

  /*!Sets the source (Intensity+Depth) frame.*/
  virtual void SetSourceFrame( const IntensityImageType & intensityImage,
                               const DepthImageType & depthImage ) = 0;

  /*!Sets the source (Intensity+Depth) frame.*/
  virtual void SetTargetFrame( const IntensityImageType & intensityImage,
                               const DepthImageType & depthImage ) = 0;

  /*!Initializes the state vector to a certain value. The optimization process uses
   *the initial state vector as the initial estimate.*/
  virtual void SetInitialStateVector( const Vector6Type & initialStateVector ) = 0;

  /*!Launches the least-squares optimization process to find the configuration of the
   *state vector parameters that maximizes the photoconsistency between the source and
   *target frame.*/
  virtual void Optimize() = 0;

  /*!Returns the optimal state vector. This method has to be called after calling the
   *Optimize() method.*/
  virtual Vector6Type GetOptimalStateVector() const = 0;

  /*!Returns the optimal 4x4 rigid transformation matrix between the source and target frame.
   *This method has to be called after calling the Optimize() method.*/
  virtual Matrix44Type GetOptimalRigidTransformationMatrix() const = 0;
};

} //end namespace phovo

#endif
