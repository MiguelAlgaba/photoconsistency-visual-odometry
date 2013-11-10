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

#ifndef _CPHOTOCONSISTENCY_ODOMETRY_ANALYTIC_
#define _CPHOTOCONSISTENCY_ODOMETRY_ANALYTIC_

#define ENABLE_GAUSSIAN_BLUR 1
#define ENABLE_BOX_FILTER_BLUR 0
#define ENABLE_OPENMP_MULTITHREADING_ANALYTIC 0 // Enables OpenMP for CPhotoconsistencyOdometryAnalytic
#define ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS 0

#include "CPhotoconsistencyOdometry.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp" //TickMeter
#include <iostream>

namespace phovo
{

namespace Analytic
{

/*!This class computes the rigid (6DoF) transformation that best aligns a pair of RGBD frames using a photoconsistency maximization approach.
To estimate the rigid transformation, this class implements a coarse to fine approach. Thus, the algorithm starts finding a first pose approximation at
a low resolution level and uses the estimate to initialize the optimization at greater image scales. Both the residuals and jacobians are computed analytically.*/
template< class TPixel, class TCoordinate >
class CPhotoconsistencyOdometryAnalytic :
    public CPhotoconsistencyOdometry< TPixel, TCoordinate >
{
public:
  typedef CPhotoconsistencyOdometry< TPixel, TCoordinate > Superclass;

  typedef typename Superclass::CoordinateType     CoordinateType;
  typedef typename Superclass::IntensityImageType IntensityImageType;
  typedef typename Superclass::DepthImageType     DepthImageType;
  typedef typename Superclass::Matrix33Type       Matrix33Type;
  typedef typename Superclass::Matrix44Type       Matrix44Type;
  typedef typename Superclass::Vector6Type        Vector6Type;
  typedef typename Superclass::Vector4Type        Vector4Type;

private:
  typedef DepthImageType                            InternalIntensityImageType;
  typedef std::vector< InternalIntensityImageType > InternalIntensityImageContainerType;
  typedef std::vector< DepthImageType >             DepthImageContainerType;
  typedef std::vector< CoordinateType >             CoordinateContainerType;
  typedef std::vector< int >                        IntegerContainerType;

  /*!Intensity (gray), depth and gradient image pyramids. Each pyramid has 'numOptimizationLevels' levels.*/
  InternalIntensityImageContainerType m_IntensityPyramid0;
  InternalIntensityImageContainerType m_IntensityPyramid1;
  DepthImageContainerType             m_DepthPyramid0;
  DepthImageContainerType             m_DepthPyramid1;
  InternalIntensityImageContainerType m_IntensityGradientXPyramid1;
  InternalIntensityImageContainerType m_IntensityGradientYPyramid1;
  /*!Camera matrix (intrinsic parameters).*/
  Matrix33Type m_IntrinsicMatrix;
  /*!Current optimization level. Level 0 corresponds to the higher image resolution.*/
  int m_OptimizationLevel;
  /*!Number of optimization levels.*/
  int m_NumOptimizationLevels;
  /*!Scaling factor to update the state vector (at each level).*/
  CoordinateContainerType m_LambdaOptimizationSteps;
  /*!Size (in pixels) of the blur filter (at each level).*/
  IntegerContainerType m_BlurFilterSizes;
  /*!Scaling factor applied to the image gradients (at each level).*/
  CoordinateContainerType m_ImageGradientsScalingFactors;
  /*!Maximum number of iterations for the Gauss-Newton algorithm (at each level).*/
  IntegerContainerType m_MaxNumIterations;
  /*!Minimum gradient norm of the jacobian (at each level).*/
  CoordinateContainerType m_MinGradientNorms;
  /*!Enable the visualization of the optimization process (only for debug).*/
  bool m_VisualizeIterations;
  /*!State vector.*/
  Vector6Type m_StateVector; //Parameter vector (x y z yaw pitch roll)
  /*!Gradient of the error function.*/
  Vector6Type m_Gradients;
  /*!Current iteration at the current optimization level.*/
  int m_Iteration;
  /*!Minimum allowed depth to consider a depth pixel valid.*/
  CoordinateType m_MinDepth;
  /*!Maximum allowed depth to consider a depth pixel valid.*/
  CoordinateType m_MaxDepth;

template< class TImage >
void BuildPyramid( const TImage & img,
                   std::vector< TImage > & pyramid,
                   const int levels, const bool applyBlur )
{
  typedef TImage ImageType;

  //Create space for all the images
  pyramid.resize( levels );

  double factor = 1.;
  for( int level=0; level<levels; level++ )
  {
    //Create an auxiliar image of factor times the size of the original image
    ImageType imgAux;
    if( level!=0 )
    {
      cv::resize( img, imgAux, cv::Size(0,0), factor, factor );
    }
    else
    {
      imgAux = img;
    }

    //Blur the resized image with different filter size depending on the current pyramid level
    if( applyBlur )
    {
      int blurFilterSize = m_BlurFilterSizes[level];
      #if ENABLE_GAUSSIAN_BLUR
      if( blurFilterSize>0 )
      {
        cv::GaussianBlur( imgAux, imgAux, cv::Size( blurFilterSize, blurFilterSize ), 3 );
        cv::GaussianBlur( imgAux, imgAux, cv::Size( blurFilterSize, blurFilterSize ), 3 );
      }
      #elif ENABLE_BOX_FILTER_BLUR
      if( blurFilterSize>0 )
      {
        cv::blur( imgAux, imgAux, cv::Size( blurFilterSize, blurFilterSize ) );
        cv::blur( imgAux, imgAux, cv::Size( blurFilterSize, blurFilterSize ) );
      }
      #endif
    }

    //Assign the resized image to the current level of the pyramid
    pyramid[level] = imgAux;

    factor = factor/2;
  }
}

void BuildDerivativesPyramids( InternalIntensityImageContainerType & imagePyramid,
                               InternalIntensityImageContainerType & derXPyramid,
                               InternalIntensityImageContainerType & derYPyramid)
{
  //Compute image gradients
  double delta = 0.0;
  int ddepth = m_IntensityPyramid0[0].type();

  //Create space for all the derivatives images
  derXPyramid.resize(imagePyramid.size());
  derYPyramid.resize(imagePyramid.size());

  for( size_t level=0; level<imagePyramid.size(); level++ )
  {
    // Compute the gradient in x
    InternalIntensityImageType imgGray1_grad_x;
    cv::Scharr( imagePyramid[level], derXPyramid[level], ddepth, 1, 0,
                m_ImageGradientsScalingFactors[level], delta, cv::BORDER_DEFAULT );

    // Compute the gradient in y
    InternalIntensityImageType imgGray1_grad_y;
    cv::Scharr( imagePyramid[level], derYPyramid[level],ddepth, 0, 1,
                m_ImageGradientsScalingFactors[level], delta, cv::BORDER_DEFAULT );
  }
}

void ComputeResidualsAndJacobians( const InternalIntensityImageType & source_grayImg,
                                   const DepthImageType & source_depthImg,
                                   const InternalIntensityImageType & target_grayImg,
                                   const InternalIntensityImageType & target_gradXImg,
                                   const InternalIntensityImageType & target_gradYImg,
                                   Numeric::RowDynamicMatrixColMajor< CoordinateType, 1 > & residuals,
                                   Numeric::RowDynamicMatrixColMajor< CoordinateType, 6 > & jacobians,
                                   InternalIntensityImageType & warped_source_grayImage) const
{
  int nRows = source_grayImg.rows;
  int nCols = source_grayImg.cols;

  CoordinateType scaleFactor = 1.0/pow(2,m_OptimizationLevel);
  CoordinateType fx = m_IntrinsicMatrix(0,0)*scaleFactor;
  CoordinateType fy = m_IntrinsicMatrix(1,1)*scaleFactor;
  CoordinateType ox = m_IntrinsicMatrix(0,2)*scaleFactor;
  CoordinateType oy = m_IntrinsicMatrix(1,2)*scaleFactor;
  CoordinateType inv_fx = 1.f/fx;
  CoordinateType inv_fy = 1.f/fy;

  CoordinateType x = m_StateVector(0);
  CoordinateType y = m_StateVector(1);
  CoordinateType z = m_StateVector(2);
  CoordinateType yaw = m_StateVector(3);
  CoordinateType pitch = m_StateVector(4);
  CoordinateType roll = m_StateVector(5);

  //Compute the rigid transformation matrix from the parameters
  Matrix44Type Rt = Matrix44Type::Identity();
  CoordinateType sin_yaw = sin(yaw);
  CoordinateType cos_yaw = cos(yaw);
  CoordinateType sin_pitch = sin(pitch);
  CoordinateType cos_pitch = cos(pitch);
  CoordinateType sin_roll = sin(roll);
  CoordinateType cos_roll = cos(roll);
  Rt(0,0) = cos_yaw * cos_pitch;
  Rt(0,1) = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll;
  Rt(0,2) = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll;
  Rt(0,3) = x;
  Rt(1,0) = sin_yaw * cos_pitch;
  Rt(1,1) = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll;
  Rt(1,2) = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll;
  Rt(1,3) = y;
  Rt(2,0) = -sin_pitch;
  Rt(2,1) = cos_pitch * sin_roll;
  Rt(2,2) = cos_pitch * cos_roll;
  Rt(2,3) = z;
  Rt(3,0) = 0.0;
  Rt(3,1) = 0.0;
  Rt(3,2) = 0.0;
  Rt(3,3) = 1.0;

  CoordinateType temp1 = cos(pitch)*sin(roll);
  CoordinateType temp2 = cos(pitch)*cos(roll);
  CoordinateType temp3 = sin(pitch);
  CoordinateType temp4 = (sin(roll)*sin(yaw)+sin(pitch)*cos(roll)*cos(yaw));
  CoordinateType temp5 = (sin(pitch)*sin(roll)*cos(yaw)-cos(roll)*sin(yaw));
  CoordinateType temp6 = (sin(pitch)*sin(roll)*sin(yaw)+cos(roll)*cos(yaw));
  CoordinateType temp7 = (-sin(pitch)*sin(roll)*sin(yaw)-cos(roll)*cos(yaw));
  CoordinateType temp8 = (sin(roll)*cos(yaw)-sin(pitch)*cos(roll)*sin(yaw));
  CoordinateType temp9 = (sin(pitch)*cos(roll)*sin(yaw)-sin(roll)*cos(yaw));
  CoordinateType temp10 = cos(pitch)*sin(roll)*cos(yaw);
  CoordinateType temp11 = cos(pitch)*cos(yaw)+x;
  CoordinateType temp12 = cos(pitch)*cos(roll)*cos(yaw);
  CoordinateType temp13 = sin(pitch)*cos(yaw);
  CoordinateType temp14 = cos(pitch)*sin(yaw);
  CoordinateType temp15 = cos(pitch)*cos(yaw);
  CoordinateType temp16 = sin(pitch)*sin(roll);
  CoordinateType temp17 = sin(pitch)*cos(roll);
  CoordinateType temp18 = cos(pitch)*sin(roll)*sin(yaw);
  CoordinateType temp19 = cos(pitch)*cos(roll)*sin(yaw);
  CoordinateType temp20 = sin(pitch)*sin(yaw);
  CoordinateType temp21 = (cos(roll)*sin(yaw)-sin(pitch)*sin(roll)*cos(yaw));
  CoordinateType temp22 = cos(pitch)*cos(roll);
  CoordinateType temp23 = cos(pitch)*sin(roll);
  CoordinateType temp24 = cos(pitch);

  #if ENABLE_OPENMP_MULTITHREADING_ANALYTIC
  #pragma omp parallel for
  #endif
  for( int r=0; r<nRows; r++ )
  {
    for( int c=0; c<nCols; c++ )
    {
      int i = nCols*r+c; //vector index

      //Compute the 3D coordinates of the pij of the source frame
      Vector4Type point3D;
      point3D(2) = source_depthImg( r, c );
      if( m_MinDepth < point3D(2) && point3D(2) < m_MaxDepth )//Compute the jacobian only for the valid points
      {
        point3D(0) = (c - ox) * point3D(2) * inv_fx;
        point3D(1) = (r - oy) * point3D(2) * inv_fy;
        point3D(3) = 1.0;

        CoordinateType px = point3D(0);
        CoordinateType py = point3D(1);
        CoordinateType pz = point3D(2);

        //Transform the 3D point using the transformation matrix Rt
        Vector4Type transformedPoint3D = Rt*point3D;

        //Project the 3D point to the 2D plane
        CoordinateType inv_transformedPz = 1.0 / transformedPoint3D(2);
        CoordinateType transformed_c = (transformedPoint3D(0) * fx) * inv_transformedPz + ox; //transformed x (2D)
        CoordinateType transformed_r = (transformedPoint3D(1) * fy) * inv_transformedPz + oy; //transformed y (2D)
        int transformed_r_int = static_cast< int >( round( transformed_r ) );
        int transformed_c_int = static_cast< int >( round( transformed_c ) );

        //Asign the intensity value to the warped image and compute the difference between the transformed
        //pixel of frame 1 and the corresponding pixel of frame 2. Compute the error function
        if( ( transformed_r_int >= 0 && transformed_r_int < nRows ) &
            ( transformed_c_int >= 0 && transformed_c_int < nCols ) )
        {
          //Obtain the pixel values that will be used to compute the pixel residual
          // pixel1: Intensity value of the pixel(r,c) of the warped frame 1
          // pixel2: Intensity value of the pixel(r,c) of frame 2
          CoordinateType pixel1 = source_grayImg( r, c );
          CoordinateType pixel2 = target_grayImg( transformed_r_int, transformed_c_int );

          //Compute the pixel jacobian
          Numeric::FixedMatrixRowMajor< CoordinateType, 2, 6 > jacobianPrRt;
          CoordinateType temp25 = 1.0/(z+py*temp1+pz*temp2-px*temp3);
          CoordinateType temp26 = temp25*temp25;

          //Derivative with respect to x
          jacobianPrRt(0,0) = fx*temp25;
          jacobianPrRt(1,0) = 0.0;

          //Derivative with respect to y
          jacobianPrRt(0,1) = 0.0;
          jacobianPrRt(1,1) = fy*temp25;

          //Derivative with respect to z
          jacobianPrRt(0,2) = -fx*(pz*temp4+py*temp5+px*temp11)*temp26;
          jacobianPrRt(1,2) = -fy*(py*temp6+pz*temp9+px*temp14+y)*temp26;

          //Derivative with respect to yaw
          jacobianPrRt(0,3) = fx*(py*temp7+pz*temp8-px*temp14)*temp25;
          jacobianPrRt(1,3) = fy*(pz*temp4+py*temp5+px*temp15)*temp25;

          //Derivative with respect to pitch
          jacobianPrRt(0,4) = fx*(py*temp10+pz*temp12-px*temp13)*temp25
                              -fx*(-py*temp16-pz*temp17-px*temp24)*(pz*temp4+py*temp5+px*temp11)*temp26;
          jacobianPrRt(1,4) = fy*(py*temp18+pz*temp19-px*temp20)*temp25
                              -fy*(-py*temp16-pz*temp17-px*temp24)*(py*temp6+pz*temp9+px*temp14+y)*temp26;

          //Derivative with respect to roll
          jacobianPrRt(0,5) = fx*(py*temp4+pz*temp21)*temp25
                              -fx*(py*temp22-pz*temp23)*(pz*temp4+py*temp5+px*temp11)*temp26;
          jacobianPrRt(1,5) = fy*(pz*temp7+py*temp9)*temp25
                              -fy*(py*temp22-pz*temp23)*(py*temp6+pz*temp9+px*temp14+y)*temp26;

          //Apply the chain rule to compound the image gradients with the projective+RigidTransform jacobians
          Numeric::FixedRowVector< CoordinateType, 2 > target_imgGradient;
          target_imgGradient(0) = target_gradXImg(i);
          target_imgGradient(1) = target_gradYImg(i);
          Numeric::FixedRowVector< CoordinateType, 6 > jacobian = target_imgGradient*jacobianPrRt;

          //Assign the pixel residual and jacobian to its corresponding row
          jacobians(i,0)=jacobian(0,0);
          jacobians(i,1)=jacobian(0,1);
          jacobians(i,2)=jacobian(0,2);
          jacobians(i,3)=jacobian(0,3);
          jacobians(i,4)=jacobian(0,4);
          jacobians(i,5)=jacobian(0,5);

          residuals( nCols * transformed_r_int + transformed_c_int , 0 ) = pixel2 - pixel1;
          if( m_VisualizeIterations )
          {
            warped_source_grayImage( transformed_r_int, transformed_c_int ) = pixel1;
          }
        }
      }
    }
  }
}

enum TerminationCriteriaType
{
  NonTerminated = -1,
  MaxIterationsReached = 0,
  GradientNormLowerThanThreshold = 1
};

bool TestTerminationCriteria() const
{
  bool optimizationFinished = false;

  CoordinateType gradientNorm = m_Gradients.norm();

  TerminationCriteriaType terminationCriteria = NonTerminated;
  if( m_Iteration >= m_MaxNumIterations[ m_OptimizationLevel ] )
  {
    terminationCriteria = MaxIterationsReached;
    optimizationFinished = true;
  }
  else if( gradientNorm < m_MinGradientNorms[ m_OptimizationLevel ] )
  {
    terminationCriteria = GradientNormLowerThanThreshold;
    optimizationFinished = true;
  }

  if( optimizationFinished )
  {
    #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Optimization level: " << m_OptimizationLevel << std::endl;
    std::cout << "Termination criteria: ";
    #endif

    switch( terminationCriteria )
    {
      case MaxIterationsReached:
        #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << " Max number of iterations reached (" << m_MaxNumIterations[ m_OptimizationLevel ] << ")" << std::endl;;
        #endif
        break;
      case GradientNormLowerThanThreshold:
        #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        std::cout << " Gradient norm is lower than threshold (" << m_MinGradientNorms[ m_OptimizationLevel ] << ")" << std::endl;
        #endif
        break;
      default :
        break;
    }

    #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
    std::cout << "Number iterations: " << m_Iteration << std::endl;
    std::cout << "gradient norm: " << gradientNorm << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    #endif
  }

  return optimizationFinished;
}

public:

CPhotoconsistencyOdometryAnalytic() : m_MinDepth( 0.3 ), m_MaxDepth( 5.0 )
{
  m_StateVector.setZero();
  m_NumOptimizationLevels = 5;
  m_BlurFilterSizes.resize( m_NumOptimizationLevels, 0 );
  m_ImageGradientsScalingFactors.resize( m_NumOptimizationLevels, 0.0625 );
  m_LambdaOptimizationSteps.resize( m_NumOptimizationLevels, 1. );
  m_MaxNumIterations.resize( m_NumOptimizationLevels, 0 );
  m_MaxNumIterations[ 2 ] = 5;
  m_MaxNumIterations[ 3 ] = 20;
  m_MaxNumIterations[ 4 ] = 50;
  m_MinGradientNorms.resize( m_NumOptimizationLevels, 300. );
  m_VisualizeIterations = false;
}

~CPhotoconsistencyOdometryAnalytic(){}

/*!Sets the minimum depth distance (m) to consider a certain pixel valid.*/
void SetMinDepth( const CoordinateType minD )
{
  m_MinDepth = minD;
}

/*!Sets the maximum depth distance (m) to consider a certain pixel valid.*/
void SetMaxDepth( const CoordinateType maxD )
{
  m_MaxDepth = maxD;
}

/*!Sets the 3x3 intrinsic camera matrix*/
void SetIntrinsicMatrix( const Matrix33Type & intrinsicMatrix )
{
  m_IntrinsicMatrix = intrinsicMatrix;
}

/*!Sets the source (Intensity+Depth) frame.*/
void SetSourceFrame( const IntensityImageType & intensityImage,
                     const DepthImageType & depthImage )
{
  //Create an auxialiary image from the imput image
  InternalIntensityImageType intensityImageAux;
  intensityImage.convertTo( intensityImageAux, depthImage.type(), 1./255 );

  //Compute image pyramids for the grayscale and depth images
  BuildPyramid( intensityImageAux, m_IntensityPyramid0, m_NumOptimizationLevels, true );
  BuildPyramid( depthImage, m_DepthPyramid0, m_NumOptimizationLevels, false );
}

/*!Sets the source (Intensity+Depth) frame. Depth image is ignored*/
void SetTargetFrame( const IntensityImageType & intensityImage,
                     const DepthImageType & depthImage )
{
  //Create an auxialiary image from the imput image
  InternalIntensityImageType intensityImageAux;
  intensityImage.convertTo( intensityImageAux, depthImage.type(), 1./255 );

  //Compute image pyramids for the grayscale and depth images
  BuildPyramid( intensityImageAux, m_IntensityPyramid1, m_NumOptimizationLevels, true );

  //Compute image pyramids for the gradients images
  BuildDerivativesPyramids( m_IntensityPyramid1, m_IntensityGradientXPyramid1, m_IntensityGradientYPyramid1 );
}

/*!Initializes the state vector to a certain value. The optimization process uses the initial state vector as the initial estimate.*/
void SetInitialStateVector( const Vector6Type & initialStateVector )
{
  m_StateVector = initialStateVector;
}

/*!Launches the least-squares optimization process to find the configuration of the state vector parameters that maximizes the photoconsistency between the source and target frame.*/
void Optimize()
{
  for( m_OptimizationLevel = m_NumOptimizationLevels-1;
       m_OptimizationLevel >= 0; m_OptimizationLevel-- )
  {
    int nRows = m_IntensityPyramid0[ m_OptimizationLevel ].rows;
    int nCols = m_IntensityPyramid0[ m_OptimizationLevel ].cols;
    int nPoints = nRows * nCols;

    m_Iteration = 0;
    while(true)
    {
      #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
      cv::TickMeter tm;tm.start();
      #endif
      InternalIntensityImageType warpedSourceIntensityImage;
      if( m_VisualizeIterations )
        warpedSourceIntensityImage = InternalIntensityImageType::zeros( nRows, nCols );

      Numeric::RowDynamicMatrixColMajor< CoordinateType, 1 > residuals;
      residuals.resize( nPoints, Eigen::NoChange );
      residuals.setZero();
      Numeric::RowDynamicMatrixColMajor< CoordinateType, 6 > jacobians;
      jacobians.resize( nPoints, Eigen::NoChange );
      jacobians.setZero();

      if( m_MaxNumIterations[ m_OptimizationLevel] > 0 ) //compute only if the number of maximum iterations are greater than 0
      {
        ComputeResidualsAndJacobians(
                            m_IntensityPyramid0[ m_OptimizationLevel ],
                            m_DepthPyramid0[ m_OptimizationLevel ],
                            m_IntensityPyramid1[ m_OptimizationLevel ],
                            m_IntensityGradientXPyramid1[ m_OptimizationLevel ],
                            m_IntensityGradientYPyramid1[ m_OptimizationLevel ],
                            residuals,
                            jacobians,
                            warpedSourceIntensityImage );

        m_Gradients = jacobians.transpose()*residuals;
        m_StateVector = m_StateVector - m_LambdaOptimizationSteps[ m_OptimizationLevel ] *
            ((jacobians.transpose()*jacobians).inverse() * m_Gradients );

        #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
        tm.stop(); std::cout << "Iteration time = " << tm.getTimeSec() << " sec." << std::endl;
        #endif
      }

      m_Iteration++;

      if( TestTerminationCriteria() ){break;}

      if( m_VisualizeIterations )
      {
        InternalIntensityImageType imgDiff = InternalIntensityImageType::zeros( nRows, nCols );
        cv::absdiff( m_IntensityPyramid1[ m_OptimizationLevel ], warpedSourceIntensityImage, imgDiff );
        cv::imshow("optimize::imgDiff",imgDiff);
        cv::waitKey(0);
      }
    }
  }

  //After all the optimization process the optimization level is 0
  m_OptimizationLevel = 0;
}

/*!Returns the optimal state vector. This method has to be called after calling the Optimize() method.*/
Vector6Type GetOptimalStateVector() const
{
  return m_StateVector;
}

/*!Returns the optimal 4x4 rigid transformation matrix between the source and target frame. This method has to be called after calling the Optimize() method.*/
Matrix44Type GetOptimalRigidTransformationMatrix() const
{
  Matrix44Type Rt;
  eigenPose( m_StateVector(0), m_StateVector(1), m_StateVector(2),
             m_StateVector(3), m_StateVector(4), m_StateVector(5), Rt );
  return Rt;
}

/*!Reads the configuration parameters from a .yml file.*/
void ReadConfigurationFile( const std::string & fileName )
{
  cv::FileStorage fs( fileName, cv::FileStorage::READ );

  //Read the number of optimization levels
  fs["numOptimizationLevels"] >> m_NumOptimizationLevels;

  #if ENABLE_GAUSSIAN_BLUR || ENABLE_BOX_FILTER_BLUR
  //Read the blur filter size at every pyramid level
  fs["blurFilterSize (at each level)"] >> m_BlurFilterSizes;
  #endif

  //Read the scaling factor for each gradient image at each level
  fs["imageGradientsScalingFactor (at each level)"] >> m_ImageGradientsScalingFactors;

  //Read the lambda factor to change the optimization step
  fs["lambda_optimization_step (at each level)"] >> m_LambdaOptimizationSteps;

  //Read the number of Levenberg-Marquardt iterations at each optimization level
  fs["max_num_iterations (at each level)"] >> m_MaxNumIterations;

  //Read optimizer minimum gradient norm at each level
  fs["min_gradient_norm (at each level)"] >> m_MinGradientNorms;

  //Read the boolean value to determine if visualize the progress images or not
  fs["visualizeIterations"] >> m_VisualizeIterations;
}
};

} //end namespace Analytic

} //end namespace phovo

#endif
