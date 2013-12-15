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

#include "config.h"
//#ifdef PHOVO_WITH_CERES // Check for Ceres-solver

#ifndef _CPHOTOCONSISTENCY_ODOMETRY_CERES_
#define _CPHOTOCONSISTENCY_ODOMETRY_CERES_

#define ENABLE_GAUSSIAN_BLUR 1
#define ENABLE_BOX_FILTER_BLUR 0
#define ENABLE_OPENMP_MULTITHREADING_CERES 0 // Enables OpenMP for CPhotoconsistencyOdometryCeres

#include "CPhotoconsistencyOdometry.h"

#include "sample.h"
#include "jet_extras.h"
#include "ceres/ceres.h"
#include "opencv2/highgui/highgui.hpp" //visualize iterations

namespace phovo
{

namespace Ceres
{

/*!This class computes the rigid (6DoF) transformation that best aligns a pair of RGBD frames using a photoconsistency maximization approach.
To estimate the rigid transformation, this class implements a coarse to fine approach. Thus, the algorithm starts finding a first pose approximation at
a low resolution level and uses the estimate to initialize the optimization at greater image scales. This class uses Ceres autodifferentiation to compute the derivatives of the cost function.*/
template< class TPixel, class TCoordinate >
class CPhotoconsistencyOdometryCeres :
    public CPhotoconsistencyOdometry< TPixel, TCoordinate >
{
public:
  typedef CPhotoconsistencyOdometry< TPixel, TCoordinate > Superclass;

  typedef typename Superclass::PixelType          PixelType;
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
  /*!Size (in pixels) of the blur filter (at each level).*/
  IntegerContainerType m_BlurFilterSizes;
  /*!Scaling factor applied to the image gradients (at each level).*/
  CoordinateContainerType m_ImageGradientsScalingFactors;
  /*!Maximum number of iterations for the optimization algorithm (at each level).*/
  IntegerContainerType m_MaxNumIterations;
  /*!Enable the visualization of the optimization process (only for debug).*/
  bool m_VisualizeIterations;
  /*!State vector.*/
  CoordinateType m_StateVector[ 6 ]; //Parameter vector (x y z yaw pitch roll)
  /*!Current iteration at the current optimization level.*/
  int m_Iteration;
  CoordinateContainerType m_FunctionTolerances;
  CoordinateContainerType m_GradientTolerances;
  CoordinateContainerType m_ParameterTolerances;
  CoordinateContainerType m_InitialTrustRegionRadiuses;
  CoordinateContainerType m_MaxTrustRegionRadiuses;
  CoordinateContainerType m_MinTrustRegionRadiuses;
  CoordinateContainerType m_MinRelativeDecreases;
  int m_NumLinearSolverThreads;
  int m_NumThreads;
  bool m_MinimizerProgressToStdout;
  /*!Minimum allowed depth to consider a depth pixel valid.*/
  CoordinateType m_MinDepth;
  /*!Maximum allowed depth to consider a depth pixel valid.*/
  CoordinateType m_MaxDepth;

  class ResidualRGBDPhotoconsistency
  {
  private:
    Matrix33Type m_IntrinsicMatrix;
    int m_OptimizationLevel;
    InternalIntensityImageType m_SourceIntensityImage;
    DepthImageType m_SourceDepthImage;
    InternalIntensityImageType m_TargetIntensityImage;
    InternalIntensityImageType m_TargetGradXImage;
    InternalIntensityImageType m_TargetGradYImage;
    CoordinateType m_MaxDepth;
    CoordinateType m_MinDepth;

  public:
    ResidualRGBDPhotoconsistency( const Matrix33Type & intrinsicMatrix,
                                  const int optimizationLevel,
                                  const InternalIntensityImageType & sourceIntensityImage,
                                  const DepthImageType & sourceDepthImage,
                                  const InternalIntensityImageType & targetIntensityImage,
                                  const InternalIntensityImageType & targetGradXImage,
                                  const InternalIntensityImageType & targetGradYImage,
                                  const CoordinateType & maxDepth,
                                  const CoordinateType & minDepth ) :
      m_IntrinsicMatrix( intrinsicMatrix ), m_OptimizationLevel( optimizationLevel ),
      m_SourceIntensityImage( sourceIntensityImage ),
      m_SourceDepthImage( sourceDepthImage ),
      m_TargetIntensityImage( targetIntensityImage ),
      m_TargetGradXImage( targetGradXImage ),
      m_TargetGradYImage( targetGradYImage ),
      m_MaxDepth( maxDepth ),
      m_MinDepth( minDepth )
    {}

    template <typename T>
    bool operator()( const T* const stateVector,
                     T* residuals ) const
    {
      int nRows = m_SourceIntensityImage.rows;
      int nCols = m_SourceIntensityImage.cols;

      //Set camera parameters depending on the optimization level
      T fx = T( m_IntrinsicMatrix(0,0) ) / pow( 2, T( m_OptimizationLevel ) );
      T fy = T( m_IntrinsicMatrix(1,1) ) / pow( 2, T( m_OptimizationLevel ) );
      T inv_fx = T( 1. ) / fx;
      T inv_fy = T( 1. ) / fy;
      T ox = T( m_IntrinsicMatrix(0,2) ) / pow( 2, T( m_OptimizationLevel ) );
      T oy = T( m_IntrinsicMatrix(1,2) ) / pow( 2, T( m_OptimizationLevel ) );

      //Compute the rigid transformation matrix from the parameters
      T x = stateVector[0];
      T y = stateVector[1];
      T z = stateVector[2];
      T yaw = stateVector[3];
      T pitch = stateVector[4];
      T roll = stateVector[5];
      T Rt[4][4];
      T sin_yaw = sin(yaw);
      T cos_yaw = cos(yaw);
      T sin_pitch = sin(pitch);
      T cos_pitch = cos(pitch);
      T sin_roll = sin(roll);
      T cos_roll = cos(roll);
      Rt[0][0] = cos_yaw * cos_pitch;
      Rt[0][1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll;
      Rt[0][2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll;
      Rt[0][3] = x;
      Rt[1][0] = sin_yaw * cos_pitch;
      Rt[1][1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll;
      Rt[1][2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll;
      Rt[1][3] = y;
      Rt[2][0] = -sin_pitch;
      Rt[2][1] = cos_pitch * sin_roll;
      Rt[2][2] = cos_pitch * cos_roll;
      Rt[2][3] = z;
      Rt[3][0] = T(0.);
      Rt[3][1] = T(0.);
      Rt[3][2] = T(0.);
      Rt[3][3] = T(1.);

      //Initialize the error function (residuals) with an initial value
      #if ENABLE_OPENMP_MULTITHREADING_CERES
      #pragma omp parallel for
      #endif
      for( int r=0; r<nRows; r++ )
      {
        for( int c=0; c<nCols; c++ )
        {
          residuals[ nCols*r+c ] = T( 0. );
        }
      }

      T residualScalingFactor = T( 1. );

      #if ENABLE_OPENMP_MULTITHREADING_CERES
      #pragma omp parallel for
      #endif
      for( int r=0; r<nRows; r++ )
      {
        T point3D[4];
        T transformedPoint3D[4];
        T transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
        T pixel1; //Intensity value of the pixel(r,c) of the warped frame 1
        T pixel2; //Intensity value of the pixel(r,c) of frame 2

        for( int c=0; c<nCols; c++ )
        {
          if( m_MinDepth < m_SourceDepthImage(r,c) && m_SourceDepthImage(r,c) < m_MaxDepth ) //Compute the residuals only for the valid points
          {
            //Compute the local 3D coordinates of pixel(r,c) of frame 1
            point3D[2] = T(m_SourceDepthImage(r,c));         //z
            point3D[0] = (T(c)-ox) * point3D[2] * inv_fx;	   //x
            point3D[1] = (T(r)-oy) * point3D[2] * inv_fy;	   //y
            point3D[3] = T(1.0);			   //homogeneous coordinate

            //Transform the 3D point using the transformation matrix Rt
            transformedPoint3D[0] = Rt[0][0]*point3D[0]+Rt[0][1]*point3D[1]+Rt[0][2]*point3D[2]+Rt[0][3]*point3D[3];
            transformedPoint3D[1] = Rt[1][0]*point3D[0]+Rt[1][1]*point3D[1]+Rt[1][2]*point3D[2]+Rt[1][3]*point3D[3];
            transformedPoint3D[2] = Rt[2][0]*point3D[0]+Rt[2][1]*point3D[1]+Rt[2][2]*point3D[2]+Rt[2][3]*point3D[3];
            transformedPoint3D[3] = Rt[3][0]*point3D[0]+Rt[3][1]*point3D[1]+Rt[3][2]*point3D[2]+Rt[3][3]*point3D[3];

            //Project the 3D point to the 2D plane
            transformed_c = ((transformedPoint3D[0] * fx) / transformedPoint3D[2]) + ox; //transformed x (2D)
            transformed_r = ((transformedPoint3D[1] * fy) / transformedPoint3D[2]) + oy; //transformed y (2D)

            //Asign the intensity value to the warped image and compute the difference between the transformed
            //pixel of frame 1 and the corresponding pixel of frame 2. Compute the error function
            if( transformed_r >= T(0.) && transformed_r < T(nRows) &&
                transformed_c >= T(0.) && transformed_c < T(nCols) )
            {
              //Compute the proyected coordinates of the transformed 3D point
              int transformed_r_scalar = static_cast<int>(ceres::JetOps<T>::GetScalar(transformed_r));
              int transformed_c_scalar = static_cast<int>(ceres::JetOps<T>::GetScalar(transformed_c));

              //Compute the pixel residual
              pixel1 = T( m_SourceIntensityImage(r,c) );
              pixel2 = SampleWithDerivative< T, InternalIntensityImageType >( m_TargetIntensityImage,
                                             m_TargetGradXImage,
                                             m_TargetGradYImage, transformed_c, transformed_r );
              residuals[ nCols * transformed_r_scalar + transformed_c_scalar ] =
                residualScalingFactor * ( pixel2 - pixel1 );
            }
          }
        }
      }

      return true;
    }
  };

  class VisualizationCallback: public ceres::IterationCallback
  {
  private:
    Matrix33Type m_IntrinsicMatrix;
    Matrix44Type m_ExtrinsicMatrix;
    InternalIntensityImageType m_SourceIntensityImage;
    DepthImageType m_SourceDepthImage;
    InternalIntensityImageType m_TargetIntensityImage;

  public:
    VisualizationCallback( const Matrix33Type & intrinsicMatrix,
                           const Matrix44Type & extrinsicMatrix,
                           const InternalIntensityImageType & sourceIntensityImage,
                           const DepthImageType & sourceDepthImage,
                           const InternalIntensityImageType & targetIntensityImage ) : m_IntrinsicMatrix( intrinsicMatrix ),
      m_ExtrinsicMatrix( extrinsicMatrix ), m_SourceIntensityImage( sourceIntensityImage ),
      m_SourceDepthImage( sourceDepthImage ), m_TargetIntensityImage( targetIntensityImage )
    {}
    virtual ceres::CallbackReturnType operator()( const ceres::IterationSummary & summary )
    {
      std::cout << "Rt: " << std::endl << m_ExtrinsicMatrix << std::endl;
      InternalIntensityImageType warpedImage;
      phovo::WarpImage< CoordinateType, CoordinateType >( m_SourceIntensityImage, m_SourceDepthImage,
                                                          warpedImage, m_ExtrinsicMatrix, m_IntrinsicMatrix );
      InternalIntensityImageType imgDiff;
      cv::absdiff( m_TargetIntensityImage, warpedImage, imgDiff );
      cv::imshow( "callback: imgDiff", imgDiff );
      cv::waitKey( 5 );

      return ceres::SOLVER_CONTINUE;
    }
  };

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

public:

CPhotoconsistencyOdometryCeres() : m_MinDepth( 0.3 ), m_MaxDepth( 5.0 )
{
  this->SetInitialStateVector( Vector6Type::Zero() );
  m_NumOptimizationLevels = 5;
  m_OptimizationLevel = m_NumOptimizationLevels-1;
  m_Iteration = 0;
  m_BlurFilterSizes.resize( m_NumOptimizationLevels, 0 );
  m_ImageGradientsScalingFactors.resize( m_NumOptimizationLevels, 0.0625 );
  m_MaxNumIterations.resize( m_NumOptimizationLevels, 0 );
  m_MaxNumIterations[ 0 ] = 0;
  m_MaxNumIterations[ 1 ] = 0;
  m_MaxNumIterations[ 2 ] = 5;
  m_MaxNumIterations[ 3 ] = 20;
  m_MaxNumIterations[ 4 ] = 50;
  m_FunctionTolerances.resize( m_NumOptimizationLevels, 1e-4 );
  m_GradientTolerances.resize( m_NumOptimizationLevels, 1e-3 );
  m_ParameterTolerances.resize( m_NumOptimizationLevels, 1e-6 );
  m_ParameterTolerances[ 0 ] = 1e-4;
  m_ParameterTolerances[ 1 ] = 1e-4;
  m_InitialTrustRegionRadiuses.resize( m_NumOptimizationLevels, 1e4 );
  m_InitialTrustRegionRadiuses[ 0 ] = 1e8;
  m_MaxTrustRegionRadiuses.resize( m_NumOptimizationLevels, 1e8 );
  m_MinTrustRegionRadiuses.resize( m_NumOptimizationLevels, 1e-32 );
  m_MinRelativeDecreases.resize( m_NumOptimizationLevels, 1e-3 );
  m_VisualizeIterations = false;
  m_NumLinearSolverThreads = 1;
  m_NumThreads = 1;
  m_MinimizerProgressToStdout = false;
}

~CPhotoconsistencyOdometryCeres(){}

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
  m_StateVector[0] = initialStateVector( 0 );
  m_StateVector[1] = initialStateVector( 1 );
  m_StateVector[2] = initialStateVector( 2 );
  m_StateVector[3] = initialStateVector( 3 );
  m_StateVector[4] = initialStateVector( 4 );
  m_StateVector[5] = initialStateVector( 5 );
}

/*!Launches the least-squares optimization process to find the configuration of the state vector parameters that maximizes the photoconsistency between the source and target frame.*/
void Optimize()
{
  for( m_OptimizationLevel = m_NumOptimizationLevels-1;
       m_OptimizationLevel >= 0; m_OptimizationLevel-- )
  {
    if( m_MaxNumIterations[ m_OptimizationLevel] > 0 ) //compute only if the number of maximum iterations are greater than 0
    {
      int nRows = m_IntensityPyramid0[ m_OptimizationLevel ].rows;
      int nCols = m_IntensityPyramid0[ m_OptimizationLevel ].cols;
      int nPoints = nRows * nCols;

      // Build the problem.
      ceres::Problem problem;

      // Set up the only cost function (also known as residual). This uses
      // auto-differentiation to obtain the derivative (jacobian).
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ResidualRGBDPhotoconsistency,ceres::DYNAMIC,6>(
            new ResidualRGBDPhotoconsistency( m_IntrinsicMatrix, m_OptimizationLevel,
                                              m_IntensityPyramid0[ m_OptimizationLevel ],
                                              m_DepthPyramid0[ m_OptimizationLevel ],
                                              m_IntensityPyramid1[ m_OptimizationLevel ],
                                              m_IntensityGradientXPyramid1[ m_OptimizationLevel ],
                                              m_IntensityGradientYPyramid1[ m_OptimizationLevel ],
                                              m_MaxDepth,
                                              m_MinDepth ),
            nPoints /*dynamic size*/),
            NULL,
            m_StateVector );

      // Run the solver!
      ceres::Solver::Options options;
      options.max_num_iterations = m_MaxNumIterations[ m_OptimizationLevel ];
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;//ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = m_MinimizerProgressToStdout;
      options.function_tolerance = m_FunctionTolerances[ m_OptimizationLevel ];
      options.gradient_tolerance = m_GradientTolerances[ m_OptimizationLevel ];
      options.parameter_tolerance = m_ParameterTolerances[ m_OptimizationLevel ];
      options.initial_trust_region_radius = m_InitialTrustRegionRadiuses[ m_OptimizationLevel ];
      options.max_trust_region_radius = m_MaxTrustRegionRadiuses[ m_OptimizationLevel ];
      options.min_trust_region_radius = m_MinTrustRegionRadiuses[ m_OptimizationLevel ];
      options.min_relative_decrease = m_MinRelativeDecreases[ m_OptimizationLevel ];
      options.num_linear_solver_threads = m_NumLinearSolverThreads;
      options.num_threads = m_NumThreads;
      options.max_num_consecutive_invalid_steps = 0;
      VisualizationCallback callback = VisualizationCallback( m_IntrinsicMatrix, GetOptimalRigidTransformationMatrix(),
                                                              m_IntensityPyramid0[ m_OptimizationLevel ],
                                                              m_DepthPyramid0[ m_OptimizationLevel ],
                                                              m_IntensityPyramid1[ m_OptimizationLevel ] );
      if( m_VisualizeIterations )
      {
        options.update_state_every_iteration = true;
        options.callbacks.push_back( &callback );
      }
      else
      {
        options.update_state_every_iteration = false;
      }

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.BriefReport() << std::endl;
    }
  }

  //After all the optimization process the optimization level is 0
  m_OptimizationLevel = 0;
}

/*!Returns the optimal state vector. This method has to be called after calling the Optimize() method.*/
Vector6Type GetOptimalStateVector() const
{
  Vector6Type statevector;
  statevector(0) = m_StateVector[0];
  statevector(1) = m_StateVector[1];
  statevector(2) = m_StateVector[2];
  statevector(3) = m_StateVector[3];
  statevector(4) = m_StateVector[4];
  statevector(5) = m_StateVector[5];

  return statevector;
}

/*!Returns the optimal 4x4 rigid transformation matrix between the source and target frame. This method has to be called after calling the Optimize() method.*/
Matrix44Type GetOptimalRigidTransformationMatrix() const
{
  Matrix44Type Rt = PoseTranslationAndEulerAngles( m_StateVector[0], m_StateVector[1], m_StateVector[2],
                                                   m_StateVector[3], m_StateVector[4], m_StateVector[5] );
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

  //Read the number of Levenberg-Marquardt iterations at each optimization level
  fs["max_num_iterations (at each level)"] >> m_MaxNumIterations;

  //Read optimizer function tolerance at each level
  fs["function_tolerance (at each level)"] >> m_FunctionTolerances;

  //Read optimizer gradient tolerance at each level
  fs["gradient_tolerance (at each level)"] >> m_GradientTolerances;

  //Read optimizer parameter tolerance at each level
  fs["parameter_tolerance (at each level)"] >> m_ParameterTolerances;

  //Read optimizer initial trust region at each level
  fs["initial_trust_region_radius (at each level)"] >> m_InitialTrustRegionRadiuses;

  //Read optimizer max trust region radius at each level
  fs["max_trust_region_radius (at each level)"] >> m_MaxTrustRegionRadiuses;

  //Read optimizer min trust region radius at each level
  fs["min_trust_region_radius (at each level)"] >> m_MinTrustRegionRadiuses;

  //Read optimizer min LM relative decrease at each level
  fs["min_relative_decrease (at each level)"] >> m_MinRelativeDecreases;

  //Read the number of threads for the linear solver
  fs["num_linear_solver_threads"] >> m_NumLinearSolverThreads;

  //Read the number of threads for the jacobian computation
  fs["num_threads"] >> m_NumThreads;

  //Read the boolean value to determine if print the minimization progress or not
  fs["minimizer_progress_to_stdout"] >> m_MinimizerProgressToStdout;

  //Read the boolean value to determine if visualize the progress images or not
  fs["visualizeIterations"] >> m_VisualizeIterations;
}
};

} //end namespace Ceres

} //end namespace phovo

#endif

//#endif  // Check for Ceres-solver
