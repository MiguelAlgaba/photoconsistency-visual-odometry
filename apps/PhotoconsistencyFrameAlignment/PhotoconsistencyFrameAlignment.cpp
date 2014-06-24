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

#define USE_PHOTOCONSISTENCY_ODOMETRY_METHOD 0 // CPhotoconsistencyOdometryAnalytic: 0
                                               // CPhotoconsistencyOdometryCeres: 1
                                               // CPhotoconsistencyOdometryBiObjective: 2

#if USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 0
  #include "CPhotoconsistencyOdometryAnalytic.h"
#elif USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 1
  #include "CPhotoconsistencyOdometryCeres.h"
#elif USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 2
  #include "CPhotoconsistencyOdometryBiObjective.h"
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp" //TickMeter

void printHelp()
{
  std::cout<<"./PhotoconsistencyFrameAlignment <config_file.yml> <imgRGB0.png> <imgDepth0.png> <imgRGB1.png> <imgDepth1.png>"<<std::endl;
}

int main (int argc,char ** argv)
{
  if(argc<5){printHelp();return -1;}

  typedef double CoordinateType;
  typedef phovo::Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  typedef phovo::Numeric::Matrix44RowMajor< CoordinateType > Matrix44Type;
  typedef phovo::Numeric::VectorCol6< CoordinateType >       Vector6Type;

  typedef unsigned char PixelType;
  typedef cv::Mat_< PixelType >      IntensityImageType;
  typedef cv::Mat_< CoordinateType > DepthImageType;

  //Set the camera parameters
  Matrix33Type intrinsicMatrix;
  intrinsicMatrix << 525., 0., 319.5,
                     0., 525., 239.5,
                     0., 0., 1.;

  //Load two RGB frames (RGB and depth images)
  IntensityImageType imgGray0 = cv::imread(argv[2],0);
  DepthImageType imgDepth0 = cv::imread(argv[3],-1);
  imgDepth0 = imgDepth0 * 1. / 1000.; // convert the depth image to meters

  IntensityImageType imgGray1 = cv::imread(argv[4],0);
  DepthImageType imgDepth1 = cv::imread(argv[5],-1);
  imgDepth1 = imgDepth1 * 1. / 1000.; // convert the depth image to meters

  //Define the photoconsistency odometry object and set the input parameters
#if USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 0
  phovo::Analytic::CPhotoconsistencyOdometryAnalytic< PixelType, CoordinateType > photoconsistencyOdometry;
#elif USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 1
  phovo::Ceres::CPhotoconsistencyOdometryCeres< PixelType, CoordinateType > photoconsistencyOdometry;
#elif USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 2
  phovo::Analytic::CPhotoconsistencyOdometryBiObjective< PixelType, CoordinateType > photoconsistencyOdometry;
#endif
  Vector6Type stateVector;
  stateVector << 0., 0., 0., 0., 0., 0.; //t0,t1,t2,w0,w1,w2
  photoconsistencyOdometry.ReadConfigurationFile( std::string( argv[1] ) );
  photoconsistencyOdometry.SetIntrinsicMatrix( intrinsicMatrix );
  photoconsistencyOdometry.SetSourceFrame( imgGray0, imgDepth0 );
  photoconsistencyOdometry.SetTargetFrame( imgGray1, imgDepth1 );
  photoconsistencyOdometry.SetInitialStateVector( stateVector );

  //Optimize the problem to estimate the rigid transformation
  cv::TickMeter tm;tm.start();
  photoconsistencyOdometry.Optimize();
  tm.stop();
  std::cout << "Time = " << tm.getTimeSec() << " sec." << std::endl;

  //Show results
  Matrix44Type Rt = photoconsistencyOdometry.GetOptimalRigidTransformationMatrix();
  std::cout << "main::Rt eigen:" << std::endl << Rt << std::endl;
  IntensityImageType warpedImage;
  phovo::WarpImage< PixelType, CoordinateType >( imgGray0, imgDepth0, warpedImage, Rt, intrinsicMatrix );
  IntensityImageType imgDiff;
  cv::absdiff( imgGray1, warpedImage, imgDiff );
  cv::imshow( "main::imgDiff", imgDiff );
  cv::waitKey( 0 );

	return 0;
}

