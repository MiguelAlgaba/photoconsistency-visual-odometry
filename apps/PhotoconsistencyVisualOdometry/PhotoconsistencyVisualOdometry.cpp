/*
 *  Photoconsistency-Visual-Odometry
 *  Multiscale Photoconsistency Visual Odometry from RGBD Images
 *  Copyright (c) 2012-2014, Miguel Algaba Borrego
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

#include "CSensorIdentifier.h"
#include "CSensorData.h"
#include "CCameraRecord.h"
#include "CMultiSensorData.h"
#include "CMultiSensorDataSource.h"

#include <iomanip>
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp" //TickMeter

void printHelp()
{
  std::cout << "./PhotoconsistencyFrameAlignment <config_file.yml> "
            << "<rgbd_dataset_directory> "
            << "<output_trajectory_file>" << std::endl;
}

int parseInputArguments( int argc, char* argv[],
                         boost::filesystem::path & configFile,
                         boost::filesystem::path & rgbDataFile,
                         boost::filesystem::path & depthDataFile,
                         boost::filesystem::path & outputTrajectoryFile )
{
  if( argc<4 )
  {
    printHelp();
    return EXIT_FAILURE;
  }

  configFile = boost::filesystem::path( argv[1] );
  if( !boost::filesystem::exists( configFile ) )
  {
    std::cerr << "Input config file " << configFile << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  boost::filesystem::path rgbdDatasetDirectory( argv[2] );
  if( !boost::filesystem::exists( rgbdDatasetDirectory ) )
  {
    std::cerr << "Input RGBD dataset directory " << rgbdDatasetDirectory << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  rgbDataFile = rgbdDatasetDirectory / boost::filesystem::path( "rgb.txt" );
  if( !boost::filesystem::exists( rgbDataFile ) )
  {
    std::cerr << "Input RGB data file " << rgbDataFile << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  depthDataFile = rgbdDatasetDirectory / boost::filesystem::path( "depth.txt" );
  if( !boost::filesystem::exists( depthDataFile ) )
  {
    std::cerr << "Input depth data file " << depthDataFile << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  outputTrajectoryFile = boost::filesystem::path( argv[3] );
  boost::filesystem::path outputDirectory( outputTrajectoryFile.parent_path() );
  if( !boost::filesystem::exists( outputDirectory ) )
  {
    if( !boost::filesystem::create_directories( outputDirectory ) )
    {
      std::cerr << "Cannot create output directory " << outputDirectory << std::endl;
      return EXIT_FAILURE;
    }
  }

  return 0;
}

int main( int argc, char* argv[] )
{
  // Basic algebra typedefs
  typedef double CoordinateType;
  typedef phovo::Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  typedef phovo::Numeric::Matrix44RowMajor< CoordinateType > Matrix44Type;
  typedef phovo::Numeric::VectorCol3< CoordinateType >       Vector3Type;
  typedef phovo::Numeric::VectorCol6< CoordinateType >       Vector6Type;
  typedef Eigen::Quaternion< CoordinateType >                QuaternionType;

  // Internal sensor data typedefs
  typedef unsigned char              PixelType;
  typedef cv::Mat_< PixelType >      IntensityImageType;
  typedef cv::Mat_< CoordinateType > DepthImageType;

  // Multi-sensor data typedefs
  typedef double                                                         TimeStampType;
  typedef phovo::SensorIdentifierType                                    SensorIdentifierType;
  typedef phovo::CMultiSensorData< SensorIdentifierType, TimeStampType > MultiSensorDataType;

  // Multi-sensor data source typedefs (RGBD data record)
  typedef phovo::CSensorData< IntensityImageType, TimeStampType >        IntensityImageDataType;
  typedef phovo::CSensorData< DepthImageType, TimeStampType >            DepthImageDataType;
  typedef phovo::CCameraRecord< IntensityImageDataType, Vector6Type >    IntensityImageRecordType;
  typedef IntensityImageRecordType::ImageDataSharedPointer               IntensityImageDataSharedPointer;
  typedef phovo::CCameraRecord< DepthImageDataType, Vector6Type >        DepthImageRecordType;
  typedef DepthImageRecordType::ImageDataSharedPointer                   DepthImageDataSharedPointer;
  typedef phovo::CMultiSensorData< SensorIdentifierType, TimeStampType > MultiSensorDataType;
  typedef phovo::CMultiSensorDataSource< MultiSensorDataType >           MultiSensorDataSourceType;

  // Photo-consistency visual odometry typedefs
#if USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 0
  typedef phovo::Analytic::CPhotoconsistencyOdometryAnalytic< PixelType, CoordinateType > PhotoconsistencyVisualOdometryType;
#elif USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 1
  typedef phovo::Ceres::CPhotoconsistencyOdometryCeres< PixelType, CoordinateType > PhotoconsistencyVisualOdometryType;
#elif USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 2
  typedef phovo::Analytic::CPhotoconsistencyOdometryBiObjective< PixelType, CoordinateType > PhotoconsistencyVisualOdometryType;
#endif

  // Parse the input arguments
  boost::filesystem::path configFile;
  boost::filesystem::path rgbDataFile;
  boost::filesystem::path depthDataFile;
  boost::filesystem::path outputTrajectoryFile;
  CoordinateType depthScalingFactor = 1. / 5000.;
  if( parseInputArguments( argc, argv, configFile, rgbDataFile, depthDataFile, outputTrajectoryFile ) )
  {
    return EXIT_FAILURE;
  }

  // Set the photo-consistency visual odometry parameters
  Matrix33Type intrinsicMatrix;
  intrinsicMatrix << 517.3, 0., 318.6,
                     0., 516.5, 255.3,
                     0., 0., 1.;
  Matrix44Type pose = Matrix44Type::Identity();
  Vector6Type stateVector = Vector6Type::Zero();
  PhotoconsistencyVisualOdometryType photoconsistencyOdometry;
  photoconsistencyOdometry.ReadConfigurationFile( configFile.string() );
  photoconsistencyOdometry.SetIntrinsicMatrix( intrinsicMatrix );

  // Open the output trajectory file
  std::ofstream trajectoryFile( outputTrajectoryFile.string() );
  if( !trajectoryFile.is_open() )
  {
    std::cerr << "Cannot open output trajectory file " << outputTrajectoryFile.string() << std::endl;
    return EXIT_FAILURE;
  }
  trajectoryFile << "# estimated trajectory" << std::endl;
  trajectoryFile << "# timestamp tx ty tz qx qy qz qw" << std::endl;

  // Set multi-sensor data record and start retrieving frames
  IntensityImageRecordType::SharedPointer intensityImageRecord( new IntensityImageRecordType );
  intensityImageRecord->SetFileName( rgbDataFile.string() );
  DepthImageRecordType::SharedPointer depthImageRecord( new DepthImageRecordType );
  depthImageRecord->SetFileName( depthDataFile.string() );
  MultiSensorDataSourceType::SharedPointer multisensorDataSource( new MultiSensorDataSourceType );
  multisensorDataSource->SetSensorDataSource( phovo::IntensityCameraIdentifier, intensityImageRecord );
  multisensorDataSource->SetSensorDataSource( phovo::DepthCameraIdentifier, depthImageRecord );
  multisensorDataSource->Start();

  MultiSensorDataSourceType::MultiSensorDataSharedPointer previousIntensityDepthData =
    multisensorDataSource->GetMultiSensorData();
  if( previousIntensityDepthData )
  {
    // Extract the previous RGB and depth images from the multi-sensor data object
    IntensityImageType previousIntensityImage =
      *previousIntensityDepthData->GetData< IntensityImageDataType >( phovo::IntensityCameraIdentifier )->GetData();
    DepthImageType previousDepthImage =
      *previousIntensityDepthData->GetData< DepthImageDataType >( phovo::DepthCameraIdentifier )->GetData() * depthScalingFactor;

    MultiSensorDataSourceType::MultiSensorDataSharedPointer currentIntensityDepthData =
      multisensorDataSource->GetMultiSensorData();
    while( currentIntensityDepthData && ( cv::waitKey(5) != 'q' ) )
    {
      // Extract the current RGB and depth images from the multi-sensor data object
      IntensityImageDataSharedPointer currentIntensityImageData =
        currentIntensityDepthData->GetData< IntensityImageDataType >( phovo::IntensityCameraIdentifier );
      IntensityImageType currentIntensityImage = *currentIntensityImageData->GetData();
      DepthImageDataSharedPointer currentDepthImageData =
        currentIntensityDepthData->GetData< DepthImageDataType >( phovo::DepthCameraIdentifier );
      DepthImageType currentDepthImage = *currentDepthImageData->GetData() * depthScalingFactor;

      photoconsistencyOdometry.SetSourceFrame( previousIntensityImage, previousDepthImage );
      photoconsistencyOdometry.SetTargetFrame( currentIntensityImage, previousDepthImage );
      photoconsistencyOdometry.SetInitialStateVector( stateVector );

      // Optimize the problem to estimate the rigid transformation
      cv::TickMeter tm;tm.start();
      photoconsistencyOdometry.Optimize();
      tm.stop();
      std::cout << "Time = " << tm.getTimeSec() << " sec." << std::endl;

      // Update the global pose of the sensor
      Matrix44Type Rt = photoconsistencyOdometry.GetOptimalRigidTransformationMatrix();
      pose *= Rt.inverse();
      Matrix33Type R = pose.block( 0, 0, 3, 3 );
      Vector3Type t = pose.block( 0, 3, 3, 1 );
      QuaternionType q( R );

      // Write the current timestamped pose to the output trajectory file
      trajectoryFile << std::setprecision( std::numeric_limits< TimeStampType >::digits10 + 1 )
                     << currentIntensityImageData->GetTimeStamp() << " "
                     << t( 0 ) << " " << t( 1 ) << " " << t( 2 ) << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

      // Show results
      std::cout << "Rt:" << std::endl << Rt << std::endl;
      IntensityImageType warpedImage;
      phovo::WarpImage< PixelType, CoordinateType >( previousIntensityImage,
                                                     previousDepthImage,
                                                     warpedImage, Rt, intrinsicMatrix );
      IntensityImageType imgDiff;
      cv::absdiff( currentIntensityImage, warpedImage, imgDiff );
      cv::imshow( "imgDiff", imgDiff );

      // Update the previous and current frames
      previousIntensityImage = currentIntensityImage;
      previousDepthImage = currentDepthImage;
      currentIntensityDepthData = multisensorDataSource->GetMultiSensorData();
    }
  }

  // Stop grabbing sensor frames and close the output trajectory file
  multisensorDataSource->Stop();
  trajectoryFile.close();

  return EXIT_SUCCESS;
}

