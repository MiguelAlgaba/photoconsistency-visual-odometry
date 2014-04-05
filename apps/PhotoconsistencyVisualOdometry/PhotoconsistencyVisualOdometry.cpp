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

#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp" //TickMeter

void printHelp()
{
  std::cout << "./PhotoconsistencyFrameAlignment <config_file.yml> <rgbd_dataset_directory>" << std::endl;
}

int parseInputArguments( int argc, char* argv[],
                         boost::filesystem::path & configFile,
                         boost::filesystem::path & rgbDataFile,
                         boost::filesystem::path & depthDataFile )
{
  if( argc<3 )
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

  return 0;
}

int main( int argc, char* argv[] )
{
  // Basic algebra typedefs
  typedef double CoordinateType;
  typedef phovo::Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  typedef phovo::Numeric::Matrix44RowMajor< CoordinateType > Matrix44Type;
  typedef phovo::Numeric::VectorCol6< CoordinateType >       Vector6Type;

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
  if( parseInputArguments( argc, argv, configFile, rgbDataFile, depthDataFile ) )
  {
    return EXIT_FAILURE;
  }

  // Set multi-sensor data record and start retrieving frames
  IntensityImageRecordType::SharedPointer intensityImageRecord( new IntensityImageRecordType );
  intensityImageRecord->SetFileName( rgbDataFile.string() );
  DepthImageRecordType::SharedPointer depthImageRecord( new DepthImageRecordType );
  depthImageRecord->SetFileName( depthDataFile.string() );
  MultiSensorDataSourceType::SharedPointer multisensorDataSource( new MultiSensorDataSourceType );
  multisensorDataSource->SetSensorDataSource( phovo::IntensityCameraIdentifier, intensityImageRecord );
  multisensorDataSource->SetSensorDataSource( phovo::DepthCameraIdentifier, depthImageRecord );
  multisensorDataSource->Start();

  MultiSensorDataSourceType::MultiSensorDataSharedPointer intensityDepthData;
  do
  {
    // Extract the RGB and depth images from the multi-sensor data object (if available)
    intensityDepthData = multisensorDataSource->GetMultiSensorData();
    if( intensityDepthData )
    {
      IntensityImageDataSharedPointer intensityImageData =
        intensityDepthData->GetData< IntensityImageDataType >( phovo::IntensityCameraIdentifier );
      DepthImageDataSharedPointer depthImageData =
        intensityDepthData->GetData< DepthImageDataType >( phovo::DepthCameraIdentifier );
    }
  }
  while( intensityDepthData );
  multisensorDataSource->Stop();

  return EXIT_SUCCESS;
}

