/*
 *  Photoconsistency-Visual-Odometry
 *  Multiscale Photoconsistency Visual Odometry from RGBD Images
 *  Copyright (c) 2014, Miguel Algaba Borrego
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

#ifndef CCAMERA_RECORD_H
#define CCAMERA_RECORD_H

#include "CSensorRecordBase.h"
#include "CImageReader.h"
#include <fstream>
#include "boost/filesystem/path.hpp"

namespace phovo
{
template< class TImageData, class TReferenceFrame >
class CCameraRecord :
  public CSensorRecordBase< TImageData, TReferenceFrame >
{
public:
  typedef CSensorRecordBase< TImageData, TReferenceFrame > Superclass;
  typedef CCameraRecord< TImageData, TReferenceFrame >     Self;
  typedef std::shared_ptr< Self >                          SharedPointer;

  typedef typename Superclass::SensorDataType          ImageDataType;
  typedef typename Superclass::ReferenceFrameType      ReferenceFrameType;
  typedef typename Superclass::SensorDataSharedPointer ImageDataSharedPointer;

  CCameraRecord() : Superclass()
  {}

  ~CCameraRecord()
  {}

  void Start()
  {
    // Open the camera record file
    this->m_CameraRecordFile.open( this->m_FileName );
    if( !this->m_CameraRecordFile.is_open() )
    {
      std::string errorMsg( "Unable to open camera record file " + this->m_FileName  );
      throw std::runtime_error( errorMsg );
    }
  }

  ImageDataSharedPointer GetSensorData()
  {
    bool retrievedData = false;
    while( this->m_CameraRecordFile.good() && !retrievedData )
    {
      std::string fileLine;
      std::getline( this->m_CameraRecordFile, fileLine );
      if( this->m_CameraRecordFile.eof() || fileLine[0] == '#' )
      {
        continue;
      }
      std::istringstream iss( fileLine );
      typename ImageDataType::TimeStampType timeStamp;
      std::string imageFileName;
      iss >> timeStamp;
      iss >> imageFileName;
      retrievedData = true;

      boost::filesystem::path cameraRecordDirectory( boost::filesystem::path( this->m_FileName ).parent_path() );
      boost::filesystem::path imagePath( cameraRecordDirectory / imageFileName );
      typename ImageDataType::DataSharedPointer image(
        new typename ImageDataType::DataType(
          CImageReader< typename ImageDataType::DataType >::ReadImage( imagePath.string() ) ) );
      this->m_SensorData.reset( new ImageDataType );
      this->m_SensorData->SetData( image );
      this->m_SensorData->SetTimeStamp( timeStamp );
    }

    if( !retrievedData )
    {
      this->m_SensorData.reset();
    }

    return this->m_SensorData;
  }

  void Stop()
  {
    this->m_CameraRecordFile.close();
  }

private:
  std::ifstream m_CameraRecordFile;

};
} //end namespace phovo
#endif
