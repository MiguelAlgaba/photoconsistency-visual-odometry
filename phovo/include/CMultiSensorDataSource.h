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

#ifndef CMULTI_SENSOR_DATA_SOURCE_H
#define CMULTI_SENSOR_DATA_SOURCE_H

#include <map>
#include <memory>
#include "CDataSourceBase.h"

namespace phovo
{
template< class TMultiSensorData >
class CMultiSensorDataSource
{
public:
  typedef TMultiSensorData                                      MultiSensorDataType;
  typedef typename MultiSensorDataType::TimeStampType           TimeStampType;
  typedef typename MultiSensorDataType::SensorDataType          SensorDataType;
  typedef typename MultiSensorDataType::SensorDataSharedPointer SensorDataSharedPointer;
  typedef typename MultiSensorDataType::SensorIdentifierType    SensorIdentifierType;
  typedef typename MultiSensorDataType::SharedPointer           MultiSensorDataSharedPointer;

  typedef CDataSourceBase< TimeStampType >             SensorDataSourceType;
  typedef typename SensorDataSourceType::SharedPointer SensorDataSourceSharedPointer;

  typedef CMultiSensorDataSource< MultiSensorDataType > Self;
  typedef std::shared_ptr< Self >                       SharedPointer;

  CMultiSensorDataSource() : m_MultiSensorData( new MultiSensorDataType ),
    m_SensorDataSourceMap( SensorIdentifierSensorDataSourceMapType() )
  {}

  ~CMultiSensorDataSource()
  {}

  void SetSensorDataSource( const SensorIdentifierType & sensorId,
                            const SensorDataSourceSharedPointer sensorDataSource )
  {
    this->m_SensorDataSourceMap.insert(
      typename SensorIdentifierSensorDataSourceMapType::value_type( sensorId, sensorDataSource ) );
  }

  MultiSensorDataSharedPointer GetMultiSensorData()
  {
    this->m_MultiSensorData.reset( new MultiSensorDataType );
    typename SensorIdentifierSensorDataSourceMapType::iterator sdsIt = this->m_SensorDataSourceMap.begin();
    typename SensorIdentifierSensorDataSourceMapType::iterator sdsItEnd = this->m_SensorDataSourceMap.end();
    while( sdsIt != sdsItEnd )
    {
      SensorDataSharedPointer sensorData = sdsIt->second->GetData();
      if( !sensorData )
      {
        this->m_MultiSensorData.reset();
        break;
      }
      this->m_MultiSensorData->SetData( sdsIt->first, sensorData );
      ++sdsIt;
    }
    return this->m_MultiSensorData;
  }

  void Start()
  {
    typename SensorIdentifierSensorDataSourceMapType::iterator sdsIt = this->m_SensorDataSourceMap.begin();
    typename SensorIdentifierSensorDataSourceMapType::iterator sdsItEnd = this->m_SensorDataSourceMap.end();
    while( sdsIt != sdsItEnd )
    {
      sdsIt->second->Start();
      ++sdsIt;
    }
  }

  void Stop()
  {
    typename SensorIdentifierSensorDataSourceMapType::iterator sdsIt = this->m_SensorDataSourceMap.begin();
    typename SensorIdentifierSensorDataSourceMapType::iterator sdsItEnd = this->m_SensorDataSourceMap.end();
    while( sdsIt != sdsItEnd )
    {
      sdsIt->second->Stop();
      ++sdsIt;
    }
  }

protected:
  typedef std::map< SensorIdentifierType, SensorDataSourceSharedPointer >
    SensorIdentifierSensorDataSourceMapType;

  MultiSensorDataSharedPointer            m_MultiSensorData;
  SensorIdentifierSensorDataSourceMapType m_SensorDataSourceMap;

private:
  CMultiSensorDataSource( const Self & ); // purposely not implemented
  void operator = ( const Self & );       // purposely not implemented
};
} //end namespace phovo
#endif
