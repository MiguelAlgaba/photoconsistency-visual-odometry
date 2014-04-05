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

#ifndef CMULTI_SENSOR_DATA_H
#define CMULTI_SENSOR_DATA_H

#include <map>
#include <memory>
#include "CDataBase.h"

namespace phovo
{
template< class TSensorIdentifier, class TTimeStamp >
class CMultiSensorData
{
public:
  typedef TSensorIdentifier                      SensorIdentifierType;
  typedef TTimeStamp                             TimeStampType;
  typedef CDataBase< TimeStampType >             SensorDataType;
  typedef typename SensorDataType::SharedPointer SensorDataSharedPointer;

  typedef CMultiSensorData        Self;
  typedef std::shared_ptr< Self > SharedPointer;

  CMultiSensorData() : m_SensorDataMap( SensorIdentifierDataMapType() )
  {}

  ~CMultiSensorData()
  {}

  void SetData( const SensorIdentifierType & sensorId, const SensorDataSharedPointer data )
  {
    this->m_SensorDataMap.insert( typename SensorIdentifierDataMapType::value_type( sensorId, data ) );
  }

  template< class TSensorData >
  typename TSensorData::SharedPointer GetData( const SensorIdentifierType & sensorId ) const
  {
    typename TSensorData::SharedPointer sensorData;
    typename SensorIdentifierDataMapType::const_iterator it = this->m_SensorDataMap.find( sensorId );
    typename SensorIdentifierDataMapType::const_iterator itEnd = this->m_SensorDataMap.end();
    if( it != itEnd )
    {
      sensorData = std::static_pointer_cast< TSensorData >( it->second );
    }
    return sensorData;
  }

private:
  typedef std::map< SensorIdentifierType, SensorDataSharedPointer > SensorIdentifierDataMapType;

  SensorIdentifierDataMapType m_SensorDataMap;

  CMultiSensorData( const CMultiSensorData & ); // purposely not implemented
  void operator = ( const CMultiSensorData & ); // purposely not implemented
};
} //end namespace phovo
#endif
