/*
 *  Photoconsistency-Visual-Odometry
 *  Multiscale Photoconsistency Visual Odometry from RGBD Images
 *  Copyright (c) 2013, Miguel Algaba Borrego
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

#ifndef MATRIX_H
#define MATRIX_H

#include "eigen3/Eigen/Core"

namespace phovo
{
namespace Numeric
{ 
// --- Dynamic matrix
template< class T >
class DynamicMatrixRowMajor :
    public Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor | Eigen::AutoAlign >
{
public:
  typedef Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor | Eigen::AutoAlign > Base;

  DynamicMatrixRowMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  DynamicMatrixRowMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  DynamicMatrixRowMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T >
class DynamicMatrixColMajor :
    public Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor | Eigen::AutoAlign >
{
public:
  typedef Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::ColMajor | Eigen::AutoAlign > Base;

  DynamicMatrixColMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  DynamicMatrixColMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  DynamicMatrixColMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

// --- Row dynamic matrix
template< class T, int cols >
class RowDynamicMatrixRowMajor :
    public Eigen::Matrix< T, Eigen::Dynamic, cols,
                          Eigen::RowMajor | Eigen::AutoAlign >
{
public:
  typedef Eigen::Matrix< T, Eigen::Dynamic, cols,
                         Eigen::RowMajor | Eigen::AutoAlign > Base;

  RowDynamicMatrixRowMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  RowDynamicMatrixRowMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  RowDynamicMatrixRowMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T, int cols >
class RowDynamicMatrixColMajor :
    public Eigen::Matrix< T, Eigen::Dynamic, cols,
                          Eigen::ColMajor | Eigen::AutoAlign >
{
public:
  typedef Eigen::Matrix< T, Eigen::Dynamic, cols,
                         Eigen::ColMajor | Eigen::AutoAlign > Base;

  RowDynamicMatrixColMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  RowDynamicMatrixColMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  RowDynamicMatrixColMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

// --- Col dynamic matrix
template< class T, int rows >
class ColDynamicMatrixRowMajor :
    public Eigen::Matrix< T, rows, Eigen::Dynamic,
                          Eigen::RowMajor | Eigen::AutoAlign >
{
public:
  typedef Eigen::Matrix< T, rows, Eigen::Dynamic,
                         Eigen::RowMajor | Eigen::AutoAlign > Base;

  ColDynamicMatrixRowMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  ColDynamicMatrixRowMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  ColDynamicMatrixRowMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T, int rows >
class ColDynamicMatrixColMajor :
    public Eigen::Matrix< T, rows, Eigen::Dynamic,
                          Eigen::ColMajor | Eigen::AutoAlign >
{
public:
  typedef Eigen::Matrix< T, rows, Eigen::Dynamic,
                         Eigen::ColMajor | Eigen::AutoAlign > Base;

  ColDynamicMatrixColMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  ColDynamicMatrixColMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  ColDynamicMatrixColMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

// --- Fixed matrix
template< class T, int rows, int cols >
class FixedMatrixRowMajor :
    public Eigen::Matrix< T, rows, cols, Eigen::RowMajor | Eigen::AutoAlign >
{
public:
  typedef Eigen::Matrix< T, rows, cols, Eigen::RowMajor | Eigen::AutoAlign > Base;

  FixedMatrixRowMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  FixedMatrixRowMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  FixedMatrixRowMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T, int rows, int cols >
class FixedMatrixColMajor :
    public Eigen::Matrix< T, rows, cols, Eigen::ColMajor | Eigen::AutoAlign >
{
public:
  typedef Eigen::Matrix< T, rows, cols, Eigen::ColMajor | Eigen::AutoAlign > Base;

  FixedMatrixColMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  FixedMatrixColMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  FixedMatrixColMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

// --- Fixed vector
template< class T, int rows >
class FixedColVector :
    public FixedMatrixColMajor< T, rows, 1 >
{
public:
  typedef FixedMatrixColMajor< T, rows, 1 > Base;

  FixedColVector( void ) : Base()
  {}
  template< typename OtherDerived >
  FixedColVector( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  FixedColVector & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T, int cols >
class FixedRowVector :
    public FixedMatrixRowMajor< T, 1, cols >
{
public:
  typedef FixedMatrixRowMajor< T, 1, cols > Base;

  FixedRowVector( void ) : Base()
  {}
  template< typename OtherDerived >
  FixedRowVector( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  FixedRowVector & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

// --- 4x4 matrix
template< class T >
class Matrix44RowMajor :
    public FixedMatrixRowMajor< T, 4, 4 >
{
public:
  typedef FixedMatrixRowMajor< T, 4, 4 > Base;

  Matrix44RowMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  Matrix44RowMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  Matrix44RowMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T >
class Matrix44ColMajor :
    public FixedMatrixColMajor< T, 4, 4 >
{
public:
  typedef FixedMatrixColMajor< T, 4, 4 > Base;

  Matrix44ColMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  Matrix44ColMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  Matrix44ColMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T >
using Matrix44 = Matrix44RowMajor< T >;

// --- 3x3 matrix
template< class T >
class Matrix33RowMajor :
    public FixedMatrixRowMajor< T, 3, 3 >
{
public:
  typedef FixedMatrixRowMajor< T, 3, 3 > Base;

  Matrix33RowMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  Matrix33RowMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  Matrix33RowMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T >
class Matrix33ColMajor :
    public FixedMatrixColMajor< T, 3, 3 >
{
public:
  typedef FixedMatrixColMajor< T, 3, 3 > Base;

  Matrix33ColMajor( void ) : Base()
  {}
  template< typename OtherDerived >
  Matrix33ColMajor( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  Matrix33ColMajor & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T >
using Matrix33 = Matrix33RowMajor< T >;

// --- 6-dimensional vector
template< class T >
class VectorRow6 :
    public FixedRowVector< T, 6 >
{
public:
  typedef FixedRowVector< T, 6 > Base;

  VectorRow6( void ) : Base()
  {}
  template< typename OtherDerived >
  VectorRow6( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  VectorRow6 & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T >
class VectorCol6 :
    public FixedColVector< T, 6 >
{
public:
  typedef FixedColVector< T, 6 > Base;

  VectorCol6( void ) : Base()
  {}
  template< typename OtherDerived >
  VectorCol6( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  VectorCol6 & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

// --- 4-dimensional vector
template< class T >
class VectorRow4 :
    public FixedRowVector< T, 4 >
{
public:
  typedef FixedRowVector< T, 4 > Base;

  VectorRow4( void ) : Base()
  {}
  template< typename OtherDerived >
  VectorRow4( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  VectorRow4 & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T >
class VectorCol4 :
    public FixedColVector< T, 4 >
{
public:
  typedef FixedColVector< T, 4 > Base;

  VectorCol4( void ) : Base()
  {}
  template< typename OtherDerived >
  VectorCol4( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  VectorCol4 & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

// --- 3-dimensional vector
template< class T >
class VectorRow3 :
    public FixedRowVector< T, 3 >
{
public:
  typedef FixedRowVector< T, 3 > Base;

  VectorRow3( void ) : Base()
  {}
  template< typename OtherDerived >
  VectorRow3( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  VectorRow3 & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};

template< class T >
class VectorCol3 :
    public FixedColVector< T, 3 >
{
public:
  typedef FixedColVector< T, 3 > Base;

  VectorCol3( void ) : Base()
  {}
  template< typename OtherDerived >
  VectorCol3( const Eigen::MatrixBase< OtherDerived > & other )
      : Base( other )
  {}
  template< typename OtherDerived >
  VectorCol3 & operator= ( const Eigen::MatrixBase< OtherDerived > & other )
  {
    this->Base::operator=( other );
    return *this;
  }
};
} //end namespace Numeric
} //end namespace phovo
#endif
