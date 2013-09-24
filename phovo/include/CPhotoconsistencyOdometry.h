/*
 *  Photoconsistency-Visual-Odometry
 *  Multiscale Photoconsistency Visual Odometry from RGBD Images
 *  Copyright (c) 2012, Miguel Algaba Borrego
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
#include <eigen3/Eigen/Dense>

namespace PhotoconsistencyOdometry
{

void eigenPose(float x,
           float y,
           float z,
           float yaw,
           float pitch,
           float roll,
           Eigen::Matrix4f & pose)
{
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

}

template <class T>
void warpImage(cv::Mat & imgGray,
               cv::Mat & imgDepth,
               cv::Mat & imgGrayWarped,
               Eigen::Matrix4f & Rt,
               Eigen::Matrix3f & cameraMatrix,int level=0)
{
    float fx = cameraMatrix(0,0)/pow(2,level);
    float fy = cameraMatrix(1,1)/pow(2,level);
    float inv_fx = 1.f/fx;
    float inv_fy = 1.f/fy;
    float ox = cameraMatrix(0,2)/pow(2,level);
    float oy = cameraMatrix(1,2)/pow(2,level);

    Eigen::Vector4f point3D;
    Eigen::Vector4f transformedPoint3D;
    int transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1

    imgGrayWarped = cv::Mat::zeros(imgGray.rows,imgGray.cols,imgGray.type());

    #if ENABLE_OPENMP_MULTITHREADING_WARP_IMAGE
    #pragma omp parallel for private(point3D,transformedPoint3D,transformed_r,transformed_c)
    #endif
    for(int r=0;r<imgGray.rows;r++)
    {
        for(int c=0;c<imgGray.cols;c++)
        {
            if(imgDepth.at<float>(r,c)>0) //If has valid depth value
            {
                //Compute the local 3D coordinates of pixel(r,c) of frame 1
                point3D(2) = imgDepth.at<float>(r,c); //z
                point3D(0) = (c-ox) * point3D(2) * inv_fx;	   //x
                point3D(1) = (r-oy) * point3D(2) * inv_fy;	   //y
                point3D(3) = 1.0;			   //homogeneous coordinate

                //Transform the 3D point using the transformation matrix Rt
                transformedPoint3D = Rt * point3D;

                //Project the 3D point to the 2D plane
                transformed_c = ((transformedPoint3D(0) * fx) / transformedPoint3D(2)) + ox; //transformed x (2D)
                transformed_r = ((transformedPoint3D(1) * fy) / transformedPoint3D(2)) + oy; //transformed y (2D)

                //Asign the intensity value to the warped image and compute the difference between the transformed
                //pixel of frame 1 and the corresponding pixel of frame 2. Compute the error function
                if(transformed_r>=0 && transformed_r < imgGray.rows &
                   transformed_c>=0 && transformed_c < imgGray.cols)
                {
                    imgGrayWarped.at<T>(transformed_r,transformed_c)=imgGray.at<T>(r,c);
                }
            }
        }
    }
}

/*!This abstract class defines the mandatory methods that any derived class must implement to compute the rigid (6DoF) transformation that best aligns a pair of RGBD frames using a photoconsistency maximization approach.*/
class CPhotoconsistencyOdometry
{
public:
  /*!Sets the 3x3 matrix of (pinhole) camera intrinsic parameters used to obtain the 3D colored point cloud from the RGB and depth images.*/
  virtual void setCameraMatrix(Eigen::Matrix3f & camMat)=0;
  /*!Sets the source (Intensity+Depth) frame.*/
  virtual void setSourceFrame(cv::Mat & imgGray,cv::Mat & imgDepth)=0;
  /*!Sets the source (Intensity+Depth) frame.*/
  virtual void setTargetFrame(cv::Mat & imgGray,cv::Mat & imgDepth)=0;
  /*!Initializes the state vector to a certain value. The optimization process uses the initial state vector as the initial estimate.*/
  virtual void setInitialStateVector(const std::vector<double> & initialStateVector)=0;
  /*!Launches the least-squares optimization process to find the configuration of the state vector parameters that maximizes the photoconsistency between the source and target frame.*/
  virtual void optimize()=0;
  /*!Returns the optimal state vector. This method has to be called after calling the optimize() method.*/
  virtual void getOptimalStateVector(std::vector<double> & optimalStateVector)=0;
  /*!Returns the optimal 4x4 rigid transformation matrix between the source and target frame. This method has to be called after calling the optimize() method.*/
  virtual void getOptimalRigidTransformationMatrix(Eigen::Matrix4f & optimal_Rt)=0;
};

} //end namespace PhotoconsistencyOdometry

#endif
