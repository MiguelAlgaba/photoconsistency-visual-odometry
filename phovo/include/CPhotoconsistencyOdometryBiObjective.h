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

#ifndef _CPHOTOCONSISTENCY_ODOMETRY_BIOBJECTIVE_
#define _CPHOTOCONSISTENCY_ODOMETRY_BIOBJECTIVE_

#define ENABLE_GAUSSIAN_BLUR 1
#define ENABLE_BOX_FILTER_BLUR 0
#define ENABLE_OPENMP_MULTITHREADING 0
#define ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS 0

#include "CPhotoconsistencyOdometry.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp" //TickMeter
#include <iostream>

namespace PhotoconsistencyOdometry
{

namespace BiObjective
{

/*!This class computes the rigid (6DoF) transformation that best aligns a pair of RGBD frames using a photoconsistency maximization approach.
To estimate the rigid transformation, this class implements a coarse to fine approach minimizing the photometric and depth error simultaneously. Thus, the algorithm starts finding a first pose approximation at
a low resolution level and uses the estimate to initialize the optimization at greater image scales. Both the residuals and jacobians are computed analytically.*/
class CPhotoconsistencyOdometryBiObjective : public CPhotoconsistencyOdometry
{

private:

    /*!Intensity (gray), depth and gradient image pyramids. Each pyramid has 'numOptimizationLevels' levels.*/
    std::vector<cv::Mat> gray0Pyr,gray1Pyr,depth0Pyr,depth1Pyr,gray1GradXPyr,gray1GradYPyr,depth1GradXPyr,depth1GradYPyr;
    /*!Camera matrix (intrinsic parameters).*/
    Eigen::Matrix3f cameraMatrix;
    /*!Current optimization level. Level 0 corresponds to the higher image resolution.*/
    int optimizationLevel;
    /*!Number of optimization levels.*/
    int numOptimizationLevels;
    /*!Scaling factor to update the state vector (at each level).*/
    std::vector<float>lambda_optimization_step;
    /*!Size (in pixels) of the blur filter (at each level).*/
    std::vector<int> blurFilterSize;
    /*!Scaling factor applied to the image gradients (at each level).*/
    std::vector<float> imageGradientsScalingFactor;
    /*!Maximum number of iterations for the Gauss-Newton algorithm (at each level).*/
    std::vector<int> max_num_iterations;
    /*!Minimum gradient norm of the jacobian (at each level).*/
    std::vector<float> min_gradient_norm;
    /*!Enable the visualization of the optimization process (only for debug).*/
    bool visualizeIterations;
    /*!State vector.*/
    Eigen::Matrix<double,6,1> stateVector; //Parameter vector (x y z yaw pitch roll)
    /*!Gradient of the error function.*/
    Eigen::Matrix<double,6,1> gradients;
    /*!Current iteration at the current optimization level.*/
    int iter;
    /*!Minimum allowed depth to consider a depth pixel valid.*/
    float minDepth;
    /*!Maximum allowed depth to consider a depth pixel valid.*/
    float maxDepth;
    /*!Depth component gain. This variable is used to scale the depth values so that depth components are similar to intensity values.*/
    float depthComponentGain;

    void buildPyramid(cv::Mat & img,std::vector<cv::Mat>& pyramid,int levels,bool applyBlur)
    {
        //Create space for all the images
        pyramid.resize(levels);

        double factor = 1;
        for(int level=0;level<levels;level++)
        {
            //Create an auxiliar image of factor times the size of the original image
            cv::Mat imgAux;
            if(level!=0)
            {
                cv::resize(img,imgAux,cv::Size(0,0),factor,factor);
            }
            else
            {
                imgAux = img;
            }

            //Blur the resized image with different filter size depending on the current pyramid level
            if(applyBlur)
            {
                #if ENABLE_GAUSSIAN_BLUR
                if(blurFilterSize[level]>0)
                {
                    cv::GaussianBlur(imgAux,imgAux,cv::Size(blurFilterSize[level],blurFilterSize[level]),3);
                    cv::GaussianBlur(imgAux,imgAux,cv::Size(blurFilterSize[level],blurFilterSize[level]),3);
                }
                #elif ENABLE_BOX_FILTER_BLUR
                if(blurFilterSize[level]>0)
                {
                    cv::blur(imgAux,imgAux,cv::Size(blurFilterSize[level],blurFilterSize[level]));
                    cv::blur(imgAux,imgAux,cv::Size(blurFilterSize[level],blurFilterSize[level]));
                }
                #endif
            }

            //Assign the resized image to the current level of the pyramid
            pyramid[level]=imgAux;

            factor = factor/2;
        }
    }

    void buildIntensityDerivativesPyramids(std::vector<cv::Mat>& imagePyramid,std::vector<cv::Mat>& derXPyramid,std::vector<cv::Mat>& derYPyramid)
    {
        //Compute image gradients
        int delta = 0;
        int ddepth = CV_32FC1;

        //Create space for all the derivatives images
        derXPyramid.resize(imagePyramid.size());
        derYPyramid.resize(imagePyramid.size());

        for(int level=0;level<imagePyramid.size();level++)
        {
            // Compute the gradient in x
            cv::Scharr( imagePyramid[level], derXPyramid[level], ddepth, 1, 0, imageGradientsScalingFactor[level], delta, cv::BORDER_DEFAULT );

            // Compute the gradient in y
            cv::Scharr( imagePyramid[level], derYPyramid[level], ddepth, 0, 1, imageGradientsScalingFactor[level], delta, cv::BORDER_DEFAULT );
        }
    }

    float maxDepthValue(cv::Mat & image)
    {
        float maxDepth = 0;

        for(int r=0;r<image.rows;r++)
        {
            for(int c=0;c<image.cols;c++)
            {
                if(image.at<float>(r,c)>maxDepth)
                {
                    maxDepth = image.at<float>(r,c);
                }
            }
        }

        return maxDepth;
    }

    void buildDepthDerivativesPyramids(std::vector<cv::Mat>& imagePyramid,std::vector<cv::Mat>& derXPyramid,std::vector<cv::Mat>& derYPyramid)
    {
        //Compute image gradients
        int delta = 0;
        int ddepth = CV_32FC1;

        //Create space for all the derivatives images
        derXPyramid.resize(imagePyramid.size());
        derYPyramid.resize(imagePyramid.size());

        for(int level=0;level<imagePyramid.size();level++)
        {
            cv::Mat imgNormalizedDepth;
            imagePyramid[level].convertTo(imgNormalizedDepth, CV_32FC1,1./maxDepth);

            // Compute the gradient in x
            cv::Scharr( imgNormalizedDepth, derXPyramid[level], ddepth, 1, 0, imageGradientsScalingFactor[level], delta, cv::BORDER_DEFAULT );

            // Compute the gradient in y
            cv::Scharr( imgNormalizedDepth, derYPyramid[level], ddepth, 0, 1, imageGradientsScalingFactor[level], delta, cv::BORDER_DEFAULT );
        }
    }

    //Separated jacobians
    void computeResidualsAndJacobians(cv::Mat & source_grayImg,
                                      cv::Mat & source_depthImg,
                                      cv::Mat & target_grayImg,
                                      cv::Mat & target_depthImg,
                                      cv::Mat & target_intensityGradXImg,
                                      cv::Mat & target_intensityGradYImg,
                                      cv::Mat & target_depthGradXImg,
                                      cv::Mat & target_depthGradYImg,
                                      Eigen::Matrix<double,Eigen::Dynamic,1> & residuals,
                                      Eigen::Matrix<double,Eigen::Dynamic,6> & jacobians,
                                      cv::Mat & warped_source_grayImage)
    {
        int nRows = source_grayImg.rows;
        int nCols = source_grayImg.cols;

        double scaleFactor = 1.0/pow(2,optimizationLevel);
        double fx = cameraMatrix(0,0)*scaleFactor;
        double fy = cameraMatrix(1,1)*scaleFactor;
        double ox = cameraMatrix(0,2)*scaleFactor;
        double oy = cameraMatrix(1,2)*scaleFactor;
        float inv_fx = 1.f/fx;
        float inv_fy = 1.f/fy;

        double x = stateVector[0];
        double y = stateVector[1];
        double z = stateVector[2];
        double yaw = stateVector[3];
        double pitch = stateVector[4];
        double roll = stateVector[5];

        //Compute the rigid transformation matrix from the parameters
        Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
        double sin_yaw = sin(yaw);
        double cos_yaw = cos(yaw);
        double sin_pitch = sin(pitch);
        double cos_pitch = cos(pitch);
        double sin_roll = sin(roll);
        double cos_roll = cos(roll);
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
        Rt(3,0) = 0;
        Rt(3,1) = 0;
        Rt(3,2) = 0;
        Rt(3,3) = 1;

        depthComponentGain = cv::mean(target_grayImg).val[0]/cv::mean(target_depthImg).val[0];

        #if ENABLE_OPENMP_MULTITHREADING
        #pragma omp parallel for
        #endif
        for (int r=0;r<nRows;r++)
        {
            for (int c=0;c<nCols;c++)
            {
                int i = nCols*r+c; //vector index

                //Compute the 3D coordinates of the pij of the source frame
                Eigen::Vector4f point3D;
                point3D(2)=source_depthImg.at<float>(r,c);
                if(minDepth < point3D(2) && point3D(2) < maxDepth)//Compute the jacobian only for the valid points
                {
                    point3D(0)=(c - ox) * point3D(2) * inv_fx;
                    point3D(1)=(r - oy) * point3D(2) * inv_fy;
                    point3D(3)=1;

                    double px = point3D(0);
                    double py = point3D(1);
                    double pz = point3D(2);

                    //Transform the 3D point using the transformation matrix Rt
                    Eigen::Vector4f  transformedPoint3D = Rt*point3D;

                    //Project the 3D point to the 2D plane
                    double inv_transformedPz = 1.0/transformedPoint3D(2);
                    double transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
                    transformed_c = (transformedPoint3D(0) * fx)*inv_transformedPz + ox; //transformed x (2D)
                    transformed_r = (transformedPoint3D(1) * fy)*inv_transformedPz + oy; //transformed y (2D)
                    int transformed_r_int = round(transformed_r);
                    int transformed_c_int = round(transformed_c);

                    //Asign the intensity value to the warped image and compute the difference between the transformed
                    //pixel of frame 1 and the corresponding pixel of frame 2. Compute the error function
                    if((transformed_r_int>=0 && transformed_r_int < nRows) &
                       (transformed_c_int>=0 && transformed_c_int < nCols))
                    {
                        //Obtain the pixel values that will be used to compute the intensity residual
                        double intensity1; //Intensity value of the pixel(r,c) of the warped frame 1
                        double intensity2; //Intensity value of the pixel(r,c) of frame 2
                        intensity1 = source_grayImg.at<float>(r,c);
                        intensity2 = target_grayImg.at<float>(transformed_r_int,transformed_c_int);

                        //Obtain the depth values that will be used to the compute the depth residual
                        double depth1; //Depth value of the pixel(r,c) of the warped frame 1
                        double depth2; //Depth value of the pixel(r,c) of frame 2
                        depth1 = source_depthImg.at<float>(r,c);
                        depth2 = target_depthImg.at<float>(transformed_r_int,transformed_c_int);

                        //Compute the rigid transformation jacobian
                        Eigen::Matrix<double,3,6> jacobianRt;

                            //Derivative with respect to x
                            jacobianRt(0,0)=1;
                            jacobianRt(1,0)=0;
                            jacobianRt(2,0)=0;

                            //Derivative with respect to y
                            jacobianRt(0,1)=0;
                            jacobianRt(1,1)=1;
                            jacobianRt(2,1)=0;

                            //Derivative with respect to z
                            jacobianRt(0,2)=0;
                            jacobianRt(1,2)=0;
                            jacobianRt(2,2)=1;

                            //Derivative with respect to yaw
                            jacobianRt(0,3)=py*(-sin(pitch)*sin(roll)*sin(yaw)-cos(roll)*cos(yaw))+pz*(sin(roll)*cos(yaw)-sin(pitch)*cos(roll)*sin(yaw))-cos(pitch)*px*sin(yaw);
                            jacobianRt(1,3)=pz*(sin(roll)*sin(yaw)+sin(pitch)*cos(roll)*cos(yaw))+py*(sin(pitch)*sin(roll)*cos(yaw)-cos(roll)*sin(yaw))+cos(pitch)*px*cos(yaw);
                            jacobianRt(2,3)=0;

                            //Derivative with respect to pitch
                            jacobianRt(0,4)=cos(pitch)*py*sin(roll)*cos(yaw)+cos(pitch)*pz*cos(roll)*cos(yaw)-sin(pitch)*px*cos(yaw);
                            jacobianRt(1,4)=cos(pitch)*py*sin(roll)*sin(yaw)+cos(pitch)*pz*cos(roll)*sin(yaw)-sin(pitch)*px*sin(yaw);
                            jacobianRt(2,4)=-sin(pitch)*py*sin(roll)-sin(pitch)*pz*cos(roll)-cos(pitch)*px;

                            //Derivative with respect to roll
                            jacobianRt(0,5)=py*(sin(roll)*sin(yaw)+sin(pitch)*cos(roll)*cos(yaw))+pz*(cos(roll)*sin(yaw)-sin(pitch)*sin(roll)*cos(yaw));
                            jacobianRt(1,5)=pz*(-sin(pitch)*sin(roll)*sin(yaw)-cos(roll)*cos(yaw))+py*(sin(pitch)*cos(roll)*sin(yaw)-sin(roll)*cos(yaw));
                            jacobianRt(2,5)=cos(pitch)*py*cos(roll)-cos(pitch)*pz*sin(roll);

                          //Compute the proyective transformation jacobian
                          Eigen::Matrix<double,2,3> jacobianProy;

                            //Derivative with respect to x
                            jacobianProy(0,0)=fx*inv_transformedPz;
                            jacobianProy(1,0)=0;

                            //Derivative with respect to y
                            jacobianProy(0,1)=0;
                            jacobianProy(1,1)=fy*inv_transformedPz;

                            //Derivative with respect to z
                            jacobianProy(0,2)=-(fx*transformedPoint3D(0))*inv_transformedPz*inv_transformedPz;
                            jacobianProy(1,2)=-(fy*transformedPoint3D(1))*inv_transformedPz*inv_transformedPz;

                          //Intensity jacobian:
                          //Apply the chain rule to compound the intensity gradients with the projective+RigidTransform jacobians
                          Eigen::Matrix<double,1,2> target_intensityGradient;
                          target_intensityGradient(0,0)=target_intensityGradXImg.at<float>(i);
                          target_intensityGradient(0,1)=target_intensityGradYImg.at<float>(i);
                          Eigen::Matrix<double,1,6> jacobianItensity=target_intensityGradient*jacobianProy*jacobianRt;

                          //Depth jacobian:
                          //Apply the chain rule to compound the depth gradients with the projective+RigidTransform jacobians
                          Eigen::Matrix<double,1,2> target_depthGradient;
                          target_depthGradient(0,0)=target_depthGradXImg.at<float>(i);
                          target_depthGradient(0,1)=target_depthGradYImg.at<float>(i);
                          Eigen::Matrix<double,1,6> jacobianRt_z;
                          jacobianRt_z(0,0)=jacobianRt(2,0);
                          jacobianRt_z(0,1)=jacobianRt(2,1);
                          jacobianRt_z(0,2)=jacobianRt(2,2);
                          jacobianRt_z(0,3)=jacobianRt(2,3);
                          jacobianRt_z(0,4)=jacobianRt(2,4);
                          jacobianRt_z(0,5)=jacobianRt(2,5);
                          Eigen::Matrix<double,1,6> jacobianDepth=depthComponentGain*(target_depthGradient*jacobianProy*jacobianRt-jacobianRt_z);

                          //Assign the pixel residual and jacobian to its corresponding row
                          #pragma omp critical
                          {
                              //Assign intensity jacobians
                              jacobians(i,0)=jacobianItensity(0,0);
                              jacobians(i,1)=jacobianItensity(0,1);
                              jacobians(i,2)=jacobianItensity(0,2);
                              jacobians(i,3)=jacobianItensity(0,3);
                              jacobians(i,4)=jacobianItensity(0,4);
                              jacobians(i,5)=jacobianItensity(0,5);

                              //Assign intensity residuals
                              residuals(nCols*transformed_r_int+transformed_c_int,0) = intensity2 - intensity1;

                              //Assign depth jacobians
                              jacobians(2*i,0)=jacobianDepth(0,0);
                              jacobians(2*i,1)=jacobianDepth(0,1);
                              jacobians(2*i,2)=jacobianDepth(0,2);
                              jacobians(2*i,3)=jacobianDepth(0,3);
                              jacobians(2*i,4)=jacobianDepth(0,4);
                              jacobians(2*i,5)=jacobianDepth(0,5);

                              //Assign depth residuals
                              residuals(nCols*2*transformed_r_int+2*transformed_c_int,0) = depthComponentGain*(depth2 - depth1);

                              if(visualizeIterations)
                                warped_source_grayImage.at<float>(transformed_r_int,transformed_c_int) = intensity1;
                          }
                    }
                }
            }
        }
    }

    enum TerminationCriteriaType {NonTerminated = -1,MaxIterationsReached = 0,GradientNormLowerThanThreshold = 1};
    bool testTerminationCriteria()
    {
        bool optimizationFinished = false;

        double gradientNorm = gradients.norm();

        TerminationCriteriaType terminationCriteria = NonTerminated;
        if(iter>=max_num_iterations[optimizationLevel])
        {
            terminationCriteria = MaxIterationsReached;
            optimizationFinished = true;
        }
        else if(gradientNorm<min_gradient_norm[optimizationLevel])
        {
            terminationCriteria = GradientNormLowerThanThreshold;
            optimizationFinished = true;
        }

        if(optimizationFinished)
        {
            #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            std::cout<<"----------------------------------------"<<std::endl;
            std::cout<<"Optimization level: "<<optimizationLevel<<std::endl;
            std::cout<<"Termination criteria: ";
            #endif

            switch(terminationCriteria)
            {
                case MaxIterationsReached:
                    #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
                    std::cout<<" Max number of iterations reached ("<<max_num_iterations[optimizationLevel]<<")"<<std::endl;;
                    #endif
                    break;
                case GradientNormLowerThanThreshold:
                    #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
                    std::cout<<" Gradient norm is lower than threshold ("<<gradient_tolerance[optimizationLevel]<<")"<<std::endl;
                    #endif
                    break;
                default :
                    break;
            }

            #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
            std::cout<<"Number iterations: "<<iter<<std::endl;
            std::cout<<"gradient norm: "<<gradientNorm<<std::endl;
            std::cout<<"----------------------------------------"<<std::endl;
            #endif
        }

        return optimizationFinished;
    }

public:

    CPhotoconsistencyOdometryBiObjective(){minDepth=0.3;maxDepth=5.0;};

    ~CPhotoconsistencyOdometryBiObjective(){};

    /*!Sets the minimum depth distance (m) to consider a certain pixel valid.*/
    void setMinDepth(float minD)
    {
        minDepth = minD;
    }

    /*!Sets the maximum depth distance (m) to consider a certain pixel valid.*/
    void setMaxDepth(float maxD)
    {
        maxDepth = maxD;
    }

    /*!Sets the 3x3 matrix of (pinhole) camera intrinsic parameters used to obtain the 3D colored point cloud from the RGB and depth images.*/
    void setCameraMatrix(Eigen::Matrix3f & camMat)
    {
        cameraMatrix = camMat;
    }

    /*!Sets the source (Intensity+Depth) frame.*/
    void setSourceFrame(cv::Mat & imgGray,cv::Mat & imgDepth)
    {
        //Create a float auxialiary image from the imput image
        cv::Mat imgGrayFloat;
        imgGray.convertTo(imgGrayFloat, CV_32FC1, 1./255 );

        //Compute image pyramids for the grayscale and depth images
        buildPyramid(imgGrayFloat,gray0Pyr,numOptimizationLevels,true);
        buildPyramid(imgDepth,depth0Pyr,numOptimizationLevels,false); //TODO: Do not apply low-pass filtering to depth image
    }

    /*!Sets the source (Intensity+Depth) frame.*/
    void setTargetFrame(cv::Mat & imgGray,cv::Mat & imgDepth)
    {
        //Create a float auxialiary image from the imput image
        cv::Mat imgGrayFloat;
        imgGray.convertTo(imgGrayFloat, CV_32FC1, 1./255 );

        //Compute image pyramids for the grayscale and depth images
        buildPyramid(imgGrayFloat,gray1Pyr,numOptimizationLevels,true);
        buildPyramid(imgDepth,depth1Pyr,numOptimizationLevels,false); //TODO: Do not apply low-pass filtering to depth image

        //Compute image pyramids for the gradients images
        buildIntensityDerivativesPyramids(gray1Pyr,gray1GradXPyr,gray1GradYPyr);
        buildDepthDerivativesPyramids(depth1Pyr,depth1GradXPyr,depth1GradYPyr);
    }

    /*!Initializes the state vector to a certain value. The optimization process uses the initial state vector as the initial estimate.*/
    void setInitialStateVector(const std::vector<double> & initialStateVector)
    {
        stateVector[0] = initialStateVector[0];
        stateVector[1] = initialStateVector[1];
        stateVector[2] = initialStateVector[2];
        stateVector[3] = initialStateVector[3];
        stateVector[4] = initialStateVector[4];
        stateVector[5] = initialStateVector[5];
    }

    /*!Launches the least-squares optimization process to find the configuration of the state vector parameters that maximizes the photoconsistency between the source and target frame.*/
    void optimize()
    {
        for(optimizationLevel = numOptimizationLevels-1;optimizationLevel>=0;optimizationLevel--)
        {
            int nRows = gray0Pyr[optimizationLevel].rows;
            int nCols = gray0Pyr[optimizationLevel].cols;
            int nPoints = nRows * nCols;

            iter = 0;
            while(true)
            {
                #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
                cv::TickMeter tm;tm.start();
                #endif
                cv::Mat warped_source_grayImage;
                if(visualizeIterations)
                    warped_source_grayImage = cv::Mat::zeros(nRows,nCols,gray0Pyr[optimizationLevel].type());

                Eigen::Matrix<double,Eigen::Dynamic,1> residuals;
                residuals = Eigen::MatrixXd::Zero(2*nPoints,1);
                Eigen::Matrix<double,Eigen::Dynamic,6> jacobians;
                jacobians = Eigen::MatrixXd::Zero(2*nPoints,6);

                if(max_num_iterations[optimizationLevel]>0) //compute only if the number of maximum iterations are greater than 0
                {
                    computeResidualsAndJacobians(
                            gray0Pyr[optimizationLevel],
                            depth0Pyr[optimizationLevel],
                            gray1Pyr[optimizationLevel],
                            depth1Pyr[optimizationLevel],
                            gray1GradXPyr[optimizationLevel],
                            gray1GradYPyr[optimizationLevel],
                            depth1GradXPyr[optimizationLevel],
                            depth1GradYPyr[optimizationLevel],
                            residuals,
                            jacobians,
                            warped_source_grayImage);

                    gradients = jacobians.transpose()*residuals;
                    stateVector = stateVector - lambda_optimization_step[optimizationLevel]*((jacobians.transpose()*jacobians).inverse() * gradients);

                    #if ENABLE_PRINT_CONSOLE_OPTIMIZATION_PROGRESS
                    tm.stop(); std::cout << "Iteration time = " << tm.getTimeSec() << " sec." << std::endl;
                    #endif
                }

                iter++;

                if(testTerminationCriteria()){break;}

                if(visualizeIterations)
                {
                    cv::Mat imgDiff = cv::Mat::zeros(nRows,nCols,gray1Pyr[optimizationLevel].type());
                    cv::absdiff(gray1Pyr[optimizationLevel],warped_source_grayImage,imgDiff);
                    cv::imshow("optimize::imgDiff",imgDiff);
                    cv::waitKey(0);
                }
            }
        }

        //After all the optimization process the optimization level is 0
        optimizationLevel = 0;

    }

    /*!Returns the optimal state vector. This method has to be called after calling the optimize() method.*/
    void getOptimalStateVector(std::vector<double> & optimalStateVector)
    {
        optimalStateVector[0] = stateVector[0];
        optimalStateVector[1] = stateVector[1];
        optimalStateVector[2] = stateVector[2];
        optimalStateVector[3] = stateVector[3];
        optimalStateVector[4] = stateVector[4];
        optimalStateVector[5] = stateVector[5];
    }

    /*!Returns the optimal 4x4 rigid transformation matrix between the source and target frame. This method has to be called after calling the optimize() method.*/
    void getOptimalRigidTransformationMatrix(Eigen::Matrix4f & optimal_Rt)
    {
        eigenPose(stateVector[0],stateVector[1],stateVector[2],
                  stateVector[3],stateVector[4],stateVector[5],optimal_Rt);
    }

    /*!Reads the configuration parameters from a .yml file.*/
    void readConfigurationFile(std::string fileName)
    {
        cv::FileStorage fs(fileName, cv::FileStorage::READ);

        //Read the number of optimization levels
        fs["numOptimizationLevels"] >> numOptimizationLevels;

        #if ENABLE_GAUSSIAN_BLUR || ENABLE_BOX_FILTER_BLUR
        //Read the blur filter size at every pyramid level
        fs["blurFilterSize (at each level)"] >> blurFilterSize;
        #endif

        //Read the scaling factor for each gradient image at each level
        fs["imageGradientsScalingFactor (at each level)"] >> imageGradientsScalingFactor;

        //Read the lambda factor to change the optimization step
        fs["lambda_optimization_step (at each level)"] >> lambda_optimization_step;

        //Read the number of Levenberg-Marquardt iterations at each optimization level
        fs["max_num_iterations (at each level)"] >> max_num_iterations;

        //Read optimizer minimum gradient norm at each level
        fs["min_gradient_norm (at each level)"] >> min_gradient_norm;

        //Read the boolean value to determine if visualize the progress images or not
        fs["visualizeIterations"] >> visualizeIterations;
    }
};

} //end namespace BiObjective

} //end namespace PhotoconsistencyOdometry

#endif
