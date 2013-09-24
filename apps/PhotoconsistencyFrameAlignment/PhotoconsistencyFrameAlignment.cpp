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

    //Set the camera parameters
    Eigen::Matrix3f cameraMatrix; cameraMatrix <<
                525., 0., 3.1950000000000000e+02,
                0., 525., 2.3950000000000000e+02,
                0., 0., 1.;

    //Load two RGB frames (RGB and depth images)
    cv::Mat imgRGB0 = cv::imread(argv[2]);
    cv::Mat imgGray0;
    cv::cvtColor( imgRGB0, imgGray0, CV_BGR2GRAY );
    cv::Mat imgDepth0 = cv::imread(argv[3],-1);
    imgDepth0.convertTo(imgDepth0, CV_32FC1, 1./1000 );

    cv::Mat imgRGB1 = cv::imread(argv[4]);
    cv::Mat imgGray1;
    cv::cvtColor( imgRGB1, imgGray1, CV_BGR2GRAY );
    cv::Mat imgDepth1 = cv::imread(argv[5],-1);
    imgDepth1.convertTo(imgDepth1, CV_32FC1, 1./1000 );

    //Define the photoconsistency odometry object and set the input parameters
	#if USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 0
	PhotoconsistencyOdometry::Analytic::CPhotoconsistencyOdometryAnalytic photoconsistencyOdometry;
	#elif USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 1
	PhotoconsistencyOdometry::Ceres::CPhotoconsistencyOdometryCeres photoconsistencyOdometry;
	#elif USE_PHOTOCONSISTENCY_ODOMETRY_METHOD == 2
    PhotoconsistencyOdometry::BiObjective::CPhotoconsistencyOdometryBiObjective photoconsistencyOdometry;
	#endif
	photoconsistencyOdometry.readConfigurationFile(std::string(argv[1]));
    photoconsistencyOdometry.setCameraMatrix(cameraMatrix);
    photoconsistencyOdometry.setSourceFrame(imgGray0,imgDepth0);
    photoconsistencyOdometry.setTargetFrame(imgGray1,imgDepth1);
    std::vector<double> stateVector; stateVector.resize(6,0); //x,y,z,yaw,pitch,roll
    photoconsistencyOdometry.setInitialStateVector(stateVector);

    //Optimize the problem to estimate the rigid transformation
    cv::TickMeter tm;tm.start();
    photoconsistencyOdometry.optimize();
    tm.stop();
    std::cout << "Time = " << tm.getTimeSec() << " sec." << std::endl;

    //Show results
    Eigen::Matrix4f Rt;
    photoconsistencyOdometry.getOptimalRigidTransformationMatrix(Rt);
    std::cout<<"main::Rt eigen:"<<std::endl<<Rt<<std::endl;
    cv::Mat warpedImage;
    PhotoconsistencyOdometry::warpImage<uint8_t>(imgGray0,imgDepth0,warpedImage,Rt,cameraMatrix);
    cv::Mat imgDiff;
    cv::absdiff(imgGray1,warpedImage,imgDiff);
    cv::imshow("main::imgDiff",imgDiff);

    cv::waitKey(0);

	return 0;
}

