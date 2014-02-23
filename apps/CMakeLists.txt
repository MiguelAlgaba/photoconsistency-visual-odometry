option( BUILD_PHOTOCONSISTENCY_FRAME_ALIGNMENT_APPLICATION "Build PhotoconsistencyFrameAlignment application" ON )
if( ${BUILD_PHOTOCONSISTENCY_FRAME_ALIGNMENT_APPLICATION} MATCHES "ON" )
  add_subdirectory( PhotoconsistencyFrameAlignment )
endif()

option( BUILD_PHOTOCONSISTENCY_VISUAL_ODOMETRY_APPLICATION "Build PhotoconsistencyVisualOdometry application" ON )
if( ${BUILD_PHOTOCONSISTENCY_VISUAL_ODOMETRY_APPLICATION} MATCHES "ON" )
  add_subdirectory( PhotoconsistencyVisualOdometry )
endif()
