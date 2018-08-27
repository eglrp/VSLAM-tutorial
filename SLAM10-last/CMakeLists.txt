cmake_minimum_required(VERSION 3.10)
project(SLAM10_last)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
find_Package(Cholmod REQUIRED)
find_package(Ceres REQUIRED )
find_package( Sophus REQUIRED )

#####  gtsam  不一样
find_package( GTSAMCMakeTools )
find_package( GTSAM REQUIRED )
include_directories( ${GTSAM_INCLUDE_DIR} )
#######

include_directories( ${Sophus_INCLUDE_DIRS} )
include_directories(${G2O_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CHOLMOD_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/common
                    ${PROJECT_SOURCE_DIR}/ceres
                    ${PROJECT_SOURCE_DIR}/common/flags
                    ${PROJECT_SOURCE_DIR}/common/tools
                    ${PROJECT_SOURCE_DIR}/include)
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear
        g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse
        g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension
        g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser
        g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

message("find g2o libraries in: " ${G2O_INCLUDE_DIR})
message("find g2o lib in " ${G2O_LIBS})
message("find cholmod in " ${CHOLMOD_INCLUDE_DIR})

add_library(BALProblem SHARED ${PROJECT_SOURCE_DIR}/common/BALProblem.cpp)
add_library(ParseCmd SHARED ${PROJECT_SOURCE_DIR}/common/flags/command_args.cpp)
add_executable( pose_graph_g2o_SE3 src/pose_graph_g2o_SE3.cpp )
add_executable(g2o_bundle src/g2o_bundle.cpp)
add_executable(ceres_customBundle src/ceresBundle.cpp)
add_executable(pose_graph_g2o_lie_algebra src/pose_graph_g2o_lie_algebra.cpp)
add_executable( pose_graph_gtsam src/pose_graph_gtsam.cpp )
add_executable(dense_mapping src/dense_mapping.cpp)
target_link_libraries(pose_graph_g2o_lie_algebra ${Sophus_LIBRARIES} g2o_core g2o_stuff
        ${CHOLMOD_LIBRARIES})
target_link_libraries(g2o_bundle ${G2O_LIBS} ${CHOLMOD_LIBRARIES} BALProblem ParseCmd)
target_link_libraries(ceres_customBundle ${CERES_LIBRARIES} BALProblem ParseCmd)
target_link_libraries(pose_graph_g2o_SE3 ${CHOLMOD_LIBRARIES} ${G2O_LIBS})
target_link_libraries( pose_graph_gtsam
  ${CHOLMOD_LIBRARIES} gtsam
)
target_link_libraries( dense_mapping   ${OpenCV_LIBS}
        ${Sophus_LIBRARIES})