//
// Created by gtkansy on 18-5-11.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
using namespace std;
using namespace cv;

Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
void findmatches(const Mat& img1,const Mat& img2,std::vector<KeyPoint>& kep1,std::vector<KeyPoint>& kep2,
                Mat& descp1,Mat& descp2,std::vector<DMatch>& goodmatches){
    Ptr<ORB> orb=ORB::create();
    orb->detect(img1,kep1);
    orb->detect(img2,kep2);
    orb->compute(img1,kep1,descp1);
    orb->compute(img2,kep2,descp2);
    cv::BFMatcher matcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    matcher.match(descp1,descp2,matches,noArray());
    double min_dist=10000, max_dist=0;
    for ( int i = 0; i < descp1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descp1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            goodmatches.push_back ( matches[i] );
        }
    }
}
Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
            (
                    ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
                    ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
            );
}
void pose_estimation_3d3d(const std::vector<Point3f>& pts1,const std::vector<Point3f>& pts2,Mat& R,Mat& t){
    Point3f p1,p2,p3,p4;
    int N=pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }

    p1 = Point3f( Vec3f(p1) /  N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

    //svd on w
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U=svd.matrixU();
    Eigen::Matrix3d V=svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;
    Eigen::Matrix3d R_=U*(V.transpose());  //
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );
    //convert to cv::mat
    R = ( Mat_<double> ( 3,3 ) <<
                               R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
            R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
            R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
    );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
}

//提供3D到3D的边
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}     //定义

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map( _point );
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
protected:
    Eigen::Vector3d _point;
};

void bundleAdjustment (
        const vector< Point3f >& pts1,
        const vector< Point3f >& pts2,
        Mat& R, Mat& t ){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> > Block;  // pose 维度为 6, landmark 维度为 3
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverEigen<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    //Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    //Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId(0);
    pose->setEstimate( g2o::SE3Quat(
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d( 0,0,0 )
    ) );
    optimizer.addVertex( pose );
    //edge
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z) );
        edge->setId( index );
        edge->setVertex( 0, dynamic_cast<g2o::VertexSE3Expmap*> (pose) );    //dynamic_cast
        edge->setMeasurement( Eigen::Vector3d(
                pts1[i].x, pts1[i].y, pts1[i].z) );
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );       //1e4
        optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);
    }

    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;

}

int main(){
    Mat img_1 = imread("/home/gtkansy/Pictures/hh/vo/1.png", CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread("/home/gtkansy/Pictures/hh/vo/2.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat d1=imread("/home/gtkansy/Pictures/hh/vo/1_depth.png",CV_LOAD_IMAGE_UNCHANGED);
    Mat d2=imread("/home/gtkansy/Pictures/hh/vo/2_depth.png",CV_LOAD_IMAGE_UNCHANGED);
    std::vector<KeyPoint> kep1,kep2;
    Mat descp1,descp2;
    std::vector<DMatch> matches;
    findmatches(img_1,img_2,kep1,kep2,descp1,descp2,matches);
    cout<<"二维匹配的对数= "<<matches.size()<<endl;
    std::vector<Point3f> pts1_3d,pts2_3d;
    for(DMatch m:matches){
        auto dd1=d1.at<unsigned short>(kep1[m.queryIdx].pt.y,kep1[m.queryIdx].pt.x);
        auto dd2=d2.at<unsigned short>(kep2[m.trainIdx].pt.y,kep2[m.trainIdx].pt.x);
        if(dd1==0||dd2==0)
            continue;
        float ddd1=float(dd1)/5000.0;
        float ddd2=float(dd2)/5000.0;

        Point2d p1=pixel2cam(kep1[m.queryIdx].pt,K);
        Point2d p2=pixel2cam(kep2[m.trainIdx].pt,K);
        pts1_3d.push_back(Point3f(p1.x*ddd1,p1.y*ddd1,ddd1));
        pts2_3d.push_back(Point3f(p2.x*ddd2,p2.y*ddd2,ddd2));
    }
    cout<<"3d-3d pairs: "<<pts1_3d.size()<<endl;
    Mat R,t;
    pose_estimation_3d3d(pts1_3d,pts2_3d,R,t);
    cout<<R<<endl;
    cout<<t<<endl;
    bundleAdjustment (pts1_3d,pts2_3d,R,t);
}