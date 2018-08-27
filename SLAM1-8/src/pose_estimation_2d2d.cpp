//
// Created by gtkansy on 18-5-9.
//

//用对极几何求解2D->2D
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_matches(Mat img1,Mat img2,std::vector<KeyPoint>& keyp1,std::vector<KeyPoint>& keyp2,
                  Mat& desc1,Mat& desc2,std::vector<DMatch>& good_matches){
    //Ptr<ORB> orb=ORB::create(500,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
    Ptr<ORB> orb=ORB::create();
    orb->detect(img1,keyp1);
    orb->compute(img1,keyp1,desc1);
    orb->detect(img2,keyp2);
    orb->compute(img2,keyp2,desc2);
    std::vector<DMatch> matches;
    //BFMatcher matcher(NORM_HAMMING);
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    matcher->match(desc1,desc2,matches);
    double min_dist=10000,max_dist=0;
    for(int i=0;i<desc1.rows;i++){
        double dist=matches[i].distance;
        if(dist<min_dist) min_dist=dist;
        if(dist>max_dist) max_dist=dist;
    }
    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );
    for(int i=0;i<desc1.rows;i++){
        if(matches[i].distance<=max(2*min_dist,30.0)){
            good_matches.push_back(matches[i]);
        }
    }
}

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> matches,
        Mat& R, Mat& t){
    Mat K=(Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> points1,points2;
    for(int i=0;i<(int)matches.size();i++){
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);

    }
    //计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix=cv::findFundamentalMat(points1,points2,FM_8POINT);
    cout<<"fundamental_matrix is "<<"\n"<<fundamental_matrix<<endl;
    //计算本质矩阵
    Point2d principal_point ( 325.1, 249.7 );				//相机主点, TUM dataset标定值
    int focal_length = 521;						//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<"\n"<<essential_matrix<<endl;
/*    Mat essential_matrix;
    essential_matrix=cv::findEssentialMat(points1,points2,K,RANSAC);
    cout<<"essential_matrix is "<<"\n"<<essential_matrix<<endl;*/

    //计算单应矩阵
    Mat homography_matrix;
    homography_matrix=findHomography(points1,points2,RANSAC,3,noArray(),2000,0.99);
    cout<<"homography_Mtrix is "<<"\n"<<homography_matrix<<endl;
    //计算出R和t的信息
    //Point2d principal_point(325.1,249.7);
    //int focal_length=521;
    cv::recoverPose(essential_matrix,points1,points2,R,t,focal_length,principal_point);
   // cv::recoverPose(essential_matrix,points1,points2,K,R,t,noArray());
    cout<<"R is "<<"\n"<<R<<endl;
    cout<<"t is "<<"\n"<<t<<endl;
};

//c++编译器加入了临时变量不能作为非const引用的这个语义限制，意在限制这个非常规用法的潜在错误。
Point2d pixel2cam (const Point2d& p, const Mat& K )
{
    return Point2d
            (
                    ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
                    ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
            );
}
//注意求解的是相机归一化的坐标,E描述的是不同视角下相机坐标系归一化的变换关系


int main(int argc,char* argv[]) {
    if (argc != 1) {
        cout << "img1 img2" << endl;
        return 1;
    }
    Mat img1 = imread("/home/gtkansy/Pictures/hh/1.png", CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread("/home/gtkansy/Pictures/hh/1_tran.png", CV_LOAD_IMAGE_COLOR);
    std::vector<KeyPoint> keyp1, keyp2;
    Mat desc1, desc2;
    std::vector<DMatch> matches;
    find_matches(img1, img2, keyp1, keyp2, desc1, desc2, matches);
    cout << "一共找到了多少个点 " << matches.size() << endl;

    //solve R and t
    Mat R,t;
    pose_estimation_2d2d(keyp1,keyp2,matches,R,t);
    //验证E=t^R的内在性质
    Mat t_x = ( Mat_<double> ( 3,3 ) <<
            0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
            t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
            -t.at<double> ( 1.0 ),     t.at<double> ( 0,0 ),      0 );
    cout<<"t^R= "<<"\n"<<t_x*R<<endl;
    //验证对极约束
    Mat K=(Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for(DMatch m:matches){
        Point2d pt1=pixel2cam(keyp1[m.queryIdx].pt,K);
        Mat y1=(Mat_<double>(3,1)<<pt1.x,pt1.y,1);
        Point2d pt2=pixel2cam(keyp2[m.trainIdx].pt,K);
        Mat y2=(Mat_<double>(3,1)<<pt2.x,pt2.y,1);
        Mat d=y2.t()*t_x*R*y1;
        cout<<"epipolar constraint= "<<d<<endl;

    }

}