//
// Created by gtkansy on 18-5-9.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <assert.h>
using namespace std;
using namespace cv;


int main(){
    Mat img1=imread("/home/gtkansy/Pictures/hh/1.png",CV_LOAD_IMAGE_COLOR);
    Mat img2=imread("/home/gtkansy/Pictures/hh/10.png",CV_LOAD_IMAGE_COLOR);
    if(img1.empty()||img2.empty()){
        return 0;
    }
    std::vector<KeyPoint> keyp1,keyp2;
    Mat desc1,desc2;
    Ptr<ORB> orb=ORB::create(500,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
    orb->detect(img1,keyp1);
    orb->compute(img1,keyp1,desc1);
    orb->detect(img2,keyp2);
    orb->compute(img2,keyp2,desc2);
    Mat img3;
    cv::drawKeypoints(img1,keyp1,img3,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("raw",img3);

    std::vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(desc1,desc2,matches);
    double min_dist=10000,max_dist=0;
    for(int i=0;i<desc1.rows;i++){
        double dist=matches[i].distance;
        if(dist<min_dist) min_dist=dist;
        if(dist>max_dist) max_dist=dist;
    }
    cout<<"max_dist= "<<max_dist<<endl;
    std::vector<DMatch> good_matches;
    for(int i=0;i<desc1.rows;i++){
        if(matches[i].distance<=max(2*min_dist,30.0)){
            good_matches.push_back(matches[i]);
        }
    }
    Mat good_match;
    drawMatches(img1,keyp1,img2,keyp2,good_matches,good_match);
    imshow("match",good_match);
    cv::waitKey(0);
}