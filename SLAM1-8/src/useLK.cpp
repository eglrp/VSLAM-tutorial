//
// Created by gtkansy on 18-5-14.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
using namespace std;
using namespace cv;

int main(){
    string associate_file="/home/gtkansy/CLionProjects/SLAM/data/associate.txt";
    ifstream fin(associate_file);
    string rgb_file,depth_file,time_rgb,time_depth;
    list<Point2f> keypoints;     //list可以快速删除与插入
    cv::Mat color,depth,last_color;
    for(int index=0;index<100;index++){
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color=cv::imread("/home/gtkansy/CLionProjects/SLAM/data/"+rgb_file);
        depth=cv::imread("/home/gtkansy/CLionProjects/SLAM/data/"+depth_file,-1);
        if(index==0){
            vector<KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector=cv::FastFeatureDetector::create();
            detector->detect(color,kps);
            for(auto kp:kps){
                keypoints.push_back(kp.pt);
            }
            last_color=color;
            continue;
        }
        if(color.data== nullptr||depth.data== nullptr)
            continue;
        //对其他的帧采用lk跟踪特征点
        vector<cv::Point2f> next_keypoints;
        vector<cv::Point2f> prev_keypoints;
        for(auto kp:keypoints){
            prev_keypoints.push_back(kp);
        }
        vector<unsigned char> status;
        vector<float> error;
        cv::calcOpticalFlowPyrLK(last_color,color,prev_keypoints,next_keypoints,status,error);


        //把丢的点删掉,并重新赋值特征点
        int i=0;
        for(auto iter=keypoints.begin();iter!=keypoints.end();i++){
            if(status[i]==0){
                iter=keypoints.erase(iter);
                continue;
            }
            *iter=next_keypoints[i];
            iter++;
        }
        cout<<"tracking keypoints: "<<keypoints.size()<<endl;
        if(keypoints.size()==0){
            cout<<"all keypoints are lost"<<endl;
        }
        cv::Mat img_show=color.clone();
        for(auto kp:keypoints)
            cv::circle(img_show,kp,10,cv::Scalar(0,240,0),1);
        cv::imshow("corners",img_show);
        cv::waitKey(0);    //手动按空格,否则改为1
        last_color=color;
    }

}
