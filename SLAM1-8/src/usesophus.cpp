//
// Created by gtkansy on 18-3-20.
//

//http://blog.csdn.net/robinhjwy/article/details/77334189  解析
#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/so3.h"
#include "sophus/se3.h"

using namespace std;
using namespace Eigen;
using namespace Sophus;
int main(){
    Eigen::Matrix3d R=Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d(0,0,1)).toRotationMatrix();

    Sophus::SO3 SO3_R(R);     //so3从旋转矩阵构造
    Sophus::SO3 SO3_v(0,0,M_PI/2);   //也可以从旋转向量来构造
    Eigen::Quaterniond q(R);
    Sophus::SO3 SO3_q(q);          //或者四元数
    cout<<"so3 from matrix "<<SO3_R<<endl;
    cout<<"so3 from vector"<<SO3_v<<endl;
    cout<<"so3 from quaternion"<<SO3_q<<endl;
    Eigen::Vector3d so3 = SO3_R.log();
    cout<<"so3 = "<<so3.transpose()<<endl;
    cout<<"so3 hat=\n"<<Sophus::SO3::hat(so3)<<endl;     //hat为向量到反对成矩阵
// 相对的，vee为反对称矩阵到向量，相当于下尖尖运算       important
    cout<<"so3 hat vee= "<<Sophus::SO3::vee( Sophus::SO3::hat(so3) ).transpose()<<endl;

    //下面是对SE3的操作
    Vector3d t(1,0,0);
    SE3 SE3_Rt(R,t);
    SE3 SE3_qt(q,t);
    cout<<SE3_Rt<<endl;
    cout<<SE3_qt<<endl;
    typedef Matrix<double,6,1> Vector6d;
    Vector6d se3=SE3_Rt.log();
    cout<<se3.transpose()<<endl;   //se3是平移在前,输出在后面,与书中不一样
    cout<<SE3::hat(se3)<<endl;
    cout<<SE3::vee(SE3::hat(se3))<<endl;




}