//
// Created by gtkansy on 18-5-16.
//

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>    //provide rotation and transition
using namespace std;
using namespace Eigen;


int main() {

    Matrix3d rotation_matrix = Matrix3d::Identity();
    cout << rotation_matrix << endl;
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));   //旋转向量
    cout.precision(3);       //cout的输出
    cout << rotation_vector.matrix() << endl;
    rotation_matrix = rotation_vector.toRotationMatrix();     //就是变成了旋转矩阵
    cout << rotation_matrix << endl;
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);      //旋转矩阵变成欧拉角来表示,同时用ZYX的顺序,yaw,pitch,roll
    cout << euler_angles.transpose() << endl;
    //下面是变换矩阵 ,Isometry用来表示欧式变换矩阵 ,注意上面的是角度旋转矩阵  v    4*4的矩阵
    Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Vector3d(1,3,4));      //平移向量
    cout<<T.matrix()<<endl;
    //四元数
    Quaterniond q=Quaterniond(rotation_vector);    //直接从向量到四元数
    cout<<q.coeffs()<<endl;        //x,y,z,w
    q=Quaterniond(rotation_matrix);
    cout<<q.coeffs()<<endl;

}