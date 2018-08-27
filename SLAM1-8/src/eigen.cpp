#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
//稠密矩阵的代数运算(逆,特征值等)
using namespace std;
using namespace Eigen;
#define MATRIX_SIZE 50

int main(){

    Matrix<float,2,3> matrix_23;        //define
    Vector3d v_3d;
    Matrix3d matrix_33=Matrix3d::Zero();     //init 0
    Matrix<double,Dynamic,Dynamic> matrix_dynamic;     //dynamic matrix
    MatrixXd matrix_x;
    matrix_23<<1,2,3,4,5,6;
    cout<<matrix_23<<endl;
    cout<<matrix_23(1,2)<<endl;
    v_3d<<3,2,1;

    //matrix_23*v_3d   //  error, float and double
    Matrix<double,2,1> result=matrix_23.cast<double>()*v_3d;         //need to cast
    cout<<result<<endl;
    matrix_33=Matrix3d::Random();      //random函数,要么确定matrix的大小,要么就在rqandom里面确定
    cout<<matrix_33<<endl;
    cout<<matrix_33.transpose()<<endl;   //转置
    cout<<matrix_33.inverse()<<endl;    //逆
    cout<<matrix_33.sum()<<endl;    //所有元素之和
    cout<<matrix_33.trace()<<endl;   //迹
    cout<<matrix_33.determinant()<<endl;   //行列式

    //特征值
    //实对称矩阵可以保证对角化成功
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
    cout<<eigen_solver.eigenvalues()<<endl;
    cout<<eigen_solver.eigenvectors()<<endl;

    //解方程   直接求逆最简单,但是运算量大
    Matrix<double,MATRIX_SIZE,MATRIX_SIZE> matrix_NN;
    matrix_NN=MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    Matrix<double,MATRIX_SIZE,1> v_Nd;
    v_Nd=MatrixXd::Random(MATRIX_SIZE,1);    //random
    Matrix<double,MATRIX_SIZE,1> x=matrix_NN.inverse()*v_Nd;
    cout<<x<<endl;
    x=matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout<<x<<endl;
    return 0;
}