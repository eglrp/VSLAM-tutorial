//
// Created by gtkansy on 18-5-22.
//

#ifndef SLAM10_LAST_G2O_BAL_CLASS_H
#define SLAM10_LAST_G2O_BAL_CLASS_H


#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <Eigen/Core>
#include "autodiff.h"
#include "rotation.h"
#include "projection.h"


//优化变量的维度，类型
class VertexCameraBAL:public g2o::BaseVertex<9,Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL(){}
    virtual bool read(std::istream& /*is*/){
        return false;
    }
    virtual bool write(std::ostream& /*os*/)const{
        return false;
    }
    virtual void setToOriginImpl(){}
    virtual void oplusImpl(const double* update){
        //Map类用于通过C++中普通的连续指针或者数组 （raw  C/C++ arrays）来构造Eigen里的Matrix类，
        // 这就好比Eigen里的Matrix类的数据和raw C++array 共享了一片地址，也就是引用。
        //总结来看就是将c++的数组转化为eigen 中矩阵的一个类，而且这个类可以直接操作原数组。
        Eigen::VectorXd::ConstMapType v(update,VertexCameraBAL::Dimension);
        _estimate+=v;
    }
};
//优化空间点
class VertexPointBAL:public g2o::BaseVertex<3,Eigen::Vector3d >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL(){}
    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double* update){
        Eigen::Vector3d::ConstMapType v(update);
        _estimate+=v;
    }
};

//观测值的维度，类型，连接顶点的类型
class EdgeObservationBAL :public g2o::BaseBinaryEdge<2,Eigen::Vector2d,VertexCameraBAL,VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL(){}
    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }
    virtual void computeError() override {
        const VertexCameraBAL* cam= static_cast<const VertexCameraBAL*>(vertex(0));
        const VertexPointBAL* point= static_cast<const VertexPointBAL*>(vertex(1));
        ( *this )(cam->estimate().data(),point->estimate().data(),_error.data());
        //estimate()  return the current estimate of the vertex
    }
    template<typename T>
    bool operator()(const T* camera,const T* point,T*residuals)const{
        T predictions[2];
        CamProjectionWithDistortion(camera,point,predictions);
        residuals[0]=predictions[0]-T(measurement()(0));   //返回当前观测值
        residuals[1]=predictions[1]-T(measurement()(1));
        //measurement()(0)  与vertex(0)不一样
        return true;
    }
    virtual void linearizeOplus() override{
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );
        //本来可以用g2o计算好的jacobian矩阵，直接赋值，这里用ceres的自动求导
        typedef ceres::internal::AutoDiff<EdgeObservationBAL,double,VertexCameraBAL::Dimension,VertexPointBAL::Dimension> BalAutoDiff;
        Eigen::Matrix<double,Dimension,VertexCameraBAL::Dimension,Eigen::RowMajor> dError_dCamera;
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;
        //Dimension是边的维度
        double *parameters[] = { const_cast<double*> ( cam->estimate().data() ), const_cast<double*> ( point->estimate().data() ) };
        double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
        double value[Dimension];
        bool diffState = BalAutoDiff::Differentiate ( *this, parameters, Dimension, value, jacobians );

        // copy over the Jacobians (convert row-major -> column-major)
        if ( diffState )
        {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        }
        else
        {
            assert ( 0 && "Error while differentiating" );
            _jacobianOplusXi.setZero();
            _jacobianOplusXi.setZero();
        }

    }

};





#endif //SLAM10_LAST_G2O_BAL_CLASS_H
