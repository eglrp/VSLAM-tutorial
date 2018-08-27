//
// Created by gtkansy on 18-5-24.
//


//携程一个头文件  common.h
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

using namespace std;
using namespace g2o;
using namespace Sophus;

typedef Eigen::Matrix<double,6,6> Matrix6d;


Matrix6d JRInv( SE3d e )
{
    Matrix6d J;
    J.block(0,0,3,3) = SO3d::hat(e.so3().log());
    J.block(0,3,3,3) = SO3d::hat(e.translation());
    J.block(3,0,3,3) = Eigen::Matrix3d::Zero(3,3);
    J.block(3,3,3,3) = SO3d::hat(e.so3().log());
    J = J*0.5 + Matrix6d::Identity();
    return J;
}

typedef Eigen::Matrix<double,6,1> Vector6d;
class VertexSE3LieAlgebra:public g2o::BaseVertex<6,SE3d>{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool read(istream& is){
        double data[7];
        for(int i=0;i<7;i++)
            is>>data[i];
        setEstimate(SE3d(
                Eigen::Quaterniond(data[6],data[3],data[4],data[5]),
                Eigen::Vector3d(data[0],data[1],data[2])
                ));
    }
    //第一次少打了const
    bool write(ostream& os)const{
        os<<id()<<" ";
        Eigen::Quaterniond q = _estimate.unit_quaternion();
        os<<_estimate.translation().transpose()<<" ";
        os<<q.coeffs()[0]<<" "<<q.coeffs()[1]<<" "<<q.coeffs()[2]<<" "<<q.coeffs()[3]<<endl;
        return true;
    }
    virtual void setToOriginImpl(){
        _estimate=Sophus::SE3d();
    }
    virtual void oplusImpl(const double* update){

        //Sophus::SO3d::hat(Eigen::Vector3d(update[3],update[4],update[5]) );
        Eigen::Vector3d  so3d(update[3],update[4],update[5]);
        //注意SE3已经是李群了，不是;李代数，注意分别
        Sophus::SO3d so3d_update=Sophus::SO3d::exp(so3d);


        Sophus::SE3d up(
                so3d_update,
                Eigen::Vector3d(update[0], update[1], update[2] )
                );
        _estimate=up*_estimate;
    }
};

class EdgeSE3LieAlgebra:public g2o::BaseBinaryEdge<6,SE3d,VertexSE3LieAlgebra,VertexSE3LieAlgebra>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool read(istream& is){
        double data[7];
        for ( int i=0; i<7; i++ )
            is>>data[i];
        Eigen::Quaterniond q ( data[6], data[3], data[4], data[5] );
        q.normalize();
        setMeasurement (
                Sophus::SE3d ( q, Eigen::Vector3d ( data[0], data[1], data[2] ) )
        );
        for ( int i=0; i<information().rows() && is.good(); i++ )
            for ( int j=i; j<information().cols() && is.good(); j++ )
            {
                is >> information() ( i,j );
                if ( i!=j )
                    information() ( j,i ) =information() ( i,j );
            }
        return true;
    }
    bool write ( ostream& os ) const
    {
        VertexSE3LieAlgebra* v1 = static_cast<VertexSE3LieAlgebra*> (_vertices[0]);
        VertexSE3LieAlgebra* v2 = static_cast<VertexSE3LieAlgebra*> (_vertices[1]);
        os<<v1->id()<<" "<<v2->id()<<" ";
        SE3d m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os<<m.translation().transpose()<<" ";
        os<<q.coeffs()[0]<<" "<<q.coeffs()[1]<<" "<<q.coeffs()[2]<<" "<<q.coeffs()[3]<<" ";
        // information matrix
        for ( int i=0; i<information().rows(); i++ )
            for ( int j=i; j<information().cols(); j++ )
            {
                os << information() ( i,j ) << " ";
            }
        os<<endl;
        return true;
    }
    // 误差计算与书中推导一致   参考书本p272第十一章
    virtual void computeError(){
        Sophus::SE3d v1= static_cast<VertexSE3LieAlgebra*>(_vertices[0])->estimate();
        Sophus::SE3d v2 = (static_cast<VertexSE3LieAlgebra*> (_vertices[1]))->estimate();
        _error = (_measurement.inverse()*v1.inverse()*v2).log();
    }
    virtual void linearizeOplus(){
        Sophus::SE3d v1 = (static_cast<VertexSE3LieAlgebra*> (_vertices[0]))->estimate();
        Sophus::SE3d v2 = (static_cast<VertexSE3LieAlgebra*> (_vertices[1]))->estimate();
        Matrix6d J = JRInv(SE3d::exp(_error));
        // 尝试把J近似为I？
        _jacobianOplusXi = - J* v2.inverse().Adj();
        _jacobianOplusXj = J*v2.inverse().Adj();
    }

};


int main(){
    typedef  g2o::BlockSolver<g2o::BlockSolverTraits<6,6>> Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCholmod<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);


    ifstream fin ("/home/gtkansy/CLionProjects/SLAM10-last/data/sphere.g2o");
    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量

    vector<VertexSE3LieAlgebra*> vectices;
    vector<EdgeSE3LieAlgebra*> edges;
    while ( !fin.eof() )
    {
        string name;
        fin>>name;
        if ( name == "VERTEX_SE3:QUAT" )
        {
            // 顶点
            VertexSE3LieAlgebra* v = new VertexSE3LieAlgebra();
            int index = 0;
            fin>>index;
            v->setId( index );
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vectices.push_back(v);
            if ( index==0 )
                v->setFixed(true);
        }
        else if ( name=="EDGE_SE3:QUAT" )
        {
            // SE3-SE3 边
            EdgeSE3LieAlgebra* e = new EdgeSE3LieAlgebra();
            int idx1, idx2;     // 关联的两个顶点
            fin>>idx1>>idx2;
            e->setId( edgeCnt++ );
            e->setVertex( 0, optimizer.vertices()[idx1] );
            e->setVertex( 1, optimizer.vertices()[idx2] );
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if ( !fin.good() ) break;
    }

    cout<<"read total "<<vertexCnt<<" vertices, "<<edgeCnt<<" edges."<<endl;

    cout<<"prepare optimizing ..."<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    cout<<"calling optimizing ..."<<endl;
    optimizer.optimize(30);

    cout<<"saving optimization results ..."<<endl;
    // 因为用了自定义顶点且没有向g2o注册，这里保存自己来实现
    // 伪装成 SE3 顶点和边，让 g2o_viewer 可以认出
    ofstream fout("/home/gtkansy/CLionProjects/SLAM10-last/data/result_lie.g2o");
    for ( VertexSE3LieAlgebra* v:vectices )
    {
        fout<<"VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for ( EdgeSE3LieAlgebra* e:edges )
    {
        fout<<"EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
    return 0;
}



//nice