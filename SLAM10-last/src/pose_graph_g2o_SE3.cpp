//
// Created by gtkansy on 18-5-24.
//

#include <iostream>
#include <fstream>
#include <string>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

using namespace std;
using namespace g2o;
//interrupted by signal 11: SIGSEGV  在g2o::VertexSE3* v=new g2o::VertexSE3();
/*
int main(int argc,char** argv){
    if(argc!=2){
        cout<<"Usage: pose_graph_g2o_SE3 sphere.g2o"<<endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if ( !fin )
    {
        cout<<"file "<<argv[1]<<" does not exist."<<endl;
        return 1;
    }
    typedef  g2o::BlockSolver<g2o::BlockSolverTraits<6,6>> Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCholmod<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    int vertexCnt=0,edgeCnt=0;
    //文件达到末尾 返回true
    while(!fin.eof()){
        cout<<"11"<<endl;
        string name;
        fin>>name;
        if(name=="VERTEX_SE3:QUAT"){
            g2o::VertexSE3* v=new g2o::VertexSE3();
            int index=0;
            fin>>index;
            v->setId(index);
            v->read(fin);   //to see
            optimizer.addVertex(v);
            vertexCnt++;
            if(index==0)
                v->setFixed(true);  //第一个点固定为0   //to see 为什么其他时候不加上这句
        }
        else if(name=="EDGE_SE3:QUAT"){
            g2o::EdgeSE3* e=new g2o::EdgeSE3();
            int idx1,idx2;
            fin>>idx1>>idx2;
            e->setId(edgeCnt++);
            e->setVertex(0,optimizer.vertices()[idx1]);   //optimizer.vertices()是一个map，可以根据索引号找到对应的vertex
            e->setVertex(1,optimizer.vertices()[idx2]);
            e->read(fin);

            optimizer.addEdge(e);
        }
        if(!fin.is_open()) break;
    }
    cout<<"read total "<<vertexCnt<<" vertices, "<<edgeCnt<<" edges "<<endl;
    cout<<"prepare optimizing .. "<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    cout<<"calling optimizing"<<endl;
    optimizer.optimize(30);
    cout<<"saving optimaization results .."<<endl;
    optimizer.save("../data/result.g2o");
    return 0;

}*/


//换一种方式去实现
int main(){
    typedef  g2o::BlockSolver<g2o::BlockSolverTraits<6,6>> Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCholmod<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    if(!optimizer.load("/home/gtkansy/CLionProjects/SLAM10-last/data/sphere.g2o"))
    {
        cout<<"error loading graph"<<endl;
        return 0;
    }
    else{
        cout<<"Loaded "<<optimizer.vertices().size()<<" vertices"<<endl;
        cout<<"Loaded "<<optimizer.edges().size()<<" edges"<<endl;
    }

    VertexSE3* firstRobotPose= dynamic_cast<VertexSE3*>(optimizer.vertex(0));
    firstRobotPose->setFixed(true);
    //优化过程中，第一个点固定，不做优化; 优化过程中，必须固定一个pose。
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    cout<<"optimazing"<<endl;
    optimizer.optimize(30);
    cout<<"done"<<endl;
    optimizer.save("../data/sphere_after.g2o");
    return 0;



}