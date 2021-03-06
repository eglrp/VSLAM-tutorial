//
// Created by gtkansy on 18-5-22.
//

#include <iostream>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>
#include <memory>
#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "BundleParams.h"
#include "BALProblem.h"
#include "g2o_bal_class.h"


using namespace std;
using namespace Eigen;

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3> > BalBlockSolver;
//为什么有的时候是landmark的维度，有时候有事误差值的维度

void BuildProblem(const BALProblem* bal_problem,g2o::SparseOptimizer* optimizer,
                    const BundleParams& params)
{
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    const double* raw_cameras=bal_problem->cameras();
    for(int i=0;i<num_cameras;++i){
        ConstVectorRef temVecCamera(raw_cameras + camera_block_size * i,camera_block_size);
        VertexCameraBAL* pCamera = new VertexCameraBAL();
        pCamera->setEstimate(temVecCamera);   // initial value for the camera i..
        pCamera->setId(i);                    // set id for each camera vertex
        // remeber to add vertex into optimizer..
        optimizer->addVertex(pCamera);

    }

    const double* raw_points=bal_problem->points();
    for(int j=0;j<num_points;++j){
        ConstVectorRef temVecPoint(raw_points+point_block_size*j,point_block_size);
        VertexPointBAL* pPoint=new VertexPointBAL();
        pPoint->setEstimate(temVecPoint);
        pPoint->setId(j+num_cameras);
        pPoint->setMarginalized(true);
        optimizer->addVertex(pPoint);
    }
    const int num_observations=bal_problem->num_observations();
    const double* observations=bal_problem->observations();
    for(int i=0;i<num_observations;++i){
        EdgeObservationBAL* bal_edge=new EdgeObservationBAL();
        const int camera_id=bal_problem->camera_index()[i];
        const int point_id = bal_problem->point_index()[i] + num_cameras;
        if(params.robustify)
        {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0);
            bal_edge->setRobustKernel(rk);
        }
        // set the vertex by the ids for an edge observation
        bal_edge->setVertex(0,dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id)));
        bal_edge->setVertex(1,dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id)));
        bal_edge->setInformation(Eigen::Matrix2d::Identity());
        bal_edge->setMeasurement(Eigen::Vector2d(observations[2*i+0],observations[2*i + 1]));

        optimizer->addEdge(bal_edge) ;
    }

}

void WriteToBALProblem(BALProblem* bal_problem,g2o::SparseOptimizer* optimizer){
    const int num_points=bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    double* raw_cameras = bal_problem->mutable_cameras();
    for(int i=0;i<num_cameras;i++){
        VertexCameraBAL* pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        Eigen::VectorXd NewCameraVec = pCamera->estimate();
        memcpy(raw_cameras + i * camera_block_size, NewCameraVec.data(), sizeof(double) * camera_block_size);
    }
    double* raw_points=bal_problem->mutable_points();
    for(int j=0;j<num_points;j++){
        VertexPointBAL* pPoint = dynamic_cast<VertexPointBAL*>(optimizer->vertex(j + num_cameras));
        Eigen::Vector3d NewPointVec = pPoint->estimate();
        memcpy(raw_points + j * point_block_size, NewPointVec.data(), sizeof(double) * point_block_size);

    }
}

void SetSolverOptionsFromFlags(BALProblem* balProblem,const BundleParams& params,g2o::SparseOptimizer* optimizer)
{
    //省去了判断用稠密还是稀疏求解器     to see
    //linearSolver.reset(new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>());      //可以用rest进行重新初始化

    std::unique_ptr<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>> linearSolver(new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>());

    linearSolver->setBlockOrdering(true);
    std::unique_ptr<BalBlockSolver> solver_ptr(new BalBlockSolver(std::move(linearSolver)));


    g2o::OptimizationAlgorithmWithHessian* solver; //to see
    if(params.trust_region_strategy == "levenberg_marquardt"){
        solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    }
    else if(params.trust_region_strategy == "dogleg"){
        solver = new g2o::OptimizationAlgorithmDogleg(std::move(solver_ptr));
    }
    else
    {
        std::cout << "Please check your trust_region_strategy parameter again.."<< std::endl;
        exit(EXIT_FAILURE);
    }

    optimizer->setAlgorithm(solver);

}
void SolveProblem(const char* filename,const BundleParams& params){
    BALProblem bal_problem(filename);
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observatoins. " << std::endl;
    if(!params.initial_ply.empty()){
        bal_problem.WriteToPLYFile(params.initial_ply);
    }
    std::cout << "beginning problem..." << std::endl;
    // add some noise for the intial value
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);

    std::cout << "Normalization complete..." << std::endl;
    g2o::SparseOptimizer optimizer;
    SetSolverOptionsFromFlags(&bal_problem, params, &optimizer);
    BuildProblem(&bal_problem, &optimizer, params);


    std::cout << "begin optimizaiton .."<< std::endl;
    // perform the optimizaiton
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(params.num_iterations);

    std::cout << "optimization complete.. "<< std::endl;
    // write the optimized data into BALProblem class
    WriteToBALProblem(&bal_problem, &optimizer);

    // write the result into a .ply file.
    if(!params.final_ply.empty()){
        bal_problem.WriteToPLYFile(params.final_ply);
    }

}

int main(int argc,char** argv){
    BundleParams params(argc,argv);
    if(params.input.empty()){
        std::cout<< "Usage: bundle_adjuster -input <path for dataset>";
        return  1;
    }
    SolveProblem(params.input.c_str(),params);
    //c_str()返回的是一个临时指针   指向input的指针
    return 0;
}