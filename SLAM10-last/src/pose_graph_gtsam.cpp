//
// Created by gtkansy on 18-5-24.
//
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
/*#include <sophus/so3.hpp>
#include <sophus/se3.hpp>*/

//加入gtsam的头文件
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>


using namespace std;
/*using namespace Sophus;*/


int main(){
    ifstream fin("/home/gtkansy/CLionProjects/SLAM10-last/data/sphere.g2o");
    gtsam::NonlinearFactorGraph::shared_ptr graph(new gtsam::NonlinearFactorGraph);
    //因子图
    gtsam::Values::shared_ptr initial(new gtsam::Values);  //初始值
    int cntVertex=0, cntEdge = 0;
    cout<<"reading from g2o file"<<endl;
//顶点对应到初始值，边对应到因子，加入
    while (!fin.eof()){
        string tag;
        fin>>tag;
        if ( tag == "VERTEX_SE3:QUAT" ){
            //顶点
            gtsam::Key id;
            fin>>id;
            double data[7];
            for(int i=0;i<7;i++)
                fin>>data[i];
            //转换成gtsam的pose3
            gtsam::Rot3 R=gtsam::Rot3::quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t ( data[0], data[1], data[2] );
            initial->insert(id,gtsam::Pose3(R,t));
            //添加初始值
            cntVertex++;
        }

        else if(tag=="EDGE_SE3:QUAT"){
            gtsam::Matrix m=gtsam::I_6x6;
            gtsam::Key id1,id2;   //顶点是key
            fin>>id1>>id2;
            double data[7];
            for ( int i=0; i<7; i++ ) fin>>data[i];
            gtsam::Rot3 R = gtsam::Rot3::Quaternion ( data[6], data[3], data[4], data[5] );
            gtsam::Point3 t ( data[0], data[1], data[2] );
            for(int i=0;i<6;i++){
                for(int j=i;j<6;j++){
                    double mij;
                    fin>>mij;
                    m(i,j)=mij;
                    m(j,i)=mij;
                }
            }
            // g2o的信息矩阵定义方式与gtsam不同，这里对它进行修改
            gtsam::Matrix mgtsam = gtsam::I_6x6;
            mgtsam.block<3,3> ( 0,0 ) = m.block<3,3> ( 3,3 ); // cov rotation
            mgtsam.block<3,3> ( 3,3 ) = m.block<3,3> ( 0,0 ); // cov translation
            mgtsam.block<3,3> ( 0,3 ) = m.block<3,3> ( 0,3 ); // off diagonal
            mgtsam.block<3,3> ( 3,0 ) = m.block<3,3> ( 3,0 ); // off diagonal

            gtsam::SharedNoiseModel model=gtsam::noiseModel::Gaussian::Information(mgtsam);
            gtsam::NonlinearFactor::shared_ptr factor(
                    new gtsam::BetweenFactor<gtsam::Pose3>(id1,id2,gtsam::Pose3(R,t),model)
                    );
            //假设高斯模型的噪声
            graph->push_back(factor);
            cntEdge++;

        }
        if(!fin.good())
            break;

    }

    cout<<"read total "<<cntVertex<<" vertices, "<<cntEdge<<" edges."<<endl;
    //固定第一个顶点，在gtsam中相当于添加一个先验因子

    gtsam::NonlinearFactorGraph graphWithPrior=*graph;
    gtsam::noiseModel::Diagonal::shared_ptr priorModel=
            gtsam::noiseModel::Diagonal::Variances(
                    ( gtsam::Vector ( 6 ) <<1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6 ).finished()
            );
    gtsam::Key firstKey=0;
    for(const gtsam::Values::ConstKeyValuePair& key_value:*initial){
        cout<<"Adding prior to g2o file "<<endl;
        graphWithPrior.add(gtsam::PriorFactor<gtsam::Pose3>(
                key_value.key,key_value.value.cast<gtsam::Pose3>(),priorModel
                ));
        //加入先验的步骤   新来一个graph，在initial的基础上
        break;
    }

    // 开始因子图优化，配置优化选项
    cout<<"optimizing the factor graph"<<endl;
    // 我们使用 LM 优化
    gtsam::LevenbergMarquardtParams params_lm;
    params_lm.setVerbosity("ERROR");
    params_lm.setMaxIterations(20);
    params_lm.setLinearSolverType("MULTIFRONTAL_QR");
    gtsam::LevenbergMarquardtOptimizer optimizer_LM( graphWithPrior, *initial, params_lm );

    // 你可以尝试下 GN
    // gtsam::GaussNewtonParams params_gn;
    // params_gn.setVerbosity("ERROR");
    // params_gn.setMaxIterations(20);
    // params_gn.setLinearSolverType("MULTIFRONTAL_QR");
    // gtsam::GaussNewtonOptimizer optimizer ( graphWithPrior, *initial, params_gn );

    gtsam::Values result = optimizer_LM.optimize();
    cout<<"Optimization complete"<<endl;
    cout<<"initial error: "<<graph->error ( *initial ) <<endl;
    cout<<"final error: "<<graph->error ( result ) <<endl;

    cout<<"done. write to g2o ... "<<endl;


    //写入 g2o 文件，同样伪装成 g2o 中的顶点和边，以便用 g2o_viewer 查看。
    ofstream fout("/home/gtkansy/CLionProjects/SLAM10-last/data/result_gtsam.g2o");
    //顶点    result
    for(const gtsam::Values::ConstKeyValuePair& key_value:result){
        gtsam::Pose3 pose=key_value.value.cast<gtsam::Pose3>();
        gtsam::Point3 p=pose.translation();
        gtsam::Quaternion q=pose.rotation().toQuaternion();  //转换
        fout<<"VERTEX_SE3:QUAT "<<key_value.key<<" "
            <<p.x() <<" "<<p.y() <<" "<<p.z() <<" "
            <<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<" "<<endl;

    }
    //边  graph
    for(gtsam::NonlinearFactor::shared_ptr factor:*graph){
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr f=
                dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
        if(f){
            gtsam::SharedNoiseModel model = f->noiseModel();
            gtsam::noiseModel::Gaussian::shared_ptr gaussianModel = dynamic_pointer_cast<gtsam::noiseModel::Gaussian>( model );
            if(gaussianModel){
                //write the edge information

                gtsam::Matrix info=gaussianModel->R().transpose()*gaussianModel->R();

                gtsam::Pose3 pose=f->measured();
                gtsam::Point3 p=pose.translation();//transpose是转至，这个不一样
                gtsam::Quaternion q = pose.rotation().toQuaternion();
                fout<<"EDGE_SE3:QUAT "<<f->key1()<<" "<<f->key2()<<" "
                    <<p.x() <<" "<<p.y() <<" "<<p.z() <<" "
                    <<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<" ";
                gtsam::Matrix infoG2o = gtsam::I_6x6;
                infoG2o.block(0,0,3,3) = info.block(3,3,3,3); // cov translation
                infoG2o.block(3,3,3,3) = info.block(0,0,3,3); // cov rotation
                infoG2o.block(0,3,3,3) = info.block(0,3,3,3); // off diagonal
                infoG2o.block(3,0,3,3) = info.block(3,0,3,3); // off diagonal
                for ( int i=0; i<6; i++ )
                    for ( int j=i; j<6; j++ )
                    {
                        fout<<infoG2o(i,j)<<" ";
                    }
                fout<<endl;
            }
        }
    }
    fout.close();
    cout<<"done"<<endl;


}