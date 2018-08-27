#include <iostream>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <cmath>

//首先引入引入核心控件中的基础顶点和边
#include <g2o/core/base_vertex.h>  //include进核心构件中基础顶点头文件。引进后可以自己派生定义顶点.vertex:顶点
#include <g2o/core/base_unary_edge.h>  //include进核心构件中的基础一元边头文件，自定边时直接继承类重写就是了

//然后引入核心控件中的求解器
#include <g2o/core/block_solver.h>  //include进核心构件中的块求解器头文件

//引入各种优化算法的头文件，这里有好多可以引进，用啥引用啥就是了
#include <g2o/core/optimization_algorithm_gauss_newton.h>  //include进核心构件中的GN优化算法头文件
#include <g2o/core/optimization_algorithm_levenberg.h>  //include进核心构件中的LM优化算法头文件
#include <g2o/core/optimization_algorithm_dogleg.h>   //include进核心构件中的DL优化算法头文件

//引入求解器的求解方法,注意这里不是core文件中的，而是solvers中的稠密中的线性稠密求解器
#include <g2o/solvers/dense/linear_solver_dense.h>


using namespace std;

//自定义顶点类型，模板参数：优化变量维度３维，数据类型是Eigen::Vector3d
//这里的优化变量是一个[a, b, c]构成的数组，所以维度是３，类型就是Eigen中的矢量Vector3d，也就是数组
//curve曲线，curve fitting曲线拟合
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl()
    {
        _estimate <<0, 0, 0;
    }

    virtual void oplusImpl(const double* update)//Impl：接口实现.具体意思?
    {
        _estimate += Eigen::Vector3d(update);
    }

    //存盘和读盘，这里留空
    virtual bool read (istream& in) {}
    virtual bool write (ostream& out) const {}
};

//定义边，边为观测值，这里为函数误差，是一维的double类型，链接的顶点类型是上方定义好的CurveFittingVertex
class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x): BaseUnaryEdge(), _x(x) {}//子类构造时先调用基类的构造函数

    //计算曲线模型误差，也就是观测值，也就是边
    void computeError()
    {
        //_vertices是在class Edge 类中定义的一个 VertexContainer类型的protect变量. VertexContainer _vertices
        //static_cast<type A> (type B);标准转换函数，将B类型数据转换成A类型。这里将_vertices[0]转换成了const CurveFittingVertex*类型，为一个指针，赋值给v
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        //调用v的estimate()方法，赋值给abc,注意这里的abc是一个三维向量类型的。也就是个三double元素的数组。
        const Eigen::Vector3d abc = v->estimate();
        //计算误差，为测量值减去计算值，这里的_measurement也就是y值。但不知为和是二维的。这里的abc是３＊１阵，上方public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>模板参数传的也是１维。
        _error(0,0) = _measurement - std::exp( abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0) ) ;//这里 _measurement 为 y 值
    }

    virtual bool read (istream& in) {}
    virtual bool write(ostream& out) const {}

    double _x;//定义一个_x，用于数据读入计算误差
};



int main()
{

    double a=1.0, b=2.0, c=1.0;         // 真实参数值
    int N=100;                          // 数据点
    double w_sigma=1.0;                 // 噪声Sigma值
    cv::RNG rng;                        // OpenCV随机数产生器
    double abc[3] = {0,0,0};            // abc参数的估计值

    vector<double> x_data, y_data;      // 数据

    cout<<"generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (
                exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma )
        );
        cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }
    //到这步依旧是产生随机变量数组x_data和y_data

    //构建图优化，先设定g2o
    //BlockSolverTraits<3,1>块求解器特征<3, 1>
    //定义具有3, 1特征的块求解器类型为Block类型。每个误差项优化变量维度为3，误差值维度为1
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> Block;
   // Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();//线性方程求解器
   // Block* solver_ptr = new Block(linearSolver);
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverDense<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    // 梯度下降方法，从GN, LM, DogLeg 中选
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );

    g2o::SparseOptimizer optimizer;//稀疏优化，图模型，稀疏求解器
    optimizer.setAlgorithm(solver);// 设置求解器
    optimizer.setVerbose(true);// 打开调试输出

    //往图中增加顶点,这里就一个顶点，所以不用循环
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);

    //往途中增加边
    for (int i = 0; i < N; ++i)
    {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);//设置链接的顶点
        edge->setMeasurement(y_data[i]);// 观测数值
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity()*1/(w_sigma*w_sigma));
        optimizer.addEdge(edge);
    }

    //执行优化
    cout<<"start optimization"<<endl;
    optimizer.initializeOptimization();
    optimizer.optimize(100);//设置迭代次数

    //输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout<<"estimated model: "<<abc_estimate.transpose()<<endl;



    return 0;
}