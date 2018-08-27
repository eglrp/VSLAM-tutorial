#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;
using namespace ceres;

//第一部分：构建cost fuction，即代价函数，也就是寻优的目标式。这个部分需要使用仿函数（functor）这一技巧来实现，做法是定义一个cost function的结构体，在结构体内重载（）运算符，具体实现方法后续介绍。
//第二部分：通过代价函数构建待求解的优化问题。
//第三部分：配置求解器参数并求解问题，这个步骤就是设置方程怎么求解、求解过程是否输出等，然后调用一下Solve方法。

struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};


struct CURVE_FITTING_COST{
    CURVE_FITTING_COST(double x,double y):_x(x),_y(y){}
    template <typename T>
    bool operator()(const T* const abc,T* residual)const{
        residual[0]=_y-ceres::exp(abc[0]*_x*_x+abc[1]*_x+abc[2]);
        return true;
    }
    const double _x,_y;
};

int main(int argc, char** argv) {
    double a=1,b=2,c=1;
    int N=100;
    double w_sigma=1.0;
    cv::RNG rng;
    double abc[3]={0,0,0};
    vector<double> x_data,y_data;
    cout<<"产生数据了"<<endl;
    for(int i=0;i<N;i++){
        double x=i/100.0;
        x_data.push_back(x);
        y_data.push_back(ceres::exp(a*x*x+b*x+c)+rng.gaussian(w_sigma));
    }
    Problem problem;
    for(int i=0;i<N;i++){
        problem.AddResidualBlock(new AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(
                new CURVE_FITTING_COST(x_data[i],y_data[i])),NULL,abc);
    }
    Solver::Options options;
    options.linear_solver_type=DENSE_QR;
    options.minimizer_progress_to_stdout=true;
    Solver::Summary summary;
    Solve(options,&problem,&summary);
    cout<<summary.BriefReport()<<endl;
    for(auto a:abc) cout<<a<<" ";
    cout<<endl;
}
