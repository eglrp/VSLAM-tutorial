//
// Created by gtkansy on 18-4-18.
//
#include <unordered_map>
#include <vector>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <numeric>

using std::unordered_map;
using std::vector;

class bayesianfilter{
    vector<double> posterior_;   //record updated_posterior;prior at next time
    vector<int> locations_;      //record location index

public:
    bayesianfilter(){}
    ~bayesianfilter(){};

    int predict_and_update(const vector<double> &similarity,
                           const double loop_threshold,
                           const int image_index)
    {
        assert(posterior_.size()==locations_.size());
        assert(image_index>=0);
        assert(similarity.size()==locations_.size());
    }
};



//
//  RTAB_bayesian.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-02-21.
//  Copyright © 2016 jimmy. All rights reserved.

#ifndef RTAB_bayesian_cpp
#define RTAB_bayesian_cpp


#include <unordered_map>
#include <vector>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <numeric>

using std::unordered_map;
using std::vector;

// III C. Bayesian Filter Update and D. loop closure hypothsis and selection
class RTAB_BayesianFilter
{
    vector<double> posterior_;       // posterior until time t   要有预定的先验
    vector<int>    locations_;       // location index    要先给

public:
    RTAB_BayesianFilter(){}
    ~RTAB_BayesianFilter(){}

    // similarity is calculated using equation (3) and (4)
    // N_wm: working memory size
    // T_loop: Table II
    // initial_prior: initial prior for a new location
    // image_index: current image index
    // return value: -1, now location. Otherwise, loop closure index and probability
    int predict_and_update(const vector<double> & similarity,
                           const double T_loop,
                           const double initial_prior,
                           const int image_index)
    {
        assert(posterior_.size() == locations_.size());
        assert(similarity.size() == locations_.size());
        assert(image_index >= 0);
        //assert(表达式);如果表达式的值为假，整个程序将退出，并输出一条错误信息。如果表达式的值为真则继续执行后面的语句。

        vector<double> updated_posterior(posterior_.size(), 0);
        for (int i = 0; i<locations_.size(); i++) {
            // assume current loop closure index is in locations_[i]
            int cur_loopclosure_index = locations_[i];  //闭环的不加，不闭环的就把当前图片id都加进去
            double cur_likelihood     = similarity[i];  //对应这当前的观测，用来跟更新
            // sum of all possible of S_{t-1}
            double belief = 0.0;
            //需要修改为t-p
            for (int j = 0; j<locations_.size(); j++) {
                int pre_loopclousre_index = locations_[j];
                double transition_prob = 1.0;
                double pre_prior = posterior_[j];    //t-1时刻的更新的后验作为t时刻的先验
                // estimate transition probability
                if (cur_loopclosure_index == -1 && pre_loopclousre_index == -1) {
                    transition_prob = 0.9;
                }
                else if(cur_loopclosure_index != -1 && pre_loopclousre_index == -1){
                    transition_prob = 0.1/locations_.size(); // assume working memory length is locations_.size()
                }
                else if(cur_loopclosure_index == -1 && pre_loopclousre_index != -1)
                {
                    transition_prob = 0.1;
                }
                else if(cur_loopclosure_index != -1 && pre_loopclousre_index != -1)
                {
                    // discreted gaussian
                    double neighbor_distance = locations_[i] - locations_[j]; // @todo. It should be distance in the graph
                    transition_prob = RTAB_BayesianFilter::gaussian_distributioin(0.0, 1.6, neighbor_distance);
                }
                belief += transition_prob * pre_prior;
            }
            updated_posterior[i] = cur_likelihood * belief;
        }
        // normalized posterior
        double sum = std::accumulate(updated_posterior.begin(), updated_posterior.end(), 0.0);
        for (int i = 0; i<updated_posterior.size(); i++) {
            updated_posterior[i] /= sum;
        }
        assert(locations_[0] == -1);

        // D. loop closure hypothsis and selection
        //只有index=-1的概率小于一定阈值的，我们才选择试着返回闭环的index
        if (updated_posterior[0] < T_loop) {
            // detected loop closure
            long max_prob_index = std::distance(updated_posterior.begin(), std::max_element(updated_posterior.begin(), updated_posterior.end()));
            int location_index = locations_[max_prob_index];
            printf("detect loop closure %d %f", location_index, updated_posterior[location_index]);

            // update posterior
            posterior_ = updated_posterior;
            return location_index;
        }
        else
        {
            // new location
            // update posterior and locations
            posterior_ = updated_posterior;
            posterior_.push_back(initial_prior);
            locations_.push_back(image_index);

            return -1;
        }
    }


private:
    static double gaussian_distributioin(const double mu, const double sigma, const double x)
    {
        assert(sigma > 0);
        double p = 1.0/(sigma * 2.0 * 3.14159) * exp(-(x-mu)*(x-mu)/(2.0 * sigma * sigma));
        return p;
    }


};

#endif /* RTAB_bayesian_cpp */

