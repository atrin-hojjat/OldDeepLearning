//
// Created by Atrin Hojjat on 7/23/16.
//

#include "Learn.h"

const auto sig = [](VectorXd in) -> VectorXd {
    VectorXd ret (in.rows());
    for (int i = 0; i < in.rows(); ++i) {
        ret(i) = 1/(exp(-in(i))+1);
    }
//    cout << ret << endl;
    return ret;
};

_transport_func             getTransportFunc(int type){
    switch (type){
        case trans_funcs::sigmoid:
            return _transport_func(sig,
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret (in.rows());
                        VectorXd sig_val = sig(in);
                        for (int i = 0; i < in.rows(); ++i) {
                            ret(i) = (1-sig_val(i))*(sig_val(i));
                        }
//                        cout << ret << endl;
                        return ret;
                    }
            );
            break;
        case trans_funcs::pureline:
            return _transport_func(
                    [](VectorXd in) -> VectorXd {
                        return in;
                    },
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret  = VectorXd::Ones(in.rows());
                        return ret;
                    }
            );
            break;
        case trans_funcs::relu:
            return _transport_func(
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret(in.rows());

                        for (int j = 0; j < ret.size(); ++j) {
                            ret(j) = in(j) >0 ? in(j) : 0;
                        }

                        return ret;
                    },
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret(in.rows());

                        for (int j = 0; j < ret.size(); ++j) {
                            ret(j) = (int) in(j) >0;
                        }

                        return ret;
                    }
            );
            break;
        case trans_funcs::tanh:
            return _transport_func(
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret(in.rows());

                        for (int j = 0; j < ret.size(); ++j) {
                            ret(j) = std::tanh(in(j));
                        }

                        return ret;
                    },
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret(in.rows());

                        for (int j = 0; j < ret.size(); ++j) {
                            ret(j) = 1-pow(std::tanh(in(j)),2);
                        }

                        return ret;
                    }
            );
            break;
        case trans_funcs::tanh_sig:
            return _transport_func(
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret(in.rows());

                        for (int j = 0; j < ret.size(); ++j) {
                            ret(j) = 1.7159*std::tanh(0.666667*in(j)) +0.001 * in(j);
                        }

                        return ret;
                    },
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret(in.rows());

                        for (int j = 0; j < ret.size(); ++j) {
                            ret(j) = 1.14393/pow(cosh(0.666667*in(j)),2) + 0.001;
                        }

                        return ret;
                    }
            );
            break;

    }
}

void __rand_init            (MatrixXd *mat,int rows,int cols){
    *mat = MatrixXd::Zero(rows,cols);

    const long double stddev =sqrt(1.0/(cols*rows));

    std::random_device rd;

    std::mt19937 e2(rd());

    std::normal_distribution<double> dist(0,stddev);

    for (int i = 0; i < rows; ++i) {

        for (int j = 0; j < cols; ++j) {
            double r = dist(e2);
            
            (*mat)(i,j) = r;
        }
    }
}

void __rand_init            (VectorXd *mat,int rows){
    (*mat) = VectorXd::Zero(rows);

    for (int i = 0; i < rows; ++i) {
        double r = rand();
        r = r/(pow(10,(int)log10(r)+1))-.5 ;
        (*mat)(i) = r;
    }
}

void SDBP::__learn          (MatrixXd input, MatrixXd output, double learning_rate/* Alpha */, vector<_layer> layers,
                             int max_iters,double tolerance, function<VectorXd(VectorXd)> *test) {

    VectorXd ls;
    MatrixXd pweighs;
    VectorXd pbias;

    int iter_num = 0;
    int inner_iter_num = 0;

    auto __productivity = [input,output](vector<_layer> layers) -> double {
        auto __test = [layers](VectorXd input) -> VectorXd {
            vector<VectorXd> layer_a = vector<VectorXd>();
            vector<VectorXd> layer_n = vector<VectorXd>();
            layer_n.push_back(layers[0].weighs * input + layers[0].bias);
            layer_a.push_back(layers[0].transport.transport_function(layer_n[0]));
            for (int j = 1; j < layers.size(); ++j) {
                layer_n.push_back(layers[j].weighs * layer_a[j - 1] + layers[j].bias);
                layer_a.push_back(layers[j].transport.transport_function(layer_n[j]));
            }
            return layer_a[layer_a.size() - 1];
        };
        double err = 0;
        for (int k = 0; k < input.cols(); ++k) {
            VectorXd err_v = output.col(k) - __test(input.col(k));
            double sq_err_v = err_v.dot(err_v);
            err+= sq_err_v;
        }
        err/= input.cols();
        return err;
    };


    for (; ;) { //Main Loop
        for (int i = 0; i < input.cols(); ++i) {
            vector<VectorXd> layer_a = vector<VectorXd>();
            vector<VectorXd> layer_n = vector<VectorXd>();
            vector<VectorXd> layer_sensitivities = vector<VectorXd>();
            //Propagation
            layer_n.push_back(layers[0].weighs * input.col(i) + layers[0].bias);
            layer_a.push_back(layers[0].transport.transport_function(layer_n[0]));
            for (int j = 1; j < layers.size(); ++j) {
                layer_n.push_back(layers[j].weighs * layer_a[j - 1] + layers[j].bias);
                layer_a.push_back(layers[j].transport.transport_function(layer_n[j]));
            }

            layer_sensitivities.insert(layer_sensitivities.begin(), -2 *
                                                                    layers[layers.size() -
                                                                           1].transport.DerivativeMatrix(
                                                                            layer_n[layer_n.size() - 1]) *
                                                                    (output.col(i) - layer_a[layer_a.size() - 1]));
            //Backpropagation
            for (int k = layers.size() - 1; k > 0; --k) {
                ls = layer_sensitivities[0];
                pweighs = layers[k].weighs;
                pbias = layers[k].bias;
                layers[k].weighs -= learning_rate * layer_sensitivities[0] * layer_a[k - 1].transpose();
                layers[k].bias -= learning_rate * layer_sensitivities[0];
                layer_sensitivities.insert(layer_sensitivities.begin(),
                                           layers[k - 1].transport.DerivativeMatrix(layer_n[k - 1]) *
                                           pweighs.transpose() *
                                           ls); //For Next Iteration, change K to K-1
            }
            layers[0].weighs -= learning_rate * layer_sensitivities[0] * input.col(i).transpose();
            layers[0].bias -= learning_rate * layer_sensitivities[0];

            inner_iter_num++;
        }
        iter_num++;

        double productivity = 1;//__productivity(layers);

        if(productivity < tolerance) {
//            cout << productivity << endl;
            break;
        }

//        cout << iter_num << endl;


        if (iter_num >= max_iters && max_iters > 0)
            break;
    }

    *test = [layers](VectorXd input) -> VectorXd {
        vector<VectorXd> layer_a = vector<VectorXd>();
        vector<VectorXd> layer_n = vector<VectorXd>();
        layer_n.push_back(layers[0].weighs * input + layers[0].bias);
        layer_a.push_back(layers[0].transport.transport_function(layer_n[0]));
        for (int j = 1; j < layers.size(); ++j) {
            layer_n.push_back(layers[j].weighs * layer_a[j - 1] + layers[j].bias);
            layer_a.push_back(layers[j].transport.transport_function(layer_n[j]));
        }
        return layer_a[layer_a.size() - 1];
    };
}

void Batch_SDBP::__learn    (MatrixXd input, MatrixXd output, double learning_rate/* Alpha */, vector<_layer> layers,
                             int max_iters,double tolerance, unsigned int batch_size,function<VectorXd(VectorXd)> *test) {

    VectorXd ls;

    int iter_num = 0;

    auto __productivity = [input,output](vector<_layer> layers) -> double {
        auto __test = [layers](VectorXd input) -> VectorXd {
            vector<VectorXd> layer_a = vector<VectorXd>();
            vector<VectorXd> layer_n = vector<VectorXd>();
            layer_n.push_back(layers[0].weighs * input + layers[0].bias);
            layer_a.push_back(layers[0].transport.transport_function(layer_n[0]));
            for (int j = 1; j < layers.size(); ++j) {
                layer_n.push_back(layers[j].weighs * layer_a[j - 1] + layers[j].bias);
                layer_a.push_back(layers[j].transport.transport_function(layer_n[j]));
            }
            return layer_a[layer_a.size() - 1];
        };
        double err = 0;
        for (int k = 0; k < input.cols(); ++k) {
            VectorXd err_v = output.col(k) - __test(input.col(k));
            double sq_err_v = err_v.dot(err_v);
            err+= sq_err_v;
        }
        err/= input.cols();
        return err;
    };

    vector <double> iter_stats = vector<double>();

    for (; ;) { //Main Loop

        vector<MatrixXd> delta_w = vector<MatrixXd>();
        vector<VectorXd> delta_b = vector<VectorXd>();


        for (int l = 0; l < layers.size(); ++l) {
            delta_w.push_back(MatrixXd::Zero(layers[l].weighs.rows(),layers[l].weighs.cols()));
            delta_b.push_back(VectorXd::Zero(layers[l].bias.size()));
        }

        for (int i = 0; i < input.cols(); ++i) {
            vector<VectorXd> layer_a = vector<VectorXd>();
            vector<VectorXd> layer_n = vector<VectorXd>();
            vector<VectorXd> layer_sensitivities = vector<VectorXd>();
            //Propagation
            layer_n.push_back(layers[0].weighs * input.col(i) + layers[0].bias);
            layer_a.push_back(layers[0].transport.transport_function(layer_n[0]));
            for (int j = 1; j < layers.size(); ++j) {
                layer_n.push_back(layers[j].weighs * layer_a[j - 1] + layers[j].bias);
                layer_a.push_back(layers[j].transport.transport_function(layer_n[j]));
            }
//            cout << layer_a[layers.size()-1] << endl << endl;

            //Backpropagation
            layer_sensitivities.insert(layer_sensitivities.begin(), -2 *
                                                                    layers[layers.size() -
                                                                           1].transport.DerivativeMatrix(
                                                                            layer_n[layer_n.size() - 1]) *
                                                                    (output.col(i) - layer_a[layer_a.size() - 1]));
            for (int k = layers.size() - 1; k > 0; --k) {
                ls = layer_sensitivities[0];

                layer_sensitivities.insert(layer_sensitivities.begin(),
                                           layers[k - 1].transport.DerivativeMatrix(layer_n[k - 1]) *
                                           layers[k].weighs.transpose() *
                                           ls); //For Next Iteration we changed K to K-1
            }

            delta_w[0] += layer_sensitivities[0] * input.col(i).transpose();
            delta_b[0] += layer_sensitivities[0];
            for (int k = 1; k < layers.size(); ++k) {
                delta_w[k] += layer_sensitivities[k] * layer_a[k-1].transpose();
                delta_b[k] += layer_sensitivities[k];
            }

        }
        for (int m = 0; m < layers.size(); ++m) {
            layers[m].weighs -= learning_rate*delta_w[m]/input.cols();
            layers[m].bias -= learning_rate*delta_b[m]/input.cols();
        }

        ++iter_num;

        double productivity = __productivity(layers);
//        cout << productivity << endl;

        iter_stats.push_back(productivity);

        if (productivity < tolerance) {
//            cout << productivity << endl;
            break;
        }


        cout << iter_num;

        if (iter_num >= max_iters && max_iters > 0)
            break;
    }

    __plot_productivity_function__(iter_stats);

    *test = [layers](VectorXd input) -> VectorXd {
        vector<VectorXd> layer_a = vector<VectorXd>();
        vector<VectorXd> layer_n = vector<VectorXd>();
        layer_n.push_back(layers[0].weighs * input + layers[0].bias);
        layer_a.push_back(layers[0].transport.transport_function(layer_n[0]));
        for (int j = 1; j < layers.size(); ++j) {
            layer_n.push_back(layers[j].weighs * layer_a[j - 1] + layers[j].bias);
            layer_a.push_back(layers[j].transport.transport_function(layer_n[j]));
        }
        return layer_a[layer_a.size() - 1];
    };
}

void K_MEANS::__learn       (MatrixXd input, unsigned int num_clusters,unsigned int max_iters,
                             function<unsigned int (VectorXd)>* test) {
    const unsigned int num_inputs = input.cols();
    //Phase 1 : Initiating
    vector<VectorXd> clusters(num_clusters);
    for (int i = 0; i < num_clusters; ++i) {
        unsigned int at = rand() % num_inputs;
        clusters[i] = input.col(at);
    }

    vector<double> iter_stats(0);

    unsigned int iter_num = 0;
    for (; ;) {
        double err = 0;
        vector<VectorXd> means(num_clusters, VectorXd::Zero(input.rows()));
        vector<unsigned int> num_cases(num_clusters, 0);
        //Phase 2 : First Clustering
        for (int i = 0; i < num_inputs; i++) {
            VectorXd in = input.col(i);
            VectorXd temp = in - clusters[0];
            double min_dis = temp.dot(temp);
            double dis = 0;
            unsigned int min_pos = 0;
            for (int j = 1; j < num_clusters; ++j) {
                temp = in - clusters[j];
                dis = temp.dot(temp);
                if (dis < min_dis) {
                    min_dis = dis;
                    min_pos = j;
                }
            }
            err += min_dis;
            means[min_pos] += in;
            num_cases[min_pos]++;
        }
        err/=num_inputs;
        iter_stats.push_back (err);

        //Phase 3 : Changing Clusters
        for (int k = 0; k < num_clusters; ++k) {
            means[k] /= num_cases[k];
            clusters[k] = means[k];
        }
        if(++iter_num >= max_iters)break;
    }

    __plot_productivity_function__(iter_stats);

    *test = [clusters,num_clusters](VectorXd in) -> unsigned int{
        VectorXd temp = in - clusters[0];
        double min_dis = temp.dot(temp);
        double dis = 0;
        unsigned int min_pos = 0;
        for (int j = 1; j < num_clusters; ++j) {
            temp = in - clusters[j];
            dis = temp.dot(temp);
            if (dis < min_dis) {
                min_dis = dis;
                min_pos = j;
            }
        }
        return min_pos;
    };
}

void VLBP::__learn          (MatrixXd input, MatrixXd output, double initial_mue, double p, double l, vector<_layer> layers,
                             int max_iters, double tolerance, function<VectorXd(VectorXd)> *test) {

}

void LMBP::__learn          (MatrixXd input, MatrixXd output, double st_mue/* µ */,double delta/* ∂ */, vector<_layer> layers,
                             int max_epochs, double tolerance, function<VectorXd(VectorXd)> *test) {
    double mue = st_mue;


    int jacobs_dim_c = 0;
    int jacobs_dim_r = layers[layers.size() - 1].output_dimention_num * input.cols();
    int num_neurons = 0;
    for (int l = 0; l < layers.size(); ++l) {
        jacobs_dim_c += (layers[l].input_dimention_num + 1) * layers[l].output_dimention_num;
        num_neurons += layers[l].output_dimention_num;
    }

    vector<vector<VectorXd>> as = vector<vector<VectorXd>>();
    vector<vector<VectorXd>> ns = vector<vector<VectorXd>>();

    RowVectorXd err_vec(jacobs_dim_r);
    double err = 0;
    for (int k = 0; k < input.cols(); ++k) {
        VectorXd vec = input.col(k);
        vector<VectorXd> layer_a = vector<VectorXd>();
        vector<VectorXd> layer_n = vector<VectorXd>();
        layer_n.push_back(layers[0].weighs * vec + layers[0].bias);
        layer_a.push_back(layers[0].transport.transport_function(layer_n[0]));
        for (int j = 1; j < layers.size(); ++j) {
            layer_n.push_back(layers[j].weighs * layer_a[j - 1] + layers[j].bias);
            layer_a.push_back(layers[j].transport.transport_function(layer_n[j]));
        }
        layer_a[layer_a.size() - 1];

        as.push_back(layer_a);
        ns.push_back(layer_n);

        VectorXd err_v = output.col(k) - layer_a[layer_a.size() - 1];
        for (int i = 0; i < err_v.rows(); ++i) {
            err_vec.col(k * err_v.rows() + i) = err_v.row(i);
        }
        double sq_err_v = err_v.dot(err_v);
        err += sq_err_v;
    }
    err/=input.cols();
    double initial_productivity = err;

    bool minima = false;

    for (int iteration_num = 0; iteration_num < max_epochs && err > tolerance ; ++iteration_num) {
        err = 0;
        vector<vector<VectorXd>> nas = vector<vector<VectorXd>>();
        vector<vector<VectorXd>> nns = vector<vector<VectorXd>>();
        RowVectorXd nerr_vec(jacobs_dim_r);

        MatrixXd jacobian(jacobs_dim_r, jacobs_dim_c);
        int n_layers = layers.size();
        for (int i = 0; i < input.cols(); ++i) {
            //BackPropagation
            vector<MatrixXd> sensitivities(layers.size());
            sensitivities[n_layers - 1] = -layers[n_layers - 1].transport.DerivativeMatrix(ns[i][n_layers - 1]);
            for (int j = n_layers - 2; j >= 0; --j) {
                sensitivities[j] = layers[j].transport.DerivativeMatrix(ns[i][j]) * layers[j + 1].weighs.transpose() *
                                   sensitivities[j + 1];
            }
            int j_s_pos = layers[n_layers - 1].output_dimention_num * i;
            for (int k = 0; k < layers[n_layers - 1].output_dimention_num; ++k) {
                int j_col_s_pos = 0;
                for (int l = 0; l < layers[0].weighs.rows(); ++l) {
                    for (int m = 0; m < layers[0].weighs.cols(); ++m) {
                        jacobian(k + j_s_pos, j_col_s_pos++)
                                = sensitivities[0](l, k) * input(m,i);
                    }
                }
                for (int n = 0; n < layers[0].bias.size(); ++n) {
                    jacobian(k + j_s_pos, j_col_s_pos++)
                            = sensitivities[0](n, k );
                }
                for (int j = 1; j < n_layers; ++j) {
                    for (int l = 0; l < layers[j].weighs.rows(); ++l) {
                        for (int m = 0; m < layers[j].weighs.cols(); ++m) {
                            jacobian(k + j_s_pos, j_col_s_pos++)
                                    = sensitivities[j](l, k ) * as[i][j - 1](m);
                        }
                    }
                    for (int n = 0; n < layers[j].bias.size(); ++n) {
                        jacobian(k + j_s_pos, j_col_s_pos++)
                                = sensitivities[j](n, k );
                    }
                }
            }
        }

        VectorXd gradient = jacobian.transpose() * err_vec.transpose();
        MatrixXd jacob_inner_product = jacobian.transpose() * jacobian;

//        if(gradient.norm() > 1)
        while(true) {
            MatrixXd mt_1 =  jacob_inner_product+
                              MatrixXd::Identity(jacobian.cols(),jacobian.cols())*mue;
            MatrixXd mt_2 = mt_1.inverse();
            VectorXd delta_x = -1 * mt_2
                               * gradient;
            vector<_layer> new_layers = layers;

            int pos = 0;
            for (int i1 = 0; i1 < new_layers.size(); ++i1) {
                for (int i = 0; i < new_layers[i1].output_dimention_num; ++i) {
                    for (int j = 0; j < new_layers[i1].input_dimention_num; ++j) {
                        new_layers[i1].weighs(i, j) += delta_x(pos++);
                    }
                    new_layers[i1].bias(i) += delta_x(pos++);
                }
            }

            err = 0;

            for (int k = 0; k < input.cols(); ++k) {
                VectorXd vec = input.col(k);
                vector<VectorXd> layer_a = vector<VectorXd>();
                vector<VectorXd> layer_n = vector<VectorXd>();
                layer_n.push_back(new_layers[0].weighs * vec + new_layers[0].bias);
                layer_a.push_back(new_layers[0].transport.transport_function(layer_n[0]));
                for (int j = 1; j < new_layers.size(); ++j) {
                    layer_n.push_back(new_layers[j].weighs * layer_a[j - 1] + new_layers[j].bias);
                    layer_a.push_back(new_layers[j].transport.transport_function(layer_n[j]));
                }
                layer_a[layer_a.size() - 1];

                nas.push_back(layer_a);
                nns.push_back(layer_n);

                VectorXd err_v = output.col(k) - layer_a[layer_a.size() - 1];
                for (int i = 0; i < err_v.rows(); ++i) {
                    nerr_vec.col(k * err_v.rows() + i) = err_v.row(i);
                }
                double sq_err_v = err_v.dot(err_v);
                err += sq_err_v;
            }
            err/=input.cols();

            if(err == initial_productivity){
//                cout << " Minimum Reached,no more changes expected." << endl;
//                minima = true;
                break;
            }
            if (err < initial_productivity) {
                initial_productivity = err;
                as = nas;
                ns = nns;
                err_vec = nerr_vec;
                layers = new_layers;
                mue /= delta;
                break;
            } else {
                mue *= delta;
//                cout << err << endl
            }
        }

        if(gradient.norm() <1 || minima)
            break;

    }

    *test = [layers](VectorXd input) -> VectorXd {
        vector<VectorXd> layer_a = vector<VectorXd>();
        vector<VectorXd> layer_n = vector<VectorXd>();
        layer_n.push_back(layers[0].weighs * input + layers[0].bias);
        layer_a.push_back(layers[0].transport.transport_function(layer_n[0]));
        for (int j = 1; j < layers.size(); ++j) {
            layer_n.push_back(layers[j].weighs * layer_a[j - 1] + layers[j].bias);
            layer_a.push_back(layers[j].transport.transport_function(layer_n[j]));
        }
        return layer_a[layer_a.size() - 1];
    };
}
