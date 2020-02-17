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

_transport_func getTransportFunc(int type){
    switch (type){
        case sigmoid:
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
        case pureline:
            return _transport_func(
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret (in);
                        return ret;
                    },
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret  = VectorXd::Ones(in.rows());
                        return ret;
                    }
            );
            break;
        case relu:
            return _transport_func(
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret(in);

                        for (int j = 0; j < ret.size(); ++j) {
                            ret(j) = ret(j) >0 ? ret(j) : 0;
                        }

                        return ret;
                    },
                    [](VectorXd in) -> VectorXd {
                        VectorXd ret(in);

                        for (int j = 0; j < ret.size(); ++j) {
                            ret(j) = (int) ret(j) >0;
                        }

                        return ret;
                    }
            );
            break;

    }
}
void __rand_init(MatrixXd *mat,int rows,int cols){
    *mat = MatrixXd::Zero(rows,cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double r = rand();
            r = r/(pow(10,(int)log10(r)+1))-.5 ;
            (*mat)(i,j) = r;
        }
    }
}

void __rand_init(VectorXd *mat,int rows){
    (*mat) = VectorXd::Zero(rows);

    for (int i = 0; i < rows; ++i) {
        double r = rand();
        r = r/(pow(10,(int)log10(r)+1))-.5 ;
        (*mat)(i) = r;
    }
}

void SDBP::__learn(MatrixXd input, MatrixXd output, double learning_rate/* Alpha */, vector<_layer> layers, int max_iters,double tolerance,
                   function<VectorXd(VectorXd)> *test) {

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

    double productivity = __productivity(layers);
    cout << productivity << endl;

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
//            cout << setw(7) << (input.cols())*(iter_num)+i+1 << (((i+1) % 20) == 0 ? "\n" : "") ;

        }
        for (int m = 0; m < layers.size(); ++m) {
            layers[m].weighs -= (learning_rate/input.cols())*delta_w[m];
            layers[m].bias -= (learning_rate/input.cols())*delta_b[m];
        }
//        cout << setw(7) << ++iter_num << " " << ((iter_num % 20) == 0 ? "\n" : "");

        double productivity = __productivity(layers);
        cout << productivity << endl;

        if (productivity < tolerance) {
            cout << productivity << endl;
            break;
        }


        if (iter_num > max_iters && max_iters > 0)
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