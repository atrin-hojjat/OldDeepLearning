//
// Created by Atrin Hojjat on 6/26/16.
//

#include "Learn.h"

Learn::Learn(MatrixXd input, MatrixXd output,double learning_rate,double regularization_parameter)  : regularization_parameter(regularization_parameter),learning_rate(learning_rate),input(input),output(output)
{
    learn();
    drawLearningPlot();
}

Learn::Learn(MatrixXd input, MatrixXd output,double learning_rate,double regularization_parameter,long max_iters)  : regularization_parameter(regularization_parameter),learning_rate(learning_rate),input(input),output(output),max_iters(max_iters)
{
    learn();
    drawLearningPlot();
}

void Learn::learn()
{
    //Minimization

    VectorXd avgs(input.rows()),rates(input.rows());

    for (int i = 0; i < input.rows(); ++i) {
        if(!(input.row(i).maxCoeff() >10 || input.row(i).minCoeff() <-10)) {
            avgs(i) = 0;
            rates(i) = 1;
            continue;
        }
        if(input.row(i).maxCoeff() == input.row(i).minCoeff()) {
            avgs(i) = 0;
            rates(i) = 1;
            continue;
        }
        avgs(i) = input.row(i).mean();
        rates(i) = 1/(input.row(i).maxCoeff() - input.row(i).minCoeff());
        cout << avgs(i) << endl << rates(i) << endl;
        input.row(i) = (input.row(i).array()-avgs(i))*rates(i);
    }

    cout << endl << input << endl<<endl;

    //Gradient Decent

    MatrixXd w = MatrixXd::Zero(output.rows(),input.rows());

    auto pround = [](MatrixXd mat,double percision){
        MatrixXd ret(mat.rows(),mat.cols());
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                ret(i,j) = round(mat(i,j)*(1/percision))*percision;
            }
        }
        return ret;
    };

    auto J = [this](MatrixXd w)-> double{
        double out = 0;

        double regular = 0;

        for (int j = 0; j < input.rows(); ++j) {
            regular += w(j);
        }

        regular *= regularization_parameter;

        for (int i = 0; i < input.cols(); ++i) {
            VectorXd delta = w*input.col(i)-output.col(i);
            out+= delta.dot(delta);
        }

        out+=regular * input.cols();//Regularization

        out /= input.cols()*2;

        return out;
    };

    iter_stats = vector<double>();

    int x=0;
    MatrixXd wpast = MatrixXd::Zero(output.rows(),input.rows());

    bool m = max_iters > 0;

    for(;;){
        MatrixXd delta = MatrixXd::Zero(output.rows(),input.rows());

        for (int i = 0; i < input.cols(); ++i) {
            delta+=(w*input.col(i)-output.col(i))*input.col(i).transpose();
        }

        if(delta.isZero(1e-10))
            break;

        wpast = w;
        w*=1-learning_rate*(regularization_parameter/input.cols());
        w-=(learning_rate*delta)/input.cols();

        if((w-wpast).isZero(1e-10))
            break;

        iter_stats.insert(iter_stats.end(),J(w));

        x++;

        if(m && x>=max_iters)
            break;
    }

    w = pround(w,1e-10);

    cout << w;

    cout << endl << "Number Of Iterations : " << x << endl;
    cout << "Cost Function : " << J(w) << endl;

    auto test = [avgs,rates,w](VectorXd input) -> VectorXd{
        VectorXd output;

        // Normalize
        for (int i = 0; i < input.size(); ++i) {
            input(i) = (input(i)-avgs(i))*rates(i);
        }
        // Finding Value

        output = w*input;

        return output;
    };

    this->weight = w;

    this->test_func = test;
//    for_each(iter_stats.begin(),iter_stats.end(),[](double x){cout << x << ' '; });
}

void Learn::drawLearningPlot()
{
    vtkSmartPointer<vtkTable> table =
            vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> arrX =
            vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("Number Of Iterations");
    table->AddColumn(arrX);

    vtkSmartPointer<vtkFloatArray> arrC =
            vtkSmartPointer<vtkFloatArray>::New();
    arrC->SetName("Cost Function");
    table->AddColumn(arrC);

    // Fill in the table with some example values
    int numPoints = iter_stats.size();
    table->SetNumberOfRows(numPoints);
    for (int i = 0; i < numPoints; ++i)
    {
        table->SetValue(i, 0, i);
        table->SetValue(i, 1, iter_stats[i]);
    }

    // Set up the view
    vtkSmartPointer<vtkContextView> view =
            vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

    // Add multiple line plots, setting the colors etc
    vtkSmartPointer<vtkChartXY> chart =
            vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    vtkPlot *line = chart->AddPlot(vtkChart::POINTS);
    line->SetInputData(table, 0, 1);
    line->SetColor(1, 0, 0);
    line->SetWidth(2.0);
    line = chart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, 1);
    line->SetColor(0, 0, 1);
    line->SetWidth(1.5);
    view->GetInteractor()->Initialize();
    view->GetInteractor()->Start();
}
