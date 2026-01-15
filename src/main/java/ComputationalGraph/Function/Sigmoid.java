package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Sigmoid implements Function, Serializable {

    /**
     * Computes the Sigmoid activation for the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return Sigmoid(x).
     */
    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        for (double val : tensorValues) {
            double sigmoid = 1.0 / (1.0 + Math.exp(-val));
            values.add(sigmoid);
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Computes the derivative of the Sigmoid activation function.
     * @param value output of the Sigmoid(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (int i = 0; i < tensorValues.size(); i++) {
            double val = tensorValues.get(i);
            double derivative = val * (1 - val);
            double backwardValue = backwardValues.get(i);
            values.add(derivative * backwardValue);
        }
        return new Tensor(values, value.getShape());
    }
}
