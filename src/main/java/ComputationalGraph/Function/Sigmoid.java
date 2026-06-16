package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;

public class Sigmoid implements Function, Serializable {

    /**
     * Computes the Sigmoid activation for the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return Sigmoid(x).
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] tensorValues = value.getData();
        double[] values = new double[tensorValues.length];
        for (int i = 0; i < tensorValues.length; i++) {
            double val = tensorValues[i];
            double sigmoid = 1.0 / (1.0 + Math.exp(-val));
            values[i] = sigmoid;
        }
        return new FunctionResults(new Tensor(values, value.getShape()));
    }

    /**
     * Computes the derivative of the Sigmoid activation function.
     * @param value output of the Sigmoid(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        double[] tensorValues = value.getData();
        double[] backwardValues = backward.getData();
        double[] values = new double[tensorValues.length];
        for (int i = 0; i < tensorValues.length; i++) {
            double val = tensorValues[i];
            double derivative = val * (1 - val);
            double backwardValue = backwardValues[i];
            values[i] = derivative * backwardValue;
        }
        return new Tensor(values, value.getShape());
    }
}
