package ComputationalGraph.Function;

import java.io.Serializable;

import Math.Tensor;

public class Logarithm implements Function, Serializable {

    /**
     * Applies the natural logarithm function to each element of the input tensor.
     * @param value The tensor whose elements are to be transformed using the natural logarithm.
     * @return log(x) and x.
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] oldValues = value.getData();
        double[] values = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            double oldValue = oldValues[i];
            if (oldValue <= 0) {
                throw new IllegalArgumentException("Logarithm function input must be strictly positive. Found: " + oldValue);
            }
            values[i] = Math.log(oldValue);
        }
        return new FunctionResults(new Tensor(values, value.getShape()), new Tensor(oldValues, value.getShape()));
    }

    /**
     * Computes the derivative of the Logarithm function.
     * @param value input of the Logarithm(x).
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
            double derivative = 1.0 / val;
            double backwardValue = backwardValues[i];
            values[i] = derivative * backwardValue;
        }
        return new Tensor(values, value.getShape());
    }
}
