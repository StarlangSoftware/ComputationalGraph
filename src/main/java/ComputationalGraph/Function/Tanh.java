package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;

public class Tanh implements Function, Serializable {

    /**
     * Computes the Tanh activation for the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return Tanh(x).
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] oldValues = value.getData();
        double[] values = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            double oldValue = oldValues[i];
            values[i] = Math.tanh(oldValue);
        }
        return new FunctionResults(new Tensor(values, value.getShape()));
    }

    /**
     * Computes the derivative of the Tanh activation function.
     * @param value output of the Tanh(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        double[] oldValues = value.getData();
        double[] backwardValues = backward.getData();
        double[] values = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            double oldValue = oldValues[i];
            double backwardValue = backwardValues[i];
            values[i] = (1 - oldValue * oldValue) * backwardValue;
        }
        return new Tensor(values, value.getShape());
    }
}
