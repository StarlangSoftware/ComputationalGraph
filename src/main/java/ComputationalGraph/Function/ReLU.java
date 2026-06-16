package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;

public class ReLU implements Function, Serializable {

    /**
     * Computes the ReLU activation for the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return ReLU(x).
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] oldValues = value.getData();
        double[] values = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            double oldValue = oldValues[i];
            values[i] = Math.max(oldValue, 0);
        }
        return new FunctionResults(new Tensor(values, value.getShape()));
    }

    /**
     * Computes the derivative of the ReLU activation function.
     * @param value output of the ReLU(x).
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
            if (oldValue > 0) {
                values[i] = backwardValue;
            } else {
                values[i] = 0.0;
            }
        }
        return new Tensor(values, value.getShape());
    }
}
