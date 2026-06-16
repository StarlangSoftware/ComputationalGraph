package ComputationalGraph.Function;

import java.io.Serializable;

import Math.*;

public class Negation implements Function, Serializable {

    /**
     * Negates the values of the given tensor.
     * @param value The tensor whose values are to be negated.
     * @return The negated tensor.
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] oldValues = value.getData();
        double[] values = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            values[i] = -oldValues[i];
        }
        return new FunctionResults(new Tensor(values, value.getShape()));
    }

    /**
     * Calculates the derivative of the Negation function.
     * @param value output of the Negation function.
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        double[] backwardValues = backward.getData();
        double[] values = new double[backwardValues.length];
        for (int i = 0; i < backwardValues.length; i++) {
            values[i] = -backwardValues[i];
        }
        return new Tensor(values, value.getShape());
    }
}
