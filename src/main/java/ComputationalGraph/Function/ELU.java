package ComputationalGraph.Function;

import java.io.Serializable;

import Math.Tensor;

public class ELU implements Function, Serializable {

    private final double a;

    public ELU(double a) {
        this.a = a;
    }

    public ELU() {
        this(1.0);
    }

    /**
     * Computes the ELU activation for the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return ELU(x).
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] oldValues = value.getData();
        double[] values = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            double oldValue = oldValues[i];
            if (oldValue < 0) {
                values[i] = a * (Math.exp(oldValue) - 1);
            } else {
                values[i] = oldValue;
            }
        }
        return new FunctionResults(new Tensor(values, value.getShape()));
    }

    /**
     * Computes the derivative of the ELU activation function.
     * @param value output of the ELU(x).
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
            if (oldValue < 0) {
                values[i] = (oldValue + a) * backwardValue;
            } else {
                values[i] = backwardValue;
            }
        }
        return new Tensor(values, value.getShape());
    }
}
