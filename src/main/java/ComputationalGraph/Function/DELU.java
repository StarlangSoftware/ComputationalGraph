package ComputationalGraph.Function;

import java.io.Serializable;

import Math.Tensor;

public class DELU implements Function, Serializable {

    private final double a;
    private final double b;
    private final double xc;

    public DELU(double a, double b, double xc) {
        this.a = a;
        this.b = b;
        this.xc = xc;
    }

    public DELU() {
        this(1.0, 2.0, 1.25643);
    }

    /**
     * Computes the DELU activation for the given value tensor.
     * @param value The tensor whose values are to be computed.
     * @return DELU(x).
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] oldValues = value.getData();
        double[] values = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            double oldValue = oldValues[i];
            if (oldValue > this.xc) {
                values[i] = oldValue;
            } else {
                values[i] = (Math.exp(this.a * oldValue) - 1) / this.b;
            }
        }
        return new FunctionResults(new Tensor(values, value.getShape()));
    }

    /**
     * Computes the derivative of the DELU activation function.
     * @param value output of the DELU(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        double[] oldValues = value.getData();
        double[] backwardValues = backward.getData();
        double[] values = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            double backwardValue = backwardValues[i];
            double oldValue = oldValues[i];
            if (oldValue > this.xc) {
                values[i] = backwardValue;
            } else {
                values[i] = backwardValue * ((oldValue * this.b + 1) * (this.a / this.b));
            }
        }
        return new Tensor(values, value.getShape());
    }
}
