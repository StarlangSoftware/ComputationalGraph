package ComputationalGraph.Function;

import java.io.Serializable;

import Math.Tensor;

public class Power implements Function, Serializable {

    private final int n;

    public Power(int n) {
        this.n = n;
    }

    public Power() {
        this(2);
    }

    /**
     * Computes the Power of the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return pow(x) and input tensor.
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] tensorValues = value.getData();
        double[] values = new double[tensorValues.length];
        for (int i = 0; i < tensorValues.length; i++) {
            double val = tensorValues[i];
            values[i] = Math.pow(val, n);
        }
        return new FunctionResults(new Tensor(values, value.getShape()), new Tensor(tensorValues, value.getShape()));
    }

    /**
     * Computes the derivative of the Power function.
     * @param value input of the Power(x).
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
            double derivative = n * Math.pow(val, n - 1);
            double backwardValue = backwardValues[i];
            values[i] = derivative * backwardValue;
        }
        return new Tensor(values, value.getShape());
    }
}
