package ComputationalGraph.Function;

import java.io.Serializable;

import Math.Tensor;

public class DivideByDimensionSize implements Function, Serializable {

    private final int dimension;

    public DivideByDimensionSize(int dimension) {
        this.dimension = dimension;
    }

    /**
     * Performs a calculation where each element of the input tensor is divided
     * by the size of the specified dimension within the tensor's shape.
     * @param value The input tensor to be processed.
     * @return A new tensor where each element is divided by the size of the specified dimension.
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        if (dimension == -1) {
            return new FunctionResults(value);
        }
        double[] tensorValues = value.getData();
        double[] values = new double[tensorValues.length];
        int size = value.getShape()[dimension];
        for (int i = 0; i < tensorValues.length; i++) {
            double val = tensorValues[i];
            values[i] = (1.0 / size) * (val);
        }
        return new FunctionResults(new Tensor(values, value.getShape()));
    }

    /**
     * Computes the derivative of the function where each element of the tensor is
     * divided by the size of the specified dimension, propagating the gradients
     * during backpropagation.
     * @param value The input tensor that represents the forward pass output.
     * @param backward The backward tensor containing the gradients from the subsequent layers.
     * @return A tensor containing the computed gradients for the current operation.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        if (dimension == -1) {
            return backward;
        }
        double[] backwardValues = backward.getData();
        double[] values = new double[backwardValues.length];
        int size = value.getShape()[dimension];
        for (int i = 0; i < backwardValues.length; i++) {
            double backwardValue = backwardValues[i];
            double derivative = (1.0 / size);
            values[i] = derivative * backwardValue;
        }
        return new Tensor(values, value.getShape());
    }
}
