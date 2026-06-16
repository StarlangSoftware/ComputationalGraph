package ComputationalGraph.Function;

import java.io.Serializable;

import Math.Tensor;

public class AdditionByConstant implements Function, Serializable {

    private final double constant;

    public AdditionByConstant(double constant) {
        this.constant = constant;
    }

    /**
     * Computes a new tensor where a constant value is added to each element of the input tensor.
     * @param tensor The input tensor containing the data and shape.
     * @return A new tensor with the constant value added to each element, retaining the same shape as the input tensor.
     */
    @Override
    public FunctionResults calculate(Tensor tensor) {
        double[] tensorValues = tensor.getData();
        double[] values = new double[tensorValues.length];
        for (int i = 0; i < tensorValues.length; i++) {
            double val = tensorValues[i];
            double newVal = constant + val;
            values[i] = newVal;
        }
        return new FunctionResults(new Tensor(values, tensor.getShape()));
    }

    /**
     * Computes the derivative of the function given the output tensor and a backward tensor.
     * @param value The output tensor of the function.
     * @param backward The backward tensor representing the gradient from the subsequent layer.
     * @return backward
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward;
    }
}
