package ComputationalGraph.Function;

import java.io.Serializable;
import Math.Tensor;

public class MultiplyByConstant implements Function, Serializable {

    private final double constant;

    public MultiplyByConstant(double constant) {
        this.constant = constant;
    }

    /**
     * Computes a new tensor by scaling all elements of the input tensor
     * by a predetermined constant value.
     * @param tensor The input tensor whose values are to be scaled.
     * @return A new tensor where each element is the result of multiplying
     *         the corresponding element in the input tensor by the constant
     *         value.
     */
    @Override
    public FunctionResults calculate(Tensor tensor) {
        double[] tensorValues = tensor.getData();
        double[] values = new double[tensorValues.length];
        for (int i = 0; i < tensorValues.length; i++) {
            double val = tensorValues[i];
            double newVal = constant * val;
            values[i] = newVal;
        }
        return new FunctionResults(new Tensor(values, tensor.getShape()));
    }

    /**
     * Computes the derivative of the MultiplyByConstant operation.
     * This method calculates the gradient with respect to the input tensor
     * based on the constant scaling factor used in the forward pass.
     * @param value The input tensor from the forward pass.
     * @param backward The backward tensor representing the gradient
     *                 of the loss with respect to the output tensor.
     * @return A tensor containing the gradient of the loss with respect
     *         to the input tensor.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        double[] backwardValues = backward.getData();
        double[] values = new double[backwardValues.length];
        for (int i = 0; i < backwardValues.length; i++) {
            double backwardValue = backwardValues[i];
            values[i] = constant * backwardValue;
        }
        return new Tensor(values, value.getShape());
    }
}
