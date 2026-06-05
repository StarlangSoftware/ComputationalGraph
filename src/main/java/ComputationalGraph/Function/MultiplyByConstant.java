package ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;
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
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) tensor.getData();
        for (double val : tensorValues) {
            double newVal = constant * val;
            values.add(newVal);
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
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (double backwardValue : backwardValues) {
            values.add(constant * backwardValue);
        }
        return new Tensor(values, value.getShape());
    }
}
