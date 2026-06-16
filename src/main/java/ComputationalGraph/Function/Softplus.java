package ComputationalGraph.Function;

import java.io.Serializable;

import Math.Tensor;

public class Softplus implements Function, Serializable {

    /**
     * Computes the Softplus activation function for the given tensor.
     * The Softplus function is defined as ln(1 + exp(x)) and is applied element-wise.
     * @param value The input tensor whose values are to be computed.
     * @return A new tensor with the Softplus activation applied to each element.
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double[] tensorValues = value.getData();
        double[] values = new double[tensorValues.length];
        for (int i = 0; i < tensorValues.length; i++) {
            double val = tensorValues[i];
            values[i] = Math.log(1.0 + Math.exp(val));
        }
        return new FunctionResults(new Tensor(values, value.getShape()));
    }

    /**
     * Computes the derivative of the Softplus activation function.
     * @param value The output tensor of the Softplus activation.
     * @param backward The backward tensor used for gradient propagation.
     * @return A tensor representing the gradient values for the corresponding input.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        double[] tensorValues = value.getData();
        double[] backwardValues = backward.getData();
        double[] values = new double[tensorValues.length];
        for (int i = 0; i < tensorValues.length; i++) {
            double val = tensorValues[i];
            double backwardValue = backwardValues[i];
            values[i] = (1 - Math.exp(-val)) * backwardValue;
        }
        return new Tensor(values, value.getShape());
    }
}
