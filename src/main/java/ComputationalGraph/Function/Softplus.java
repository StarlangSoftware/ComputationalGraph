package ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

import Math.Tensor;

public class Softplus implements Function, Serializable {

    /**
     * Computes the Softplus activation function for the given tensor.
     * The Softplus function is defined as ln(1 + exp(x)) and is applied element-wise.
     * @param value The input tensor whose values are to be computed.
     * @return A new tensor with the Softplus activation applied to each element.
     */
    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        for (double val : tensorValues) {
            values.add(Math.log(1.0 + Math.exp(val)));
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Computes the derivative of the Softplus activation function.
     * @param value The output tensor of the Softplus activation.
     * @param backward The backward tensor used for gradient propagation.
     * @return A tensor representing the gradient values for the corresponding input.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (int i = 0; i < tensorValues.size(); i++) {
            double val = tensorValues.get(i);
            double backwardValue = backwardValues.get(i);
            values.add((1 - Math.exp(-val)) * backwardValue);
        }
        return new Tensor(values, value.getShape());
    }
}
