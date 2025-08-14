package ComputationalGraph;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Sigmoid implements Function, Serializable {

    /**
     * Computes the Sigmoid activation for the given tensor.
     */
    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) tensor.getData();
        for (double val : tensorValues) {
            double sigmoid = 1.0 / (1.0 + Math.exp(-val));
            values.add(sigmoid);
        }
        return new Tensor(values, tensor.getShape());
    }

    /**
     * Computes the derivative of the Sigmoid function.
     * Assumes `tensor` is the output of sigmoid(x).
     */
    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) tensor.getData();
        for (double val : tensorValues) {
            double derivative = val * (1 - val);
            values.add(derivative);
        }
        return backward.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
