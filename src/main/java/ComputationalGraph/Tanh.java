package ComputationalGraph;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Tanh implements Function, Serializable {

    /**
     * Computes the Tanh activation for the given tensor.
     */
    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) tensor.getData();
        for (Double oldValue : oldValues) {
            values.add(Math.tanh(oldValue));
        }
        return new Tensor(values, tensor.getShape());
    }

    /**
     * Computes the derivative of the Tanh function.
     * Assumes input is tanh(x), not raw x.
     */
    @Override
    public Tensor derivative(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) tensor.getData();
        for (Double oldValue : oldValues) {
            values.add(1 - oldValue * oldValue);
        }
        return new Tensor(values, tensor.getShape());
    }
}
