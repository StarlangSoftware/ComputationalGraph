package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class ReLU implements Function, Serializable {

    /**
     * Computes the ReLU activation for the given tensor.
     */
    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) tensor.getData();
        for (Double oldValue : oldValues) {
            values.add(Math.max(oldValue, 0));
        }
        return new Tensor(values, tensor.getShape());
    }

    /**
     * Computes the derivative of the ReLU function.
     * Assumes input is the raw pre-activation tensor.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            if (oldValue > 0) {
                values.add(1.0);
            } else {
                values.add(0.0);
            }
        }
        return backward.hadamardProduct(new Tensor(values, value.getShape()));
    }
}
