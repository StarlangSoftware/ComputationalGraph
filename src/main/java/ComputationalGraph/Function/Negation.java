package ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

import Math.*;

public class Negation implements Function, Serializable {

    /**
     * Negates the values of the given tensor.
     * @param value The tensor whose values are to be negated.
     * @return The negated tensor.
     */
    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            values.add(-oldValue);
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Calculates the derivative of the Negation function.
     * @param value output of the Negation function.
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        int size = 1;
        for (int i = 0; i < value.getShape().length; i++) {
            size *= value.getShape()[i];
        }
        for (int i = 0; i < size; i++) {
            values.add(-1.0);
        }
        return backward.hadamardProduct(new Tensor(values, value.getShape()));
    }
}
