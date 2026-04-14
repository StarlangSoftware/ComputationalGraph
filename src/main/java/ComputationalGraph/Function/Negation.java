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
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (Double backwardValue : backwardValues) {
            values.add(-backwardValue);
        }
        return new Tensor(values, value.getShape());
    }
}
