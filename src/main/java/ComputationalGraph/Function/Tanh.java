package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Tanh implements Function, Serializable {

    /**
     * Computes the Tanh activation for the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return Tanh(x).
     */
    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            values.add(Math.tanh(oldValue));
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Computes the derivative of the Tanh activation function.
     * @param value output of the Tanh(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (int i = 0; i < oldValues.size(); i++) {
            Double oldValue = oldValues.get(i);
            Double backwardValue = backwardValues.get(i);
            values.add((1 - oldValue * oldValue) * backwardValue);
        }
        return new Tensor(values, value.getShape());
    }
}
