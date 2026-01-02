package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class ReLU implements Function, Serializable {

    /**
     * Computes the ReLU activation for the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return ReLU(x).
     */
    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            values.add(Math.max(oldValue, 0));
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Computes the derivative of the ReLU activation function.
     * @param value output of the ReLU(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
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
