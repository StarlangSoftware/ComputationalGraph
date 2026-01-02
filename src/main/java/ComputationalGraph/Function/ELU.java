package ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

import Math.Tensor;

public class ELU implements Function, Serializable {

    private final double a;

    public ELU(double a) {
        this.a = a;
    }

    public ELU() {
        this.a = 1.0;
    }

    /**
     * Computes the ELU activation for the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return ELU(x).
     */
    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            if (oldValue < 0) {
                values.add(a * (Math.exp(oldValue) - 1));
            } else {
                values.add(oldValue);
            }
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Computes the derivative of the ELU activation function.
     * @param value output of the ELU(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            if (oldValue < 0) {
                values.add(oldValue + a);
            } else {
                values.add(1.0);
            }
        }
        return backward.hadamardProduct(new Tensor(values, value.getShape()));
    }
}
