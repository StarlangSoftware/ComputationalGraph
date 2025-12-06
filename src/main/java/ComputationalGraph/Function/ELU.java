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
     */
    @Override
    public Tensor calculate(Tensor matrix) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) matrix.getData();
        for (Double oldValue : oldValues) {
            if (oldValue < 0) {
                values.add(a * (Math.exp(oldValue) - 1));
            } else {
                values.add(oldValue);
            }
        }
        return new Tensor(values, matrix.getShape());
    }

    /**
     * Computes the derivative of the ELU activation function.
     * Assumes `matrix` is the output of ELU(x).
     */
    @Override
    public Tensor derivative(Tensor matrix, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) matrix.getData();
        for (Double oldValue : oldValues) {
            if (oldValue < 0) {
                values.add(oldValue + a);
            } else {
                values.add(1.0);
            }
        }
        return backward.hadamardProduct(new Tensor(values, matrix.getShape()));
    }
}
