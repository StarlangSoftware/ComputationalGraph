package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class Dropout implements Function, Serializable {

    private final double p;
    private final Random random;
    private ArrayList<Double> mask;

    public Dropout(double p, Random random) {
        this.p = p;
        this.random = random;
    }

    /**
     * Computes the dropout values for the given tensor.
     */
    @Override
    public Tensor calculate(Tensor matrix) {
        this.mask = new ArrayList<>();
        double multiplier = 1.0 / (1 - p);
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) matrix.getData();
        for (Double oldValue : oldValues) {
            double r = random.nextDouble();
            if (r > p) {
                mask.add(multiplier);
                values.add(oldValue * multiplier);
            } else {
                mask.add(0.0);
                values.add(0.0);
            }
        }
        return new Tensor(values, matrix.getShape());
    }

    /**
     * Calculates the derivative of the dropout.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward.hadamardProduct(new Tensor(mask, value.getShape()));
    }
}
