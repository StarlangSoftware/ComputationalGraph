package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class Dropout implements Function, Serializable {

    private final double p;
    private final Random random;
    private final ArrayList<Double> mask;

    public Dropout(double p, Random random) {
        this.p = p;
        this.random = random;
        this.mask = new ArrayList<>();
    }

    /**
     * Computes the dropout values for the given value tensor.
     * @param value The tensor whose values are to be computed.
     * @return Output tensor.
     */
    @Override
    public Tensor calculate(Tensor value) {
        this.mask.clear();
        double multiplier = 1.0 / (1 - p);
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
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
        return new Tensor(values, value.getShape());
    }

    /**
     * Calculates the derivative of the dropout.
     * @param value output of the dropout function.
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward.hadamardProduct(new Tensor(mask, value.getShape()));
    }
}
