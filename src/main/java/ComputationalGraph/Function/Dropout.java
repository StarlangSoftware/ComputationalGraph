package ComputationalGraph.Function;

import Math.Tensor;

import java.io.Serializable;
import java.util.Random;

public class Dropout implements Function, Serializable {

    private final double p;
    private final Random random;

    public Dropout(double p, Random random) {
        this.p = p;
        this.random = random;
    }

    /**
     * Computes the dropout values for the given value tensor.
     * @param value The tensor whose values are to be computed.
     * @return Dropout(x) and mask.
     */
    @Override
    public FunctionResults calculate(Tensor value) {
        double multiplier = 1.0 / (1 - p);
        double[] oldValues = value.getData();
        double[] values = new double[oldValues.length];
        double[] mask = new double[oldValues.length];
        for (int i = 0; i < oldValues.length; i++) {
            double oldValue = oldValues[i];
            double r = random.nextDouble();
            if (r > p) {
                mask[i] = multiplier;
                values[i] = oldValue * multiplier;
            } else {
                mask[i] = 0.0;
                values[i] = 0.0;
            }
        }
        return new FunctionResults(new Tensor(values, value.getShape()), new Tensor(mask, value.getShape()));
    }

    /**
     * Calculates the derivative of the dropout.
     * @param value mask of the dropout.
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward.hadamardProduct(value);
    }
}
