package ComputationalGraph.Initialization;

import java.io.Serializable;
import java.util.Random;
import Math.Tensor;

public class RandomInitialization implements Initialization, Serializable {

    /**
     * Initializes a tensor with random values sampled uniformly from the range [-0.01, 0.01].
     * The provided shape determines the dimensions of the tensor.
     * @param shape An array representing the shape of the tensor to be initialized. Each element
     *              corresponds to the size of a dimension in the tensor.
     * @param random An instance of {@code Random} used to generate random values for the tensor entries.
     * @return A {@code Tensor} containing the randomly initialized values with the specified shape.
     */
    @Override
    public Tensor initialize(int[] shape, Random random) {
        int total = 1;
        for (int j : shape) {
            total *= j;
        }
        double[] data = new double[total];
        for (int i = 0; i < total; i++) {
            data[i] = -0.01 + (0.02 * random.nextDouble());
        }
        return new Tensor(data, shape);
    }
}
