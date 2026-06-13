package ComputationalGraph.Initialization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import Math.Tensor;

public class UniformXavierInitialization implements Initialization, Serializable {

    /**
     * Initializes a tensor using the Uniform Xavier Initialization strategy.
     * This method generates random values uniformly sampled from a range determined
     * by the input and output dimensions of the layer being initialized. It is
     * particularly suited for neural networks with sigmoid or hyperbolic tangent
     * activation functions.
     * @param shape An array representing the shape of the tensor to be initialized.
     *              The second-to-last element specifies the number of rows,
     *              and the last element specifies the number of columns. Additional
     *              dimensions specify the shape of higher-order tensors.
     * @param random An instance of {@code Random} used to generate random values
     *               for the tensor entries.
     * @return A {@code Tensor} containing the randomly initialized values with
     *         the specified shape based on the Uniform Xavier Initialization strategy.
     */
    @Override
    public Tensor initialize(int[] shape, Random random) {
        int row = shape[shape.length - 2];
        int column = shape[shape.length - 1];
        int total = 1;
        for (int j : shape) {
            total *= j;
        }
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < total; i++) {
            data.add((2 * random.nextDouble() - 1) * Math.sqrt(6.0 / (row + column)));
        }
        return new Tensor(data, shape);
    }
}
