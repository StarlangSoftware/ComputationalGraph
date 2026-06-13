package ComputationalGraph.Initialization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import Math.Tensor;

public class HeUniformInitialization implements Initialization, Serializable {

    /**
     * Initializes a weight matrix using the He Uniform Initialization strategy.
     * This strategy is designed to maintain a suitable variance of weights in neural networks
     * with ReLU activation functions by scaling the weights according to the input and output
     * dimensions of the layer being initialized.
     * @param shape An array representing the shape of the tensor to be initialized. The second-to-last
     *              element specifies the number of rows, and the last element specifies the number
     *              of columns. Additional dimensions specify the shape of higher-order tensors.
     * @param random An instance of {@code Random} used to generate random values for the weights.
     * @return A {@code Tensor} representing the initialized weight matrix with the specified shape.
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
            data.add(((Math.sqrt(6.0 / column) + Math.sqrt(6.0 / row)) * random.nextDouble()) - Math.sqrt(6.0 / row));
        }
        return new Tensor(data, shape);
    }
}
