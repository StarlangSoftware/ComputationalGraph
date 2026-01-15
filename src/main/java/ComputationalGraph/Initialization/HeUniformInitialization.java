package ComputationalGraph.Initialization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class HeUniformInitialization implements Initialization, Serializable {

    /**
     * He Uniform Initialization.
     * <p>
     * This method initializes weights using a uniform distribution, which is typically
     * optimized for layers with ReLU activation functions. It helps in maintaining
     * the variance of activations throughout the network layers.
     * </p>
     *
     * @param row    The number of rows in the matrix (typically represents the output size / number of neurons).
     * @param column The number of columns in the matrix (typically represents the input size / fan-in).
     * @param random The {@link Random} instance used for generating values (allows for reproducibility).
     * @return An {@link ArrayList} of Doubles containing the initialized weight values.
     */
    @Override
    public ArrayList<Double> initialize(int row, int column, Random random) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                data.add(((Math.sqrt(6.0 / column) + Math.sqrt(6.0 / row)) * random.nextDouble()) - Math.sqrt(6.0 / row));
            }
        }
        return data;
    }
}
