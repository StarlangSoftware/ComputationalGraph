package ComputationalGraph.Initialization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class UniformXavierInitialization implements Initialization, Serializable {

    /**
     * Xavier Uniform Initialization.
     * <p>
     * This method initializes weights using a uniform distribution within the range
     * [-limit, limit], where the limit is sqrt(6 / (fan_in + fan_out)).
     * This strategy is designed to keep the scale of the gradients roughly the same
     * in all layers and is commonly used with Sigmoid or Tanh activation functions.
     * </p>
     *
     * @param row    The number of rows in the matrix (typically represents fan-out / output size).
     * @param column The number of columns in the matrix (typically represents fan-in / input size).
     * @param random The {@link Random} instance used for generating values.
     * @return An {@link ArrayList} containing the initialized weight values.
     */
    @Override
    public ArrayList<Double> initialize(int row, int column, Random random) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                data.add((2 * random.nextDouble() - 1) * Math.sqrt(6.0 / (row + column)));
            }
        }
        return data;
    }
}
