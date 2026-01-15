package ComputationalGraph.Initialization;

import java.io.Serializable;
import java.util.*;

public class RandomInitialization implements Initialization, Serializable {

    /**
     * Random Uniform Initialization.
     * <p>
     * This method initializes the weights with small random values uniformly distributed
     * between -0.01 and 0.01. This is a basic initialization strategy used to break
     * symmetry between neurons.
     * </p>
     *
     * @param row    The number of rows in the matrix.
     * @param column The number of columns in the matrix.
     * @param random The {@link Random} instance used for generating values.
     * @return An {@link ArrayList} containing the initialized weight values.
     */
    @Override
    public ArrayList<Double> initialize(int row, int column, Random random) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < row * column; i++) {
            data.add(-0.01 + (0.02 * random.nextDouble()));
        }
        return data;
    }
}
