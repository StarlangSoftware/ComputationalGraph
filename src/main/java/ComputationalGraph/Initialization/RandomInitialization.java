package ComputationalGraph.Initialization;

import java.io.Serializable;
import java.util.*;

public class RandomInitialization implements Initialization, Serializable {

    @Override
    public ArrayList<Double> initialize(int row, int column, Random random) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < row * column; i++) {
            data.add(-0.01 + (0.02 * random.nextDouble()));
        }
        return data;
    }
}
