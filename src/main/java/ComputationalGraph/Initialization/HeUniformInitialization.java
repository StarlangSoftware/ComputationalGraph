package ComputationalGraph.Initialization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class HeUniformInitialization implements Initialization, Serializable {

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
