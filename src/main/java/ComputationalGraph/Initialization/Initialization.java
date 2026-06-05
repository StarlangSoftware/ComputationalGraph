package ComputationalGraph.Initialization;

import java.util.*;

@FunctionalInterface
public interface Initialization {
    ArrayList<Double> initialize(int row, int column, Random random);
}
