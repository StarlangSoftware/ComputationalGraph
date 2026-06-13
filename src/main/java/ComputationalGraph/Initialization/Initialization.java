package ComputationalGraph.Initialization;

import java.util.Random;
import Math.Tensor;

@FunctionalInterface
public interface Initialization {
    Tensor initialize(int[] shape, Random random);
}
