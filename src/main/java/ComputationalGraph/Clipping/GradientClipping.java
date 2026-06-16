package ComputationalGraph.Clipping;

import java.io.Serializable;
import Math.Tensor;

public abstract class GradientClipping implements Serializable {

    private final double factor;

    public GradientClipping(double factor) {
        if (factor <= 0) {
            throw new IllegalArgumentException("Factor must be positive");
        }
        this.factor = factor;
    }

    public double getFactor() {
        return factor;
    }

    public abstract Tensor clip(Tensor backward);
}
