package ComputationalGraph.Clipping;

import java.io.Serializable;
import java.util.ArrayList;

import Math.Tensor;

public class ClippingByNorm extends GradientClipping implements Serializable {

    public ClippingByNorm(double factor) {
        super(factor);
    }

    /**
     * Clips the gradient values of the given tensor based on the norm of its elements.
     * If the norm of the gradient values exceeds a predefined threshold (factor),
     * the values are scaled to limit the norm to the threshold. Otherwise, the original gradient values are returned.
     * @param backward The tensor representing the gradients to be clipped. It contains gradient values as a list of doubles and the corresponding tensor shape.
     * @return A tensor with clipped gradient values if the norm exceeded the threshold, or the original tensor if not.
     */
    @Override
    public Tensor clip(Tensor backward) {
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        double norm = 0.0;
        for (Double backwardValue : backwardValues) {
            norm += Math.pow(backwardValue, 2);
        }
        norm = Math.sqrt(norm);
        double factor = getFactor();
        if (norm > factor) {
            ArrayList<Double> gradient = new ArrayList<>(backwardValues.size());
            for (Double backwardValue : backwardValues) {
                gradient.add((backwardValue / norm) * factor);
            }
            return new Tensor(gradient, backward.getShape());
        }
        return backward;
    }
}
