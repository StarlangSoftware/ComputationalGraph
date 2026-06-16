package ComputationalGraph.Clipping;

import java.io.Serializable;

import Math.Tensor;

public class ClippingByValue extends GradientClipping implements Serializable {

    public ClippingByValue(double factor) {
        super(factor);
    }

    /**
     * Clips the gradient values of the given tensor element-wise based on a predefined threshold (factor).
     * Each gradient value is constrained within the range [-factor, factor].
     * @param backward The tensor representing the gradients to be clipped. It contains gradient values as a list of doubles and the corresponding tensor shape.
     * @return A tensor with each gradient value clipped to the range [-factor, factor].
     */
    @Override
    public Tensor clip(Tensor backward) {
        double[] backwardValues = backward.getData();
        double[] gradient = new double[backwardValues.length];
        double factor = getFactor();
        for (int i = 0; i < backwardValues.length; i++) {
            double clippedValue = Math.max(-factor, Math.min(factor, backwardValues[i]));
            gradient[i] = clippedValue;
        }
        return new Tensor(gradient, backward.getShape());
    }
}
