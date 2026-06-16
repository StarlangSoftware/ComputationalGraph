package ComputationalGraph.Optimizer;

import ComputationalGraph.Clipping.GradientClipping;
import ComputationalGraph.Node.ComputationalNode;

import java.io.Serializable;
import java.util.*;

import ComputationalGraph.Scheduler.Scheduler;
import Math.Tensor;

public class Adam extends SGDMomentum implements Serializable {

    private final HashMap<ComputationalNode, double[]> momentumMap;
    private final double beta2;
    private final double epsilon;
    private double currentBeta1;
    private double currentBeta2;

    public Adam(Scheduler scheduler, double beta1, double beta2, double epsilon) {
        super(scheduler, beta1);
        this.momentumMap = new HashMap<>();
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.currentBeta1 = 1;
        this.currentBeta2 = 1;
    }

    public Adam(Scheduler scheduler, double beta1, double beta2, double epsilon, GradientClipping gradientClipping) {
        super(scheduler, beta1, gradientClipping);
        this.momentumMap = new HashMap<>();
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.currentBeta1 = 1;
        this.currentBeta2 = 1;
    }

    /**
     * Calculates the gradient updates using the Adam optimization algorithm.
     * This implementation follows a multi-pass approach:
     * <ol>
     * <li><b>First Pass:</b> Calculates the weighted current gradients for both the first moment (momentum)
     * and the second moment (velocity/squared gradients).</li>
     * <li><b>Second Pass (Conditional):</b> If historical data exists, adds the decayed previous
     * momentum and velocity values to the current ones.</li>
     * <li><b>State Update:</b> Stores the raw calculated moments into the history maps.</li>
     * <li><b>Bias Correction:</b> Normalizes the moments by dividing them by <code>(1 - (beta)^t)</code>
     * to account for initialization bias.</li>
     * <li><b>Final Pass:</b> Computes the parameter update using the adaptive learning rate formula:
     * <code>(new_momentum / (sqrt(new_velocity) + epsilon)) * learningRate</code>.</li>
     * </ol>
     *
     * @param node The node whose gradients are to be set.
     */
    protected double[] calculate(ComputationalNode node) {
        int backwardSize = node.getBackward().getData().length;
        double[] newValuesMomentum = new double[backwardSize];
        double[] newValuesVelocity = new double[backwardSize];
        for (int i = 0; i < backwardSize; i++) {
            double backwardValue = node.getBackward().getData()[i];
            newValuesMomentum[i] = (1 - momentum) * backwardValue;
            newValuesVelocity[i] = (1 - beta2) * (backwardValue * backwardValue);
        }
        if (momentumMap.containsKey(node)) {
            for (int i = 0; i < newValuesVelocity.length; i++) {
                newValuesVelocity[i] = newValuesVelocity[i] + beta2 * velocityMap.get(node)[i];
                newValuesMomentum[i] = newValuesMomentum[i] + momentum * momentumMap.get(node)[i];
            }
        }
        double[] momentumValues = new double[backwardSize];
        double[] velocityValues = new double[backwardSize];
        for (int i = 0; i < backwardSize; i++) {
            momentumValues[i] = newValuesMomentum[i];
            velocityValues[i] = newValuesVelocity[i];
        }
        momentumMap.put(node, momentumValues);
        velocityMap.put(node, velocityValues);
        for (int i = 0; i < backwardSize; i++) {
            newValuesMomentum[i] /= (1 - this.currentBeta1);
            newValuesVelocity[i] /= (1 - this.currentBeta2);
        }
        double[] newValues = new double[newValuesMomentum.length];
        for (int i = 0; i < newValuesMomentum.length; i++) {
            newValues[i] = (newValuesMomentum[i] / (Math.sqrt(newValuesVelocity[i]) + epsilon)) * getLearningRate();
        }
        return newValues;
    }

    /**
     * Sets the gradients for the given node using the Adam optimization algorithm.
     * @param node The node whose gradients are to be set.
     */
    @Override
    protected void setGradients(ComputationalNode node) {
        node.setBackward(new Tensor(calculate(node), node.getBackward().getShape()));
    }

    /**
     * Updates the values of all learnable nodes and momentum values of the graph.
     * @param leafNodes input nodes of the graph.
     */
    public void updateValues(ArrayList<ComputationalNode> leafNodes) {
        this.currentBeta1 *= momentum;
        this.currentBeta2 *= beta2;
        super.updateValues(leafNodes);
    }
}
