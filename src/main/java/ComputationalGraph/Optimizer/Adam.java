package ComputationalGraph.Optimizer;

import ComputationalGraph.Node.ComputationalNode;

import java.io.Serializable;
import java.util.*;
import Math.Tensor;

public class Adam extends SGDMomentum implements Serializable {

    private final HashMap<ComputationalNode, double[]> momentumMap;
    private final double beta2;
    private final double epsilon;
    private double currentBeta1;
    private double currentBeta2;

    public Adam(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon) {
        super(learningRate, etaDecrease, beta1);
        this.momentumMap = new HashMap<>();
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.currentBeta1 = 1;
        this.currentBeta2 = 1;
    }

    /**
     * Calculates the gradient updates using the Adam optimization algorithm.
     * <p>
     * This implementation follows a multi-pass approach:
     * <ol>
     * <li><b>First Pass:</b> Calculates the weighted current gradients for both the first moment (momentum)
     * and the second moment (velocity/squared gradients).</li>
     * <li><b>Second Pass (Conditional):</b> If historical data exists, adds the decayed previous
     * momentum and velocity values to the current ones.</li>
     * <li><b>State Update:</b> Stores the raw calculated moments into the history maps.</li>
     * <li><b>Bias Correction:</b> Normalizes the moments by dividing them by <code>(1 - beta)</code>
     * to account for initialization bias.</li>
     * <li><b>Final Pass:</b> Computes the parameter update using the adaptive learning rate formula:
     * <code>(new_momentum / (sqrt(new_velocity) + epsilon)) * learningRate</code>.</li>
     * </ol>
     * </p>
     *
     * @param node The node whose gradients are to be set.
     */
    protected ArrayList<Double> calculate(ComputationalNode node) {
        int backwardSize = node.getBackward().getData().size();
        ArrayList<Double> newValuesMomentum = new ArrayList<>(backwardSize);
        ArrayList<Double> newValuesVelocity = new ArrayList<>(backwardSize);
        for (int i = 0; i < backwardSize; i++) {
            double backwardValue = node.getBackward().getData().get(i);
            newValuesMomentum.add((1 - momentum) * backwardValue);
            newValuesVelocity.add((1 - beta2) * (backwardValue * backwardValue));
        }
        if (momentumMap.containsKey(node)) {
            for (int i = 0; i < newValuesVelocity.size(); i++) {
                newValuesVelocity.set(i, newValuesVelocity.get(i) + beta2 * velocityMap.get(node)[i]);
                newValuesMomentum.set(i, newValuesMomentum.get(i) + momentum * momentumMap.get(node)[i]);
            }
        }
        double[] momentumValues = new double[backwardSize];
        double[] velocityValues = new double[backwardSize];
        for (int i = 0; i < backwardSize; i++) {
            momentumValues[i] = newValuesMomentum.get(i);
            velocityValues[i] = newValuesVelocity.get(i);
        }
        momentumMap.put(node, momentumValues);
        velocityMap.put(node, velocityValues);
        newValuesMomentum.replaceAll(value -> value / (1 - this.currentBeta1));
        newValuesVelocity.replaceAll(value -> value / (1 - this.currentBeta2));
        ArrayList<Double> newValues = new ArrayList<>(newValuesMomentum.size());
        for (int i = 0; i < newValuesMomentum.size(); i++) {
            newValues.add((newValuesMomentum.get(i) / (Math.sqrt(newValuesVelocity.get(i)) + epsilon)) * learningRate);
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
     * Updates the values of all learnable nodes in the graph.
     * @param nodeMap A map of nodes to their children.
     */
    public void updateValues(HashMap<ComputationalNode, ArrayList<ComputationalNode>> nodeMap) {
        this.currentBeta1 *= momentum;
        this.currentBeta2 *= beta2;
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                updateRecursive(visited, node, nodeMap);
            }
        }
    }
}
