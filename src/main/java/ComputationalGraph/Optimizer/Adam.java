package ComputationalGraph.Optimizer;

import ComputationalGraph.Node.ComputationalNode;

import java.io.Serializable;
import java.util.*;
import Math.Tensor;

public class Adam extends SGDMomentum implements Serializable {

    private final HashMap<ComputationalNode, ArrayList<Double>> momentumMap;
    private final double beta2;
    private final double epsilon;

    public Adam(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon) {
        super(learningRate, etaDecrease, beta1);
        this.momentumMap = new HashMap<>();
        this.beta2 = beta2;
        this.epsilon = epsilon;
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
    @Override
    protected void setGradients(ComputationalNode node) {
        ArrayList<Double> newValuesMomentum = new ArrayList<>();
        ArrayList<Double> newValuesVelocity = new ArrayList<>();
        for (int i = 0; i < node.getBackward().getData().size(); i++) {
            newValuesMomentum.add((1 - momentum) * node.getBackward().getData().get(i));
            newValuesVelocity.add((1 - beta2) * (node.getBackward().getData().get(i) * node.getBackward().getData().get(i)));
        }
        if (momentumMap.containsKey(node)) {
            for (int i = 0; i < newValuesVelocity.size(); i++) {
                newValuesVelocity.set(i, newValuesVelocity.get(i) + beta2 * velocityMap.get(node).get(i));
                newValuesMomentum.set(i, newValuesMomentum.get(i) + momentum * momentumMap.get(node).get(i));
            }
        }
        momentumMap.put(node, (ArrayList<Double>) newValuesMomentum.clone());
        velocityMap.put(node, (ArrayList<Double>) newValuesVelocity.clone());
        newValuesMomentum.replaceAll(value -> value / (1 - momentum));
        newValuesVelocity.replaceAll(value -> value / (1 - beta2));
        ArrayList<Double> newValues = new ArrayList<>();
        for (int i = 0; i < newValuesMomentum.size(); i++) {
            newValues.add((newValuesMomentum.get(i) / (Math.sqrt(newValuesVelocity.get(i)) + epsilon)) * learningRate);
        }
        node.setBackward(new Tensor(newValues, node.getBackward().getShape()));
    }
}
