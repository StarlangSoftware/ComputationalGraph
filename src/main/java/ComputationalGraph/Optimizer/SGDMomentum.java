package ComputationalGraph.Optimizer;

import java.io.Serializable;
import java.util.*;

import ComputationalGraph.Node.ComputationalNode;
import Math.Tensor;

public class SGDMomentum extends Optimizer implements Serializable {

    protected final HashMap<ComputationalNode, double[]> velocityMap;
    protected final double momentum;

    public SGDMomentum(double learningRate, double etaDecrease, double momentum) {
        super(learningRate, etaDecrease);
        this.velocityMap = new HashMap<>();
        this.momentum = momentum;
    }

    /**
     * Calculates the new gradients by combining the current gradient with the previous velocity.
     * It updates the internal velocity state and modifies the node's backward tensor
     * to reflect the momentum-adjusted update step.
     *
     * @param node The node whose gradients are to be set.
     */
    @Override
    protected void setGradients(ComputationalNode node) {
        int backwardSize = node.getBackward().getData().size();
        ArrayList<Double> newValues = new ArrayList<>(backwardSize);
        for (int i = 0; i < backwardSize; i++) {
            newValues.add((1 - momentum) * node.getBackward().getData().get(i));
        }
        if (velocityMap.containsKey(node)) {
            for (int i = 0; i < newValues.size(); i++) {
                newValues.set(i, newValues.get(i) + (velocityMap.get(node)[i] * momentum));
            }
        }
        double[] velocity = newValues.stream().mapToDouble(Double::doubleValue).toArray();
        velocityMap.put(node, velocity);
        newValues.replaceAll(value -> value * learningRate);
        node.setBackward(new Tensor(newValues, node.getBackward().getShape()));
    }
}
