package ComputationalGraph.Optimizer;

import java.io.Serializable;
import java.util.*;

import ComputationalGraph.Node.ComputationalNode;
import Math.Tensor;

public abstract class Optimizer implements Serializable {

    protected double learningRate;
    private final double etaDecrease;

    public Optimizer(double learningRate, double etaDecrease) {
        this.learningRate = learningRate;
        this.etaDecrease = etaDecrease;
    }

    /**
     * Updates the learning rate of the optimizer.
     */
    public void setLearningRate() {
        this.learningRate *= this.etaDecrease;
    }

    /**
     * Checks if broadcasting be applied to the corresponding node.
     * @param node The node to check.
     * @return The index of the dimension where broadcasting is to be applied. -1 if broadcasting is not to be applied.
     */
    private int broadcast(ComputationalNode node) {
        int[] v = node.getValue().getShape();
        int[] b = node.getBackward().getShape();
        int index = -1;
        for (int i = 0; i < v.length; i++) {
            if (v[i] != b[i]) {
                if (v[i] == 1) {
                    if (index != -1) {
                        return -1;
                    }
                    index = i;
                } else {
                    throw new IllegalArgumentException("Value and Backward shapes are not compatible");
                }
            }
        }
        return index;
    }

    /**
     * Recursive helper function to update the values of learnable nodes.
     * @param visited A set of visited nodes.
     * @param node The current node being processed.
     * @param nodeMap A map of nodes to their children.
     */
    private void updateRecursive(HashSet<ComputationalNode> visited, ComputationalNode node, HashMap<ComputationalNode, ArrayList<ComputationalNode>> nodeMap) {
        visited.add(node);
        if (node.isLearnable()) {
            int index = broadcast(node);
            if (index != -1) {
                int v = 1, b = 1;
                for (int i = node.getValue().getShape().length - 1; i >= index; i--) {
                    v *= node.getValue().getShape()[i];
                    b *= node.getBackward().getShape()[i];
                }
                ArrayList<Double> backwardValues = (ArrayList<Double>) node.getBackward().getData();
                double[] values = new double[node.getValue().getData().size()];
                for (int i = 0; i < backwardValues.size(); i++) {
                    for (int j = i; j < i + b; j++) {
                        values[((j - i) % v) + v * (j / b)] += backwardValues.get(j);
                    }
                    i += b - 1;
                }
                ArrayList<Double> list = new ArrayList<>();
                for (double d : values) {
                    list.add(d);
                }
                node.setBackward(new Tensor(list, node.getValue().getShape()));
            }
            this.setGradients(node);
            node.updateValue();
        }
        if (nodeMap.containsKey(node)) {
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!visited.contains(child)) {
                    updateRecursive(visited, child, nodeMap);
                }
            }
        }
    }

    /**
     * Sets the gradients (backward values) of the node.
     * @param node The node whose gradients are to be set.
     */
    protected abstract void setGradients(ComputationalNode node);

    /**
     * Updates the values of all learnable nodes in the graph.
     * @param nodeMap A map of nodes to their children.
     */
    public void updateValues(HashMap<ComputationalNode, ArrayList<ComputationalNode>> nodeMap) {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                updateRecursive(visited, node, nodeMap);
            }
        }
    }
}
