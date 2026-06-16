package ComputationalGraph.Optimizer;

import java.io.Serializable;
import java.util.*;

import ComputationalGraph.Clipping.GradientClipping;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Scheduler.Scheduler;
import Math.Tensor;

public abstract class Optimizer implements Serializable {

    private double learningRate;
    private final Scheduler scheduler;
    private final GradientClipping gradientClipping;

    public Optimizer(Scheduler scheduler, GradientClipping gradientClipping) {
        this.learningRate = scheduler.getInitialLearningRate();
        this.scheduler = scheduler;
        this.gradientClipping = gradientClipping;
    }

    public Optimizer(Scheduler scheduler) {
        this.learningRate = scheduler.getInitialLearningRate();
        this.scheduler = scheduler;
        this.gradientClipping = null;
    }

    /**
     * Updates the learning rate of the optimizer.
     */
    public void setLearningRate() {
        this.learningRate = this.scheduler.updateLearningRate();
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
     */
    private void updateRecursive(HashSet<ComputationalNode> visited, ComputationalNode node) {
        visited.add(node);
        if (node.isLearnable()) {
            int index = broadcast(node);
            if (index != -1) {
                int v = 1, b = 1;
                for (int i = node.getValue().getShape().length - 1; i >= index; i--) {
                    v *= node.getValue().getShape()[i];
                    b *= node.getBackward().getShape()[i];
                }
                double[] backwardValues = node.getBackward().getData();
                double[] values = new double[node.getValue().getData().length];
                for (int i = 0; i < backwardValues.length; i++) {
                    for (int j = i; j < i + b; j++) {
                        values[((j - i) % v) + v * (j / b)] += backwardValues[j];
                    }
                    i += b - 1;
                }
                node.setBackward(new Tensor(values, node.getValue().getShape()));
            }
            if (this.gradientClipping != null) {
                node.setBackward(this.gradientClipping.clip(node.getBackward()));
            }
            this.setGradients(node);
            node.updateValue();
        }
        for (int t = 0; t < node.childrenSize(); t++) {
            ComputationalNode child = node.getChild(t);
            if (!visited.contains(child)) {
                updateRecursive(visited, child);
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
     * @param leafNodes input nodes of the graph.
     */
    public void updateValues(ArrayList<ComputationalNode> leafNodes) {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : leafNodes) {
            if (!visited.contains(node)) {
                updateRecursive(visited, node);
            }
        }
    }

    protected double getLearningRate() {
        return this.learningRate;
    }
}
