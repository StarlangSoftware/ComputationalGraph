package ComputationalGraph.Optimizer;

import ComputationalGraph.Node.ComputationalNode;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class AdamW extends Adam implements Serializable {

    private final double weightDecay;

    public AdamW(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon, double weightDecay) {
        super(learningRate, etaDecrease, beta1, beta2, epsilon);
        this.weightDecay = weightDecay;
    }

    /**
     * Sets the gradients for the given node using the AdamW optimization algorithm.
     * @param node The node whose gradients are to be set.
     */
    @Override
    protected void setGradients(ComputationalNode node) {
        ArrayList<Double> gradients = calculate(node);
        ArrayList<Double> values = (ArrayList<Double>) node.getValue().getData();
        for (int i = 0; i < gradients.size(); i++) {
            gradients.set(i, gradients.get(i) + (learningRate * weightDecay * values.get(i)));
        }
        node.setBackward(new Tensor(gradients, node.getBackward().getShape()));
    }
}
