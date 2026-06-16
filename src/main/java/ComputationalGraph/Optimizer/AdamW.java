package ComputationalGraph.Optimizer;

import ComputationalGraph.Clipping.GradientClipping;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Scheduler.Scheduler;
import Math.Tensor;

import java.io.Serializable;

public class AdamW extends Adam implements Serializable {

    private final double weightDecay;

    public AdamW(Scheduler scheduler, double beta1, double beta2, double epsilon, double weightDecay) {
        super(scheduler, beta1, beta2, epsilon);
        this.weightDecay = weightDecay;
    }

    public AdamW(Scheduler scheduler, double beta1, double beta2, double epsilon, double weightDecay, GradientClipping gradientClipping) {
        super(scheduler, beta1, beta2, epsilon, gradientClipping);
        this.weightDecay = weightDecay;
    }

    /**
     * Sets the gradients for the given node using the AdamW optimization algorithm.
     * @param node The node whose gradients are to be set.
     */
    @Override
    protected void setGradients(ComputationalNode node) {
        double[] gradients = calculate(node);
        double[] values = node.getValue().getData();
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] = gradients[i] + (getLearningRate() * weightDecay * values[i]);
        }
        node.setBackward(new Tensor(gradients, node.getBackward().getShape()));
    }
}
