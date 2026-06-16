package ComputationalGraph.Optimizer;

import java.io.Serializable;

import ComputationalGraph.Clipping.GradientClipping;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Scheduler.Scheduler;
import Math.Tensor;

public class StochasticGradientDescent extends Optimizer implements Serializable {

    public StochasticGradientDescent(Scheduler scheduler) {
        super(scheduler);
    }

    public StochasticGradientDescent(Scheduler scheduler, GradientClipping gradientClipping) {
        super(scheduler, gradientClipping);
    }

    /**
     * Sets the gradients (backward values) of the node to the learning rate times the backward values.
     * @param node The node whose gradients are to be set.
     */
    @Override
    protected void setGradients(ComputationalNode node) {
        double[] backward = node.getBackward().getData();
        double[] values = new double[backward.length];
        for (int i = 0; i < backward.length; i++) {
            values[i] = backward[i] * getLearningRate();
        }
        node.setBackward(new Tensor(values, node.getBackward().getShape()));
    }
}
