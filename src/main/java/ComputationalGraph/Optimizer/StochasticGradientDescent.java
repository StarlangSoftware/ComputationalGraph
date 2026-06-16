package ComputationalGraph.Optimizer;

import java.io.Serializable;
import java.util.ArrayList;

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
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> backward = (ArrayList<Double>) node.getBackward().getData();
        for (Double aDouble : backward) {
            values.add(aDouble * getLearningRate());
        }
        node.setBackward(new Tensor(values, node.getBackward().getShape()));
    }
}
