package ComputationalGraph.Optimizer;

import java.io.Serializable;
import java.util.ArrayList;

import ComputationalGraph.Node.ComputationalNode;
import Math.Tensor;

public class StochasticGradientDescent extends Optimizer implements Serializable {

    public StochasticGradientDescent(double learningRate, double etaDecrease) {
        super(learningRate, etaDecrease);
    }

    @Override
    protected void setGradients(ComputationalNode node) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> backward = (ArrayList<Double>) node.getBackward().getData();
        for (Double aDouble : backward) {
            values.add(aDouble * this.learningRate);
        }
        node.setBackward(new Tensor(values, node.getBackward().getShape()));
    }
}
