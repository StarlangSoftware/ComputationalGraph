package ComputationalGraph.Optimizer;

import java.io.Serializable;
import java.util.*;

import ComputationalGraph.Node.ComputationalNode;
import Math.Tensor;

public class SGDMomentum extends Optimizer implements Serializable {

    protected final HashMap<ComputationalNode, ArrayList<Double>> velocityMap;
    protected final double momentum;

    public SGDMomentum(double learningRate, double etaDecrease, double momentum) {
        super(learningRate, etaDecrease);
        this.velocityMap = new HashMap<>();
        this.momentum = momentum;
    }

    @Override
    protected void setGradients(ComputationalNode node) {
        ArrayList<Double> newValues = new ArrayList<>();
        for (int i = 0; i < node.getBackward().getData().size(); i++) {
            newValues.add((1 - momentum) * node.getBackward().getData().get(i));
        }
        if (velocityMap.containsKey(node)) {
            for (int i = 0; i < newValues.size(); i++) {
                newValues.set(i, newValues.get(i) + (velocityMap.get(node).get(i) * momentum));
            }
        }
        velocityMap.put(node, (ArrayList<Double>) newValues.clone());
        newValues.replaceAll(value -> value * learningRate);
        node.setBackward(new Tensor(newValues, node.getBackward().getShape()));
    }
}
