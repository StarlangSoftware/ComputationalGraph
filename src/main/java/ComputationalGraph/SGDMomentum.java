package ComputationalGraph;

import java.io.Serializable;
import java.util.*;
import Math.Tensor;

public class SGDMomentum extends Optimizer implements Serializable {

    private final HashMap<ComputationalNode, ArrayList<Double>> velocityMap;
    private final double momentum;

    public SGDMomentum(double learningRate, double etaDecrease, double momentum) {
        super(learningRate, etaDecrease);
        this.velocityMap = new HashMap<>();
        this.momentum = momentum;
    }

    @Override
    protected void setGradients(ComputationalNode node) {
        ArrayList<Double> newValues = (ArrayList<Double>) node.getBackward().getData();
        newValues.replaceAll(value -> (1 - momentum) * value);
        if (velocityMap.containsKey(node)) {
            ArrayList<Double> oldVelocity = velocityMap.get(node);
            oldVelocity.replaceAll(value -> momentum * value);
            for (int i = 0; i < newValues.size(); i++) {
                newValues.set(i, newValues.get(i) + oldVelocity.get(i));
            }
        }
        velocityMap.put(node, (ArrayList<Double>) newValues.clone());
        newValues.replaceAll(value -> value * learningRate);
        node.setBackward(new Tensor(newValues, node.getBackward().getShape()));
    }
}
