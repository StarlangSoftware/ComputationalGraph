package ComputationalGraph.Optimizer;

import ComputationalGraph.Node.ComputationalNode;

import java.io.Serializable;
import java.util.*;
import Math.Tensor;

public class Adam extends SGDMomentum implements Serializable {

    private final HashMap<ComputationalNode, ArrayList<Double>> momentumMap;
    private final double beta2;
    private final double epsilon;

    public Adam(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon) {
        super(learningRate, etaDecrease, beta1);
        this.momentumMap = new HashMap<>();
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }
    @Override
    protected void setGradients(ComputationalNode node) {
        ArrayList<Double> newValuesMomentum = (ArrayList<Double>) node.getBackward().getData();
        ArrayList<Double> newValuesVelocity = new ArrayList<>();
        for (Double value : newValuesMomentum) {
            newValuesVelocity.add(value * value);
        }
        newValuesMomentum.replaceAll(value -> (1 - momentum) * value);
        newValuesVelocity.replaceAll(value -> (1 - beta2) * value);
        if (momentumMap.containsKey(node)) {
            ArrayList<Double> oldVelocity = velocityMap.get(node);
            oldVelocity.replaceAll(value -> beta2 * value);
            ArrayList<Double> oldMomentum = momentumMap.get(node);
            oldMomentum.replaceAll(value -> momentum * value);
            for (int i = 0; i < newValuesVelocity.size(); i++) {
                newValuesVelocity.set(i, newValuesVelocity.get(i) + oldVelocity.get(i));
                newValuesMomentum.set(i, newValuesMomentum.get(i) + oldMomentum.get(i));
            }
        }
        momentumMap.put(node, (ArrayList<Double>) newValuesMomentum.clone());
        velocityMap.put(node, (ArrayList<Double>) newValuesVelocity.clone());
        newValuesMomentum.replaceAll(value -> value / (1 - momentum));
        newValuesVelocity.replaceAll(value -> value / (1 - beta2));
        ArrayList<Double> newValues = new ArrayList<>();
        for (int i = 0; i < newValuesMomentum.size(); i++) {
            newValues.add((newValuesMomentum.get(i) / (Math.sqrt(newValuesVelocity.get(i)) + epsilon)) * learningRate);
        }
        node.setBackward(new Tensor(newValues, node.getBackward().getShape()));
    }
}
