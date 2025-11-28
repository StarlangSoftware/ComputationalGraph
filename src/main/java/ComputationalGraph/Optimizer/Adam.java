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
        ArrayList<Double> newValuesMomentum = new ArrayList<>();
        ArrayList<Double> newValuesVelocity = new ArrayList<>();
        for (int i = 0; i < node.getBackward().getData().size(); i++) {
            newValuesMomentum.add((1 - momentum) * node.getBackward().getData().get(i));
            newValuesVelocity.add((1 - beta2) * (node.getBackward().getData().get(i) * node.getBackward().getData().get(i)));
        }
        if (momentumMap.containsKey(node)) {
            for (int i = 0; i < newValuesVelocity.size(); i++) {
                newValuesVelocity.set(i, newValuesVelocity.get(i) + beta2 * velocityMap.get(node).get(i));
                newValuesMomentum.set(i, newValuesMomentum.get(i) + momentum * momentumMap.get(node).get(i));
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
