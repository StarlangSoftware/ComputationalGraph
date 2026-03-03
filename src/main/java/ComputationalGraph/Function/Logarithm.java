package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;

import java.io.Serializable;
import java.util.ArrayList;
import Math.Tensor;

public class Logarithm implements Function, Serializable {

    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) value.getData();
        for (Double oldValue : oldValues) {
            values.add(Math.log(oldValue));
        }
        return new Tensor(values, value.getShape());
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (int i = 0; i < tensorValues.size(); i++) {
            double val = tensorValues.get(i);
            double derivative = 1.0 / Math.exp(val);
            double backwardValue = backwardValues.get(i);
            values.add(derivative * backwardValue);
        }
        return new Tensor(values, value.getShape());
    }

    @Override
    public ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased) {
        ComputationalNode newNode = new FunctionNode(isBiased, this);
        inputNodes.get(0).add(newNode);
        return newNode;
    }
}
