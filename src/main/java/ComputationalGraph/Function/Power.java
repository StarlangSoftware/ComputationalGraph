package ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import Math.Tensor;

public class Power implements Function, Serializable {

    private final int n;

    public Power(int n) {
        this.n = n;
    }

    public Power() {
        this.n = 2;
    }

    /**
     * Computes the Power of the given tensor.
     * @param value The tensor whose values are to be computed.
     * @return pow(x).
     */
    @Override
    public Tensor calculate(Tensor value) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        for (double val : tensorValues) {
            values.add(Math.pow(val, n));
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Computes the derivative of the Power function.
     * @param value output of the Power(x).
     * @param backward Backward tensor.
     * @return Gradient value of the corresponding node.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        for (int i = 0; i < tensorValues.size(); i++) {
            double val = tensorValues.get(i);
            double derivative = n * Math.pow(Math.pow(val, 1.0 / n), n - 1);
            double backwardValue = backwardValues.get(i);
            values.add(derivative * backwardValue);
        }
        return new Tensor(values, value.getShape());
    }

    @Override
    public ComputationalNode addEdge(ComputationalNode input, boolean isBiased) {
        ComputationalNode newNode = new FunctionNode(isBiased, this);
        input.addChild(newNode);
        newNode.addParent(input);
        return newNode;
    }
}
