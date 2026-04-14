package ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import Math.Tensor;

public class DivideByDimensionSize implements FunctionCalculator, Serializable {

    private final int dimension;

    public DivideByDimensionSize(int dimension) {
        this.dimension = dimension;
    }

    /**
     * Performs a calculation where each element of the input tensor is divided
     * by the size of the specified dimension within the tensor's shape.
     * @param value The input tensor to be processed.
     * @return A new tensor where each element is divided by the size of the specified dimension.
     */
    @Override
    public Tensor calculate(Tensor value) {
        if (dimension == -1) {
            return value;
        }
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) value.getData();
        int size = value.getShape()[dimension];
        for (double val : tensorValues) {
            values.add((1.0 / size) * (val));
        }
        return new Tensor(values, value.getShape());
    }

    /**
     * Computes the derivative of the function where each element of the tensor is
     * divided by the size of the specified dimension, propagating the gradients
     * during backpropagation.
     * @param value The input tensor that represents the forward pass output.
     * @param backward The backward tensor containing the gradients from the subsequent layers.
     * @return A tensor containing the computed gradients for the current operation.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        if (dimension == -1) {
            return backward;
        }
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> backwardValues = (ArrayList<Double>) backward.getData();
        int size = value.getShape()[dimension];
        for (Double backwardValue : backwardValues) {
            double derivative = (1.0 / size);
            values.add(derivative * backwardValue);
        }
        return new Tensor(values, value.getShape());
    }

    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode newNode = new FunctionNode(isBiased, this);
        inputNode.add(newNode);
        return newNode;
    }
}
