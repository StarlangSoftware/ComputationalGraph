package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class Mish implements FunctionCombiner, Serializable {

    /**
     * Adds computational edges to the given input node, creating a series of linked nodes
     * that combine the Softplus, Tanh, and Multiplication operations. This ultimately
     * constructs the Mish activation function.
     * @param inputNode The source computational node to which the Mish activation function is added.
     * @param isBiased Determines whether the resulting multiplication node is biased.
     * @return The final ComputationalNode representing the Mish activation function.
     */
    @Override
    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode softplus = new FunctionNode(false, new Softplus());
        inputNode.add(softplus);
        ComputationalNode tanh = new FunctionNode(false, new Tanh());
        softplus.add(tanh);
        ComputationalNode mish = new MultiplicationNode(false, isBiased, true);
        inputNode.add(mish);
        tanh.add(mish);
        return mish;
    }
}
