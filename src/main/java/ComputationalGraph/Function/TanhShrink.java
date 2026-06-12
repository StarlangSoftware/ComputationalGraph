package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;

import java.io.Serializable;

public class TanhShrink implements FunctionCombiner, Serializable {

    /**
     * Adds computational edges to the given input node, creating a series of linked nodes
     * that combine the Negation, Tanh, and Addition operations. This ultimately
     * constructs the TanhShrink activation function.
     * @param inputNode The source computational node to which the TanhShrink activation function is added.
     * @param isBiased Determines whether the resulting computational node is biased.
     * @return The final ComputationalNode representing TanhShrink activation function.
     */
    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode tanh = new FunctionNode(false, new Tanh());
        inputNode.add(tanh);
        ComputationalNode negativeTanh = new FunctionNode(false, new Negation());
        tanh.add(negativeTanh);
        ComputationalNode tanhShrink = new ComputationalNode(false, isBiased);
        inputNode.add(tanhShrink);
        negativeTanh.add(tanhShrink);
        return tanhShrink;
    }
}
