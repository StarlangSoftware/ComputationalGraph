package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class SiLU implements FunctionCombiner, Serializable {

    private final double beta;

    public SiLU(double beta) {
        this.beta = beta;
    }

    public SiLU() {
        this(1.0);
    }

    /**
     * Adds edges to the computational graph by combining the input node with
     * other computational nodes to implement a SiLU activation function.
     * The method creates and connects intermediate nodes such as a sigmoid function node
     * and a multiplication node to the input node.
     * @param inputNode The input computational node to which the edge is added.
     * @param isBiased Indicates whether the resulting multiplication node is biased.
     * @return The computational node representing the resulting SiLU activation function.
     */
    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode sigmoid = new FunctionNode(false, new Sigmoid());
        if (beta != 1.0) {
            ComputationalNode multiplyWithBeta = new FunctionNode(false, new MultiplyByConstant(beta));
            inputNode.add(multiplyWithBeta);
            multiplyWithBeta.add(sigmoid);
        } else {
            inputNode.add(sigmoid);
        }
        ComputationalNode silu = new MultiplicationNode(false, isBiased, true);
        sigmoid.add(silu);
        inputNode.add(silu);
        return silu;
    }
}
