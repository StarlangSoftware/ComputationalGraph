package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class Mish implements CompositeFunction, Serializable {

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
