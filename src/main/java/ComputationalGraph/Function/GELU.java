package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class GELU implements FunctionCombiner, Serializable {

    @Override
    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        FunctionNode power3 = new FunctionNode(false, new Power(3));
        inputNode.add(power3);
        FunctionNode multiplyBy0_044715 = new FunctionNode(false, new MultiplyByConstant(0.044715));
        power3.add(multiplyBy0_044715);
        ComputationalNode additionNode = new ComputationalNode();
        inputNode.add(additionNode);
        multiplyBy0_044715.add(additionNode);
        FunctionNode multiplyBySqrt2OverPi = new FunctionNode(false, new MultiplyByConstant(Math.sqrt(2.0 / Math.PI)));
        additionNode.add(multiplyBySqrt2OverPi);
        FunctionNode tanh = new FunctionNode(false, new Tanh());
        multiplyBySqrt2OverPi.add(tanh);
        FunctionNode add1 = new FunctionNode(false, new AdditionByConstant(1.0));
        tanh.add(add1);
        ComputationalNode multiplicationNode = new MultiplicationNode(false, false, true);
        inputNode.add(multiplicationNode);
        add1.add(multiplicationNode);
        FunctionNode gelu = new FunctionNode(isBiased, new MultiplyByConstant(0.5));
        multiplicationNode.add(gelu);
        return gelu;
    }
}
