package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;

@FunctionalInterface
public interface CompositeFunction {
    ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased);
}
