package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;

@FunctionalInterface
public interface FunctionCombiner {
    ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased);
}
