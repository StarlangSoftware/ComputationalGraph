package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;

@FunctionalInterface
public interface Function {
    ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased);
}
