package ComputationalGraph.Loss;


import ComputationalGraph.Node.ComputationalNode;

@FunctionalInterface
public interface Loss {
    ComputationalNode addEdge(ComputationalNode inputNode, ComputationalNode classLabelNode, int batchDimension);
}
