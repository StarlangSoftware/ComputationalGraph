package ComputationalGraph.Loss;


import ComputationalGraph.Node.ComputationalNode;

@FunctionalInterface
public interface Loss {
    ComputationalNode addLoss(ComputationalNode inputNode, ComputationalNode classLabelNode, int batchDimension);
}
