package ComputationalGraph;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.Function.*;
import ComputationalGraph.Node.*;
import Math.Tensor;

import java.io.*;
import java.util.*;

public abstract class ComputationalGraph implements Serializable {

    protected ComputationalNode outputNode;
    protected ArrayList<ComputationalNode> inputNodes;
    private ArrayList<ComputationalNode> leafNodes;
    protected final NeuralNetworkParameter parameters;

    public ComputationalGraph(NeuralNetworkParameter parameters) {
        this.inputNodes = new ArrayList<>();
        this.parameters = parameters;
    }

    /**
     * Trains the computational graph using the given training set and parameters.
     * @param trainSet The training set.
     */
    public abstract void train(ArrayList<Tensor> trainSet);

    /**
     * Tests the computational graph on the given test set.
     * @param testSet The test set.
     * @return The classification performance of the computational graph on the test set.
     */
    public abstract ClassificationPerformance test(ArrayList<Tensor> testSet);

    /**
     * Retrieves the output value(s) of the given output node in the computational graph.
     * @return A list of doubles representing the output value(s) of the output node.
     */
    protected abstract ArrayList<Double> getOutputValue();

    /**
     * Randomly shuffles the elements within the provided list of tensors.
     * @param tensors The list of tensors to be shuffled.
     * @param random  The instance of Random used to generate random indices for shuffling.
     */
    protected void shuffle(ArrayList<Tensor> tensors, Random random) {
        int size = tensors.size();
        for (int j = 0; j < size; j++) {
            int i1 = random.nextInt(size);
            int i2 = random.nextInt(size);
            Tensor tmp = tensors.get(i1);
            tensors.set(i1, tensors.get(i2));
            tensors.set(i2, tmp);
        }
    }

    protected ComputationalNode addEdge(ComputationalNode first, Object second, boolean isBiased) {
        if (second instanceof Function) {
            ComputationalNode newNode = new FunctionNode(isBiased, (Function) second);
            first.add(newNode);
            return newNode;
        } else if (second instanceof FunctionCombiner) {
            return ((FunctionCombiner) second).addEdge(first, isBiased);
        } else {
            ComputationalNode newNode;
            if (second instanceof MultiplicationNode) {
                newNode = new MultiplicationNode(false, isBiased, ((MultiplicationNode) second).isHadamard(), first);
            } else {
                throw new IllegalArgumentException("Illegal Type of Object: second");
            }
            first.add(newNode);
            ((ComputationalNode) second).add(newNode);
            return newNode;
        }
    }

    protected ComputationalNode addEdge(ComputationalNode first, Object second) {
        return addEdge(first, second, false);
    }

    protected void addLoss(ComputationalNode classLabelNode) {
        if (outputNode == null) {
            throw new IllegalArgumentException("Output node must be initialized first.");
        }
        ComputationalNode lossNode = parameters.getLossFunction().addLoss(outputNode, classLabelNode, parameters.getBatchDimension());
        this.leafNodes = new ArrayList<>();
        ArrayList<ComputationalNode> queue = new ArrayList<>();
        HashSet<ComputationalNode> visited = new HashSet<>();
        queue.add(lossNode);
        while (!queue.isEmpty()) {
            ComputationalNode currentNode = queue.remove(0);
            if (currentNode.parentsSize() == 0) {
                leafNodes.add(currentNode);
            }
            for (int i = 0; i < currentNode.parentsSize(); i++) {
                ComputationalNode parent = currentNode.getParent(i);
                if (!visited.contains(parent)) {
                    visited.add(parent);
                    queue.add(parent);
                }
            }
        }
    }

    protected ComputationalNode addEdge(ComputationalNode first, ComputationalNode second, boolean isBiased, boolean isHadamard) {
        ComputationalNode newNode = new MultiplicationNode(false, isBiased, isHadamard, first);
        first.add(newNode);
        second.add(newNode);
        return newNode;
    }

    protected ComputationalNode addAdditionEdge(ComputationalNode first, ComputationalNode second, boolean isBiased) {
        ComputationalNode newNode = new ComputationalNode(false, isBiased);
        first.add(newNode);
        second.add(newNode);
        return newNode;
    }

    /**
     * Concatenates the given nodes along the given dimension.
     * @param nodes List of nodes to be concatenated.
     * @param dimension Dimension along which the nodes need to be concatenated.
     * @return A new node that connects to the given nodes.
     */
    protected ComputationalNode concatEdges(ArrayList<ComputationalNode> nodes, int dimension) {
        ConcatenatedNode newNode = new ConcatenatedNode(dimension);
        for (ComputationalNode node : nodes) {
            node.add(newNode);
            newNode.addNode(node);
        }
        return newNode;
    }

    /**
     * Recursive helper function to perform depth-first search for topological sorting.
     * @param node The current node being processed.
     * @param visited A set of visited nodes.
     * @return A list representing the partial topological order.
     */
    private LinkedList<ComputationalNode> sortRecursive(ComputationalNode node, HashSet<ComputationalNode> visited) {
        LinkedList<ComputationalNode> queue = new LinkedList<>();
        visited.add(node);
        for (int i = 0; i < node.childrenSize(); i++) {
            ComputationalNode child = node.getChild(i);
            if (!visited.contains(child)) {
                queue.addAll(sortRecursive(child, visited));
            }
        }
        queue.offer(node);
        return queue;
    }

    /**
     * Performs topological sorting on the computational graph.
     * @return A list representing the topological order of the nodes.
     */
    private LinkedList<ComputationalNode> topologicalSort() {
        LinkedList<ComputationalNode> sortedList = new LinkedList<>();
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : leafNodes) {
            if (!visited.contains(node)) {
                LinkedList<ComputationalNode> queue = sortRecursive(node, visited);
                while (!queue.isEmpty()) {
                    sortedList.offerLast(queue.pollFirst());
                }
            }
        }
        return sortedList;
    }

    /**
     * Recursive helper function to clear the values and gradients of nodes.
     */
    private void clearRecursive(HashSet<ComputationalNode> visited, ComputationalNode node) {
        visited.add(node);
        if (!node.isLearnable()) {
            node.setValue(null);
        }
        node.setBackward(null);
        for (int i = 0; i < node.childrenSize(); i++) {
            ComputationalNode child = node.getChild(i);
            if (!visited.contains(child)) {
                clearRecursive(visited, child);
            }
        }
    }

    /**
     * Clears the values and gradients of all nodes in the graph.
     */
    private void clear() {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : leafNodes) {
            if (!visited.contains(node)) {
                clearRecursive(visited, node);
            }
        }
    }

    /**
     * Swaps the last two dimensions of the Tensor.
     * @param length dimension size.
     */
    private int[] transposeAxes(int length) {
        int[] axes = new int[length];
        for (int i = 0; i < axes.length - 2; i++) {
            axes[i] = i;
        }
        axes[axes.length - 1] = axes.length - 2;
        axes[axes.length - 2] = axes.length - 1;
        return axes;
    }

    /**
     * Removes the bias term from the tensor.
     * @param tensor for which the bias term needs to be removed.
     * @return Tensor without bias term.
     */
    private Tensor getBiasedPartial(Tensor tensor) {
        int[] endIndexes = new int[tensor.getShape().length];
        for (int i = 0; i < endIndexes.length; i++) {
            if (i == endIndexes.length - 1) {
                endIndexes[i] = tensor.getShape()[i] - 1;
            } else {
                endIndexes[i] = tensor.getShape()[i];
            }
        }
        return tensor.partial(new int[tensor.getShape().length], endIndexes);
    }

    /**
     * Calculates the derivative of the child node with respect to the parent node.
     * @param node Parent node.
     * @param child Child node.
     * @return The gradient tensor.
     */
    private Tensor calculateDerivative(ComputationalNode node, ComputationalNode child) {
        if (child.parentsSize() == 0) {
            return null;
        }
        Tensor backward;
        if (child.isBiased()) {
            backward = getBiasedPartial(child.getBackward());
        } else {
            backward = child.getBackward();
        }
        if (child instanceof FunctionNode) {
            Function function = ((FunctionNode) child).getFunction();
            Tensor childValue;
            if (child.isBiased()) {
                childValue = getBiasedPartial(child.getValue());
            } else {
                childValue = child.getValue();
            }
            return function.derivative(childValue, backward);
        } else {
            if (child instanceof ConcatenatedNode) {
                int index = ((ConcatenatedNode) child).getIndex(node);
                int blockSize = backward.getShape()[((ConcatenatedNode) child).getDimension()] / child.parentsSize();
                int dimensions = blockSize;
                int[] shape = new int[backward.getShape().length];
                for (int i = 0; i < backward.getShape().length; i++) {
                    if (((ConcatenatedNode) child).getDimension() > i) {
                        shape[i] = backward.getShape()[i];
                    } else if (((ConcatenatedNode) child).getDimension() < i) {
                        dimensions *= backward.getShape()[i];
                        shape[i] = backward.getShape()[i];
                    } else {
                        shape[i] = blockSize;
                    }
                }
                ArrayList<Double> childValues = (ArrayList<Double>) backward.getData(), newValues = new ArrayList<>();
                int i = index * dimensions;
                while (i < childValues.size()) {
                    for (int k = 0; k < dimensions; k++) {
                        newValues.add(childValues.get(i + k));
                    }
                    i += child.parentsSize() * dimensions;
                }
                return new Tensor(newValues, shape);
            } else {
                if (child instanceof MultiplicationNode) {
                    ComputationalNode left = child.getParent(0);
                    ComputationalNode right = child.getParent(1);
                    if (left == node) {
                        Tensor rightValue = right.getValue();
                        if (((MultiplicationNode) child).isHadamard()) {
                            return rightValue.hadamardProduct(backward);
                        }
                        return backward.multiply(rightValue.transpose(transposeAxes(rightValue.getShape().length)));
                    }
                    Tensor leftValue = left.getValue();
                    if (((MultiplicationNode) child).isHadamard()) {
                        return leftValue.hadamardProduct(backward);
                    }
                    if (leftValue != null && backward != null) {
                        return leftValue.transpose(transposeAxes(leftValue.getShape().length)).multiply(backward);
                    }
                    throw new NullPointerException("Backward and/or left child values are null.");
                }
                return backward;
            }
        }
    }

    /**
     * Performs backpropagation on the computational graph.
     */
    protected void backpropagation() {
        LinkedList<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return;
        ComputationalNode outputNode = sortedNodes.remove(0);
        ArrayList<Double> backward = new ArrayList<>();
        for (int i = 0; i < outputNode.getValue().getData().size(); i++) {
            backward.add(1.0);
        }
        outputNode.setBackward(new Tensor(backward, outputNode.getValue().getShape()));
        while (!sortedNodes.isEmpty()) {
            ComputationalNode node = sortedNodes.remove(0);
            if (node.childrenSize() > 0) {
                for (int i = 0; i < node.childrenSize(); i++) {
                    ComputationalNode child = node.getChild(i);
                    Tensor derivative = calculateDerivative(node, child);
                    if (derivative != null) {
                        if (node.getBackward() == null) {
                            node.setBackward(derivative);
                        } else {
                            node.setBackward(node.getBackward().add(derivative));
                        }
                    }
                }
            }
        }
        this.parameters.getOptimizer().updateValues(this.leafNodes);
        clear();
    }

    /**
     * Add a bias term to the node's value by appending a column of ones.
     * @param tensor The node whose value needs to be biased.
     */
    private void getBiased(ComputationalNode tensor) {
        int lastDimensionSize = tensor.getValue().getShape()[tensor.getValue().getShape().length - 1];
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) tensor.getValue().getData();
        for (int i = 0; i < oldValues.size(); i++) {
            values.add(oldValues.get(i));
            if ((i + 1) % lastDimensionSize == 0) {
                values.add(1.0);
            }
        }
        int[] shape = new int[tensor.getValue().getShape().length];
        for (int i = 0; i < shape.length; i++) {
            if (i == shape.length - 1) {
                shape[i] = tensor.getValue().getShape()[i] + 1;
            } else {
                shape[i] = tensor.getValue().getShape()[i];
            }
        }
        Tensor biasedValue = new Tensor(values, shape);
        tensor.setValue(biasedValue);
    }

    /**
     * Perform a forward pass and return predicted class indices.
     * @return A list of predicted class indices.
     */
    protected ArrayList<Double> predict() {
        ArrayList<Double> classLabels = forwardCalculation(false);
        clear();
        return classLabels;
    }

    /**
     * Perform a forward pass for the training phase.
     * @return A list of predicted class indices.
     */
    protected ArrayList<Double> forwardCalculation() {
        return forwardCalculation(true);
    }

    /**
     * Perform a forward pass through the computational graph.
     * @param isTraining indicates whether the forward pass is for training or inference.
     * @return A list of predicted class indices.
     */
    private ArrayList<Double> forwardCalculation(boolean isTraining) {
        LinkedList<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return new ArrayList<>();
        HashMap<ComputationalNode, ComputationalNode[]> concatenatedNodeMap = new HashMap<>();
        HashMap<ComputationalNode, Integer> counterMap = new HashMap<>();
        while (sortedNodes.size() > 1) {
            ComputationalNode currentNode = sortedNodes.removeLast();
            if (currentNode.isBiased()) {
                getBiased(currentNode);
            }
            if (currentNode.getValue() == null) {
                throw new IllegalArgumentException("leaf node's value must be initialized first.");
            }
            if (currentNode.childrenSize() > 0) {
                if (currentNode.equals(outputNode) && !isTraining) {
                    break;
                }
                for (int t = 0; t < currentNode.childrenSize(); t++) {
                    ComputationalNode child = currentNode.getChild(t);
                    if (child.getValue() == null) {
                        if (child instanceof FunctionNode) {
                            Function function = ((FunctionNode) child).getFunction();
                            Tensor currentValue = currentNode.getValue();
                            if (function instanceof Dropout) {
                                if (isTraining) {
                                    child.setValue(function.calculate(currentValue));
                                } else {
                                    child.setValue(new Tensor(currentValue.getData(), currentValue.getShape()));
                                }
                            } else {
                                child.setValue(function.calculate(currentValue));
                            }
                        } else {
                            if (child instanceof ConcatenatedNode) {
                                if (!concatenatedNodeMap.containsKey(child)) {
                                    concatenatedNodeMap.put(child, new ComputationalNode[child.parentsSize()]);
                                }
                                concatenatedNodeMap.get(child)[((ConcatenatedNode) child).getIndex(currentNode)] = currentNode;
                                if (!counterMap.containsKey(child)) {
                                    counterMap.put(child, 0);
                                }
                                counterMap.put(child, counterMap.get(child) + 1);
                                if (child.parentsSize() == counterMap.get(child)) {
                                    child.setValue(concatenatedNodeMap.get(child)[0].getValue());
                                    for (int i = 1; i < concatenatedNodeMap.get(child).length; i++) {
                                        child.setValue(child.getValue().concat(concatenatedNodeMap.get(child)[i].getValue(), ((ConcatenatedNode) child).getDimension()));
                                    }
                                }
                            } else {
                                child.setValue(currentNode.getValue());
                            }
                        }
                    } else {
                        if (child instanceof MultiplicationNode) {
                            Tensor childValue = child.getValue();
                            Tensor currentValue = currentNode.getValue();
                            if (((MultiplicationNode) child).isHadamard()) {
                                child.setValue(childValue.hadamardProduct(currentValue));
                            } else if (!((MultiplicationNode) child).getPriorityNode().equals(currentNode)) {
                                child.setValue(childValue.multiply(currentValue));
                            } else {
                                child.setValue(currentValue.multiply(childValue));
                            }
                        } else {
                            Tensor result = child.getValue();
                            Tensor currentValue = currentNode.getValue();
                            child.setValue(result.add(currentValue));
                        }
                    }
                }
            }
        }
        return getOutputValue();
    }

    /**
     * The save method takes a file name as an input and writes the model to that file.
     * @param fileName File name.
     */
    public void save(String fileName) {
        FileOutputStream outFile;
        ObjectOutputStream outObject;
        try {
            outFile = new FileOutputStream(fileName);
            outObject = new ObjectOutputStream(outFile);
            outObject.writeObject(this);
        } catch (IOException ignored) {
            System.out.println("Object could not be saved.");
        }
    }

    /**
     * The loadModel method takes a file name as an input and loads the {@link ComputationalGraph} related to that file.
     * @param fileName File name.
     */
    public static ComputationalGraph loadModel(String fileName) {
        FileInputStream inFile;
        ObjectInputStream inObject;
        try {
            inFile = new FileInputStream(fileName);
            inObject = new ObjectInputStream(inFile);
            return (ComputationalGraph) inObject.readObject();
        } catch (IOException | ClassNotFoundException e) {
            return null;
        }
    }
}