package ComputationalGraph;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.Function.Dropout;
import ComputationalGraph.Function.Function;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.ConcatenatedNode;
import ComputationalGraph.Node.MultiplicationNode;
import ComputationalGraph.Optimizer.Optimizer;
import Math.Tensor;

import java.io.*;
import java.util.*;

public abstract class ComputationalGraph implements Serializable {

    private final HashMap<ComputationalNode, ArrayList<ComputationalNode>> nodeMap = new HashMap<>();
    private final HashMap<ComputationalNode, ArrayList<ComputationalNode>> reverseNodeMap = new HashMap<>();
    protected ArrayList<ComputationalNode> inputNodes;

    public ComputationalGraph() {
        this.inputNodes = new ArrayList<>();
    }

    public abstract void train(ArrayList<Tensor> trainSet, NeuralNetworkParameter parameters);
    public abstract ClassificationPerformance test(ArrayList<Tensor> testSet);
    protected abstract ArrayList<Integer> getClassLabels(ComputationalNode outputNode);

    protected ComputationalNode addEdge(ComputationalNode first, Object second, boolean isBiased) {
        ComputationalNode newNode;
        if (second instanceof Function) {
            newNode = new ComputationalNode(false, (Function) second, isBiased);
        } else if (second instanceof MultiplicationNode) {
            newNode = new MultiplicationNode(false, isBiased, ((MultiplicationNode) second).isHadamard(), first);
        } else {
            throw new IllegalArgumentException("Illegal Type of Object: second");
        }
        nodeMap.computeIfAbsent(first, k -> new ArrayList<>()).add(newNode);
        reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(first);
        if (second instanceof ComputationalNode) {
            nodeMap.computeIfAbsent((ComputationalNode) second, k -> new ArrayList<>()).add(newNode);
            reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add((ComputationalNode) second);
        }
        return newNode;
    }

    protected ComputationalNode addEdge(ComputationalNode first, ComputationalNode second, boolean isBiased, boolean isHadamard) {
        ComputationalNode newNode = new MultiplicationNode(false, isBiased, isHadamard, first);
        nodeMap.computeIfAbsent(first, k -> new ArrayList<>()).add(newNode);
        reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(first);
        nodeMap.computeIfAbsent(second, k -> new ArrayList<>()).add(newNode);
        reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(second);
        return newNode;
    }

    protected ComputationalNode addAdditionEdge(ComputationalNode first, ComputationalNode second, boolean isBiased) {
        ComputationalNode newNode = new ComputationalNode(false, null, isBiased);
        nodeMap.computeIfAbsent(first, k -> new ArrayList<>()).add(newNode);
        reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(first);
        nodeMap.computeIfAbsent(second, k -> new ArrayList<>()).add(newNode);
        reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(second);
        return newNode;
    }

    protected ComputationalNode concatEdges(ArrayList<ComputationalNode> nodes, int dimension) {
        ConcatenatedNode newNode = new ConcatenatedNode(dimension);
        for (ComputationalNode node : nodes) {
            nodeMap.computeIfAbsent(node, k -> new ArrayList<>()).add(newNode);
            reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(node);
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
        if (nodeMap.containsKey(node)) {
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!visited.contains(child)) {
                    queue.addAll(sortRecursive(child, visited));
                }
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
        for (ComputationalNode node : nodeMap.keySet()) {
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
        if (nodeMap.containsKey(node)) {
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!visited.contains(child)) {
                    clearRecursive(visited, child);
                }
            }
        }
    }

    /**
     * Clears the values and gradients of all nodes in the graph.
     */
    private void clear() {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                clearRecursive(visited, node);
            }
        }
    }

    /**
     * Swaps last two dimensions of the Tensor.
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
     * Calculates the derivative of the child node with respect to the parent node.
     * @param node Parent node.
     * @param child Child node.
     * @return The gradient tensor.
     */
    private Tensor calculateDerivative(ComputationalNode node, ComputationalNode child) {
        ArrayList<ComputationalNode> reverseChildren = reverseNodeMap.get(child);
        if (reverseChildren == null || reverseChildren.isEmpty()) {
            return null;
        }
        Tensor backward;
        if (child.isBiased()) {
            int[] endIndexes = new int[child.getBackward().getShape().length];
            for (int i = 0; i < endIndexes.length; i++) {
                if (i == endIndexes.length - 1) {
                    endIndexes[i] = child.getBackward().getShape()[i] - 1;
                } else {
                    endIndexes[i] = child.getBackward().getShape()[i];
                }
            }
            backward = child.getBackward().partial(new int[child.getBackward().getShape().length], endIndexes);
        } else {
            backward = child.getBackward();
        }
        if (child.getFunction() != null) {
            Function function = child.getFunction();
            Tensor childValue;
            if (child.isBiased()) {
                int[] endIndexes = new int[child.getValue().getShape().length];
                for (int i = 0; i < endIndexes.length; i++) {
                    if (i == endIndexes.length - 1) {
                        endIndexes[i] = child.getValue().getShape()[i] - 1;
                    } else {
                        endIndexes[i] = child.getValue().getShape()[i];
                    }
                }
                childValue = child.getValue().partial(new int[child.getValue().getShape().length], endIndexes);
            } else {
                childValue = child.getValue();
            }
            return function.derivative(childValue, backward);
        } else {
            if (child instanceof ConcatenatedNode) {
                int index = ((ConcatenatedNode) child).getIndex(node);
                int blockSize = backward.getShape()[((ConcatenatedNode) child).getDimension()] / reverseChildren.size();
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
                int cur = 0;
                int i = 0;
                while (i < childValues.size()) {
                    if (cur % reverseChildren.size() == index) {
                        for (int k = 0; k < dimensions; k++) {
                            newValues.add(childValues.get(i + k));
                        }
                    }
                    cur++;
                    i += dimensions;
                }
                return new Tensor(newValues, shape);
            } else {
                if (child instanceof MultiplicationNode) {
                    ComputationalNode left = reverseChildren.get(0);
                    ComputationalNode right = reverseChildren.get(1);
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
                    throw new NullPointerException("Backward and/or left child values are null");
                }
                return backward;
            }
        }
    }

    /**
     * Computes the difference between the predicted and actual values (R - Y).
     * @param output The output node of the computational graph.
     * @param classLabelIndex A list of true class labels (index of the correct class for each sample).
     */
    private void calculateRMinusY(ComputationalNode output, ArrayList<Integer> classLabelIndex) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> outputValues = (ArrayList<Double>) output.getValue().getData();
        int lastDimension = output.getValue().getShape()[output.getValue().getShape().length - 1];
        for (int i = 0; i < outputValues.size(); i++) {
            if (i % lastDimension == classLabelIndex.get(i / lastDimension)) {
                values.add(1 - outputValues.get(i));
            } else {
                values.add(-outputValues.get(i));
            }
        }
        Tensor backward = new Tensor(values, output.getValue().getShape());
        output.setBackward(backward);
    }

    /**
     * Performs backpropagation on the computational graph.
     * @param optimizer Optimizer to be used for updating the values.
     * @param classLabelIndex The true class labels (as a list of integers).
     */
    protected void backpropagation(Optimizer optimizer, ArrayList<Integer> classLabelIndex) {
        LinkedList<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return;
        ComputationalNode outputNode = sortedNodes.remove(0);
        calculateRMinusY(outputNode, classLabelIndex);
        if (!sortedNodes.isEmpty()) {
            sortedNodes.remove(0).setBackward(outputNode.getBackward());
        }
        while (!sortedNodes.isEmpty()) {
            ComputationalNode node = sortedNodes.remove(0);
            ArrayList<ComputationalNode> children = nodeMap.get(node);
            if (children != null) {
                for (ComputationalNode child : children) {
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
        optimizer.updateValues(this.nodeMap);
        clear();
    }

    /**
     * Add a bias term to the node's value by appending a column of ones.
     */
    private void getBiased(ComputationalNode first) {
        int lastDimensionSize = first.getValue().getShape()[first.getValue().getShape().length - 1];
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> oldValues = (ArrayList<Double>) first.getValue().getData();
        for (int i = 0; i < oldValues.size(); i++) {
            values.add(oldValues.get(i));
            if ((i + 1) % lastDimensionSize == 0) {
                values.add(1.0);
            }
        }
        int[] shape = new int[first.getValue().getShape().length];
        for (int i = 0; i < shape.length; i++) {
            if (i == shape.length - 1) {
                shape[i] = first.getValue().getShape()[i] + 1;
            } else {
                shape[i] = first.getValue().getShape()[i];
            }
        }
        Tensor biasedValue = new Tensor(values, shape);
        first.setValue(biasedValue);
    }

    /**
     * Perform a forward pass and return predicted class indices.
     */
    protected ArrayList<Integer> predict() {
        ArrayList<Integer> classLabels = forwardCalculation(false);
        clear();
        return classLabels;
    }

    /**
     * Perform a forward pass for the training phase.
     */
    protected ArrayList<Integer> forwardCalculation() {
        return forwardCalculation(true);
    }

    /**
     * Perform a forward pass through the computational graph.
     * @param isDropout Whether to perform dropout or not.
     * @return A list of predicted class indices.
     */
    private ArrayList<Integer> forwardCalculation(boolean isDropout) {
        LinkedList<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return new ArrayList<>();
        ComputationalNode outputNode = sortedNodes.getFirst();
        HashMap<ComputationalNode, ComputationalNode[]> concatenatedNodeMap = new HashMap<>();
        HashMap<ComputationalNode, Integer> counterMap = new HashMap<>();
        while (sortedNodes.size() > 1) {
            ComputationalNode currentNode = sortedNodes.removeLast();
            if (currentNode.isBiased()) {
                getBiased(currentNode);
            }
            if (currentNode.getValue() == null) {
                throw new IllegalArgumentException("Current node's value is null");
            }
            ArrayList<ComputationalNode> children = nodeMap.get(currentNode);
            if (children != null) {
                for (ComputationalNode child : children) {
                    if (child.getValue() == null) {
                        if (child.getFunction() != null) {
                            Function function = child.getFunction();
                            Tensor currentValue = currentNode.getValue();
                            if (function instanceof Dropout) {
                                if (isDropout) {
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
                                    concatenatedNodeMap.put(child, new ComputationalNode[reverseNodeMap.get(child).size()]);
                                }
                                concatenatedNodeMap.get(child)[((ConcatenatedNode) child).getIndex(currentNode)] = currentNode;
                                if (!counterMap.containsKey(child)) {
                                    counterMap.put(child, 0);
                                }
                                counterMap.put(child, counterMap.get(child) + 1);
                                if (reverseNodeMap.get(child).size() == counterMap.get(child)) {
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
        return getClassLabels(outputNode);
    }

    /**
     * The save method takes a file name as an input and writes model to that file.
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