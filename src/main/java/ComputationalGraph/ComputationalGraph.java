package ComputationalGraph;

import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
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

    public abstract void train(Tensor trainSet, Parameter parameters);
    public abstract ClassificationPerformance test(Tensor testSet);
    protected abstract ArrayList<Integer> getClassLabels(ComputationalNode outputNode);

    protected ComputationalNode addEdge(ComputationalNode first, Object second, boolean isBiased) {
        ComputationalNode newNode;
        if (second instanceof Function) {
            newNode = new ComputationalNode(false, (Function) second, isBiased);
        } else if (second instanceof MultiplicationNode) {
            newNode = new MultiplicationNode(false, isBiased, ((MultiplicationNode) second).isHadamard());
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
        ComputationalNode newNode = new MultiplicationNode(false, isBiased, isHadamard);
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

    protected ComputationalNode concatEdges(ArrayList<ComputationalNode> nodes) {
        ComputationalNode newNode = new ConcatenatedNode();
        for (ComputationalNode node : nodes) {
            nodeMap.computeIfAbsent(node, k -> new ArrayList<>()).add(newNode);
            reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(node);
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
     * Recursive helper function to update the values of learnable nodes.
     */
    private void updateRecursive(HashSet<ComputationalNode> visited, ComputationalNode node) {
        visited.add(node);
        if (node.isLearnable()) {
            node.updateValue();
        }
        if (nodeMap.containsKey(node)) {
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!visited.contains(child)) {
                    updateRecursive(visited, child);
                }
            }
        }
    }

    /**
     * Updates the values of all learnable nodes in the graph.
     */
    private void updateValues() {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                updateRecursive(visited, node);
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
        ComputationalNode left = reverseChildren.get(0);
        if (reverseChildren.size() == 1) {
            Function function = child.getFunction();
            Tensor backward = child.getBackward();
            return function.derivative(child.getValue(), backward);
        } else {
            if (child instanceof ConcatenatedNode) {
                int index = ((ConcatenatedNode) child).getIndex(node);
                int blockSize = child.getBackward().getShape()[child.getBackward().getShape().length - 1] / reverseChildren.size();
                ArrayList<Double> values = (ArrayList<Double>) child.getBackward().getData(), newValues = new ArrayList<>();
                int[] shape = new int[child.getBackward().getShape().length];
                int size = 1;
                for (int i = 0; i < shape.length - 1; i++) {
                    shape[i] = child.getBackward().getShape()[i];
                    size *= shape[i];
                }
                shape[shape.length - 1] = blockSize;
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < blockSize; j++) {
                        newValues.add(values.get((i * child.getBackward().getShape()[child.getBackward().getShape().length - 1]) + (index * blockSize + j)));
                    }
                }
                return new Tensor(newValues, shape);
            } else {
                if (child instanceof MultiplicationNode) {
                    ComputationalNode right = reverseChildren.get(1);
                    Tensor backward = child.getBackward();
                    if (left == node) {
                        Tensor rightValue = right.getValue();
                        if (((MultiplicationNode) child).isHadamard()) {
                            return rightValue.hadamardProduct(backward);
                        }
                        if (!child.isBiased()) {
                            if (backward != null && rightValue != null) {
                                return backward.multiply(rightValue.transpose(transposeAxes(rightValue.getShape().length)));
                            }
                            throw new NullPointerException("Backward and/or right child values are null");
                        }
                        int[] endIndexes = new int[backward.getShape().length];
                        for (int i = 0; i < endIndexes.length; i++) {
                            if (i == endIndexes.length - 1) {
                                endIndexes[i] = backward.getShape()[i] - 1;
                            } else {
                                endIndexes[i] = backward.getShape()[i];
                            }
                        }
                        Tensor partial = backward.partial(new int[backward.getShape().length], endIndexes);
                        if (partial != null && rightValue != null) {
                            return partial.multiply(rightValue.transpose(transposeAxes(rightValue.getShape().length)));
                        }
                        throw new NullPointerException("Backward and/or right child values are null");
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
                return child.getBackward();
            }
        }
    }

    /**
     * Computes the difference between the predicted and actual values (R - Y).
     * @param output The output node of the computational graph.
     * @param learningRate The learning rate for gradient descent.
     * @param classLabelIndex A list of true class labels (index of the correct class for each sample).
     */
    private void calculateRMinusY(ComputationalNode output, double learningRate, ArrayList<Integer> classLabelIndex) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> outputValues = (ArrayList<Double>) output.getValue().getData();
        int lastDimension = output.getValue().getShape()[output.getValue().getShape().length - 1];
        for (int i = 0; i < outputValues.size(); i++) {
            if (i % lastDimension == classLabelIndex.get(i / lastDimension)) {
                values.add((1 - outputValues.get(i)) * learningRate);
            } else {
                values.add(-outputValues.get(i) * learningRate);
            }
        }
        Tensor backward = new Tensor(values, output.getValue().getShape());
        output.setBackward(backward);
    }

    /**
     * Performs backpropagation on the computational graph.
     * @param learningRate The learning rate for gradient descent.
     * @param classLabelIndex The true class labels (as a list of integers).
     */
    protected void backpropagation(double learningRate, ArrayList<Integer> classLabelIndex) {
        LinkedList<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return;
        ComputationalNode outputNode = sortedNodes.remove(0);
        calculateRMinusY(outputNode, learningRate, classLabelIndex);
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
        updateValues();
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
    public ArrayList<Integer> predict() {
        ArrayList<Integer> classLabels = forwardCalculation();
        clear();
        return classLabels;
    }

    /**
     * Perform a forward pass through the computational graph.
     * @return A list of predicted class indices.
     */
    protected ArrayList<Integer> forwardCalculation() {
        LinkedList<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return new ArrayList<>();
        ComputationalNode outputNode = sortedNodes.getFirst();
        while (sortedNodes.size() > 1) {
            ComputationalNode currentNode = sortedNodes.removeLast();
            boolean isBiased = false;
            ArrayList<ComputationalNode> children = nodeMap.get(currentNode);
            if (children != null) {
                for (ComputationalNode child : children) {
                    if (child.getValue() == null) {
                        if (child.getFunction() != null) {
                            Function function = child.getFunction();
                            Tensor currentValue = currentNode.getValue();
                            if (currentValue != null) {
                                child.setValue(function.calculate(currentValue));
                            }
                        } else {
                            if (!isBiased && currentNode.isBiased()) {
                                getBiased(currentNode);
                                isBiased = true;
                            }
                            child.setValue(currentNode.getValue());
                            if (child instanceof ConcatenatedNode) {
                                ((ConcatenatedNode) child).addNode(currentNode);
                            }
                        }
                    } else {
                        if (child instanceof ConcatenatedNode) {
                            child.setValue(child.getValue().concat(currentNode.getValue()));
                            ((ConcatenatedNode) child).addNode(currentNode);
                        } else {
                            if (child instanceof MultiplicationNode) {
                                if (!isBiased && currentNode.isBiased()) {
                                    getBiased(currentNode);
                                    isBiased = true;
                                }
                                Tensor childValue = child.getValue();
                                Tensor currentValue = currentNode.getValue();
                                if (childValue != null && currentValue != null) {
                                    if (((MultiplicationNode) child).isHadamard()) {
                                        child.setValue(childValue.hadamardProduct(currentValue));
                                    } else if (childValue.getShape()[childValue.getShape().length - 1] == currentValue.getShape()[currentValue.getShape().length - 2]) {
                                        child.setValue(childValue.multiply(currentValue));
                                    } else {
                                        child.setValue(currentValue.multiply(childValue));
                                    }
                                }
                            } else {
                                Tensor result = child.getValue();
                                Tensor currentValue = currentNode.getValue();
                                if (result != null && currentValue != null) {
                                    child.setValue(result.add(currentValue));
                                }
                            }
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