package ComputationalGraph;

import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
import Math.Tensor;

import java.io.*;
import java.util.*;

public abstract class ComputationalGraph {

    private final HashMap<ComputationalNode, List<ComputationalNode>> nodeMap = new HashMap<>();
    private final HashMap<ComputationalNode, List<ComputationalNode>> reverseNodeMap = new HashMap<>();
    protected ArrayList<ComputationalNode> inputNodes;

    public ComputationalGraph() {
        this.inputNodes  = new ArrayList<>();
    }

    public abstract void train(Tensor trainSet, Parameter parameters);
    public abstract ClassificationPerformance test(Tensor testSet);
    public abstract void loadModel(String fileName);

    public ComputationalNode addEdge(ComputationalNode first, Object second, boolean isBiased) {
        ComputationalNode newNode;
        if (second instanceof Function) {
            newNode = new ComputationalNode(false, isBiased, null, (Function) second, null);
        } else if (second instanceof ComputationalNode) {
            newNode = new ComputationalNode(false, isBiased, ((ComputationalNode) second).getOperator(), null, null);
        } else {
            throw new IllegalArgumentException("Invalid type for 'second'. Must be a ComputationalNode or FunctionType.");
        }
        nodeMap.computeIfAbsent(first, k -> new ArrayList<>()).add(newNode);
        reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add(first);

        if (second instanceof ComputationalNode) {
            nodeMap.computeIfAbsent((ComputationalNode) second, k -> new ArrayList<>()).add(newNode);
            reverseNodeMap.computeIfAbsent(newNode, k -> new ArrayList<>()).add((ComputationalNode) second);
        }
        return newNode;
    }


    /**
     * Recursive helper function to perform depth-first search for topological sorting.
     * @param node The current node being processed.
     * @param visited A set of visited nodes.
     * @return A list representing the partial topological order.
     */
    private LinkedList<ComputationalNode> sortRecursive(ComputationalNode node, Set<ComputationalNode> visited) {
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
    public ArrayList<ComputationalNode> topologicalSort() {
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
        return new ArrayList<>(sortedList);
    }

    /**
     * Recursive helper function to clear the values and gradients of nodes.
     */
    private void clearRecursive(Set<ComputationalNode> visited, ComputationalNode node) {
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
    public void clear() {
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
    private void updateRecursive(Set<ComputationalNode> visited, ComputationalNode node) {
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
    public void updateValues() {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                updateRecursive(visited, node);
            }
        }
    }

    /**
     * Calculates the derivative of the child node with respect to the parent node.
     * @param node Parent node.
     * @param child Child node.
     * @return The gradient tensor.
     */
    public Tensor calculateDerivative(ComputationalNode node, ComputationalNode child) {
        List<ComputationalNode> reverseChildren = reverseNodeMap.get(child);
        if (reverseChildren == null || reverseChildren.isEmpty()) {
            return null;
        }
        ComputationalNode left = reverseChildren.get(0);
        if (reverseChildren.size() == 1) {
            Function function = child.getFunction();
            Tensor backward = child.getBackward();
            Tensor derivative = function.derivative(child.getValue());
            if (backward != null && derivative != null) {
                return backward.hadamardProduct(derivative);
            }
            return null;
        } else {
            ComputationalNode right = reverseChildren.get(1);
            if (child.getOperator() != null) {
                switch (child.getOperator()) {
                    case "*":
                        if (left == node) {
                            if (!child.isBiased()) {
                                Tensor backward = child.getBackward();
                                Tensor rightValue = right.getValue();
                                if (backward != null && rightValue != null) {
                                    return backward.multiply(rightValue.transpose(null));
                                }
                                return null;
                            }
                            Tensor backward = child.getBackward();
                            Tensor partial = backward.partial(new int[]{0, 0}, new int[]{backward.getShape()[0], backward.getShape()[1] - 1});
                            Tensor rightValue = right.getValue();
                            if (partial != null && rightValue != null) {
                                return partial.multiply(rightValue.transpose(null));
                            }
                            return null;
                        }
                        Tensor leftValue = left.getValue();
                        Tensor backward = child.getBackward();
                        if (leftValue != null && backward != null) {
                            return leftValue.transpose(null).multiply(backward);
                        }
                        return null;
                    case "+":
                        return child.getBackward();
                    case "-":
                        if (left == node) {
                            return child.getBackward();
                        } else {
                            Tensor result = child.getBackward();
                            if (result != null) {
                                int[] shape = result.getShape();
                                int totalElements = computeNumElements(shape);
                                for (int i = 0; i < totalElements; i++) {
                                    int[] indices = unflattenIndex(i, computeStrides(shape));
                                    result.set(indices, -result.getValue(indices));
                                }
                                return result;
                            }
                            return null;
                        }
                }
            }
        }
        return null;
    }

    /**
     * Computes the difference between the predicted and actual values (R - Y).
     * @param output The output node of the computational graph.
     * @param learningRate The learning rate for gradient descent.
     * @param classLabelIndex A list of true class labels (index of the correct class for each sample).
     */
    public void calculateRMinusY(ComputationalNode output, double learningRate, List<Integer> classLabelIndex) {
        Tensor outputValue = output.getValue();
        if (outputValue == null) return;
        
        int[] shape = outputValue.getShape();
        if (shape.length < 2) {
            throw new IllegalArgumentException("Output tensor must have at least 2 dimensions for classification. Got shape " + Arrays.toString(shape));
        }
        
        // Create backward tensor with same shape as output
        List<Double> backwardData = new ArrayList<>();
        int totalElements = computeNumElements(shape);
        for (int i = 0; i < totalElements; i++) {
            backwardData.add(0.0);
        }
        Tensor backward = new Tensor(backwardData, shape);
        
        // Calculate batch size (all dimensions except the last one)
        int[] batchShape = Arrays.copyOfRange(shape, 0, shape.length - 1);
        int batchSize = computeNumElements(batchShape);
        int classAxis = shape[shape.length - 1];
        
        // For each sample in the batch
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < classAxis; j++) {
                // Create indices for this position
                int[] indices = unflattenIndex(i, computeStrides(batchShape));
                int[] fullIndices = Arrays.copyOf(indices, indices.length + 1);
                fullIndices[fullIndices.length - 1] = j;
                
                double pred = outputValue.getValue(fullIndices);
                double target = (classLabelIndex.get(i) == j) ? 1.0 : 0.0;
                backward.set(fullIndices, (target - pred) * learningRate);
            }
        }
        output.setBackward(backward);
    }

    /**
     * Performs backpropagation on the computational graph.
     * @param learningRate The learning rate for gradient descent.
     * :param classLabelIndex: The true class labels (as a list of integers).
     */
    public void backpropagation(double learningRate, List<Integer> classLabelIndex) {
        List<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return;
        ComputationalNode outputNode = sortedNodes.remove(0);
        calculateRMinusY(outputNode, learningRate, classLabelIndex);
        if (!sortedNodes.isEmpty()) {
            sortedNodes.remove(0).setBackward(outputNode.getBackward());
        }
        while (!sortedNodes.isEmpty()) {
            ComputationalNode node = sortedNodes.remove(0);
            List<ComputationalNode> children = nodeMap.get(node);
            if (children != null) {
                for (ComputationalNode child : children) {
                    Tensor derivative = calculateDerivative(node, child);
                    if (derivative != null) {
                        if (node.getBackward() == null) {
                            node.setBackward(derivative);
                        } else {
                            Tensor currentBackward = node.getBackward();
                            int[] shape = currentBackward.getShape();
                            int totalElements = computeNumElements(shape);
                            for (int i = 0; i < totalElements; i++) {
                                int[] indices = unflattenIndex(i, computeStrides(shape));
                                double current = currentBackward.getValue(indices);
                                double delta = derivative.getValue(indices);
                                currentBackward.set(indices, current + delta);
                            }
                        }
                    }
                }
            }
        }
        updateValues();
        clear();
    }

    /**
     * Add a bias term to the node's value by appending a 1 along the last axis.
     */
    public void getBiased(ComputationalNode first) {
        Tensor firstValue = first.getValue();
        if (firstValue == null) return;
        
        int[] originalShape = firstValue.getShape();
        int[] newShape = Arrays.copyOf(originalShape, originalShape.length);
        newShape[newShape.length - 1] = originalShape[originalShape.length - 1] + 1;
        
        // Create biased tensor data
        List<Double> biasedData = new ArrayList<>();
        
        // Copy original data
        int totalElements = computeNumElements(originalShape);
        for (int i = 0; i < totalElements; i++) {
            int[] indices = unflattenIndex(i, computeStrides(originalShape));
            biasedData.add(firstValue.getValue(indices));
        }
        
        // Append bias = 1 for each outer sample
        int[] batchShape = Arrays.copyOfRange(originalShape, 0, originalShape.length - 1);
        int batchSize = computeNumElements(batchShape);
        for (int i = 0; i < batchSize; i++) {
            biasedData.add(1.0);
        }
        
        Tensor biasedTensor = new Tensor(biasedData, newShape);
        first.setValue(biasedTensor);
    }

    /**
     * Perform a forward pass and return predicted class indices.
     */
    public List<Integer> predict() {
        ArrayList<Integer> classLabels = forwardCalculation();
        clear();
        return classLabels;
    }

    /**
     * Perform a forward pass through the computational graph.
     * @return A list of predicted class indices.
     */
    public ArrayList<Integer> forwardCalculation() {
        ArrayList<ComputationalNode> sortedNodes = topologicalSort();
        if (sortedNodes.isEmpty()) return new ArrayList<>();
        ComputationalNode outputNode = sortedNodes.get(0);
        while (sortedNodes.size() > 1) {
            ComputationalNode currentNode = sortedNodes.remove(sortedNodes.size() - 1);
            List<ComputationalNode> children = nodeMap.get(currentNode);
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
                            if (currentNode.isBiased()) {
                                getBiased(currentNode);
                            }
                            child.setValue(currentNode.getValue());
                        }
                    } else {
                        if (child.getFunction() == null && child.getOperator() != null) {
                            switch (child.getOperator()) {
                                case "*": {
                                    if (currentNode.isBiased()) {
                                        getBiased(currentNode);
                                    }
                                    Tensor childValue = child.getValue();
                                    Tensor currentValue = currentNode.getValue();
                                    if (childValue != null && currentValue != null) {
                                        if (childValue.getShape()[1] == currentValue.getShape()[0]) {
                                            child.setValue(childValue.multiply(currentValue));
                                        } else {
                                            child.setValue(currentValue.multiply(childValue));
                                        }
                                    }
                                    break;
                                }
                                case "+": {
                                    Tensor result = child.getValue();
                                    Tensor currentValue = currentNode.getValue();
                                    if (result != null && currentValue != null) {
                                        child.setValue(result.add(currentValue));
                                    }
                                    break;
                                }
                                case "-": {
                                    Tensor result = child.getValue();
                                    Tensor currentValue = currentNode.getValue();
                                    if (result != null && currentValue != null) {
                                        child.setValue(result.subtract(currentValue));
                                    }
                                    break;
                                }
                                default:
                                    throw new IllegalArgumentException("Unsupported operator: " + child.getOperator());
                            }
                        }
                    }
                }
            }
        }
        ArrayList<Integer> classLabelIndices = new ArrayList<>();
        Tensor outputValue = outputNode.getValue();
        if (outputValue != null) {
            int[] shape = outputValue.getShape();
            if (shape.length < 2) {
                throw new IllegalArgumentException("Output tensor must have at least 2 dimensions for classification. Got shape " + Arrays.toString(shape));
            }
            
            // Calculate batch size (all dimensions except the last one)
            int[] batchShape = Arrays.copyOfRange(shape, 0, shape.length - 1);
            int batchSize = computeNumElements(batchShape);
            int classAxis = shape[shape.length - 1];
            
            // For each sample in the batch, find the class with maximum probability
            for (int i = 0; i < batchSize; i++) {
                double maxVal = Double.NEGATIVE_INFINITY;
                int labelIndex = -1;
                
                for (int j = 0; j < classAxis; j++) {
                    // Create indices for this position
                    int[] indices = unflattenIndex(i, computeStrides(batchShape));
                    int[] fullIndices = Arrays.copyOf(indices, indices.length + 1);
                    fullIndices[fullIndices.length - 1] = j;
                    
                    double val = outputValue.getValue(fullIndices);
                    if (maxVal < val) {
                        maxVal = val;
                        labelIndex = j;
                    }
                }
                classLabelIndices.add(labelIndex);
            }
        }
        return classLabelIndices;
    }

    /**
     * Helper method to compute the number of elements in a tensor shape.
     * @param shape The shape array.
     * @return Total number of elements.
     */
    private int computeNumElements(int[] shape) {
        int product = 1;
        for (int dim : shape) {
            product *= dim;
        }
        return product;
    }

    /**
     * Helper method to compute strides for a tensor shape.
     * @param shape The shape array.
     * @return Strides array.
     */
    private int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int product = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = product;
            product *= shape[i];
        }
        return strides;
    }

    /**
     * Helper method to convert a flat index to multi-dimensional indices.
     * @param flatIndex The flat index to convert.
     * @param strides The strides array.
     * @return Multi-dimensional indices.
     */
    private int[] unflattenIndex(int flatIndex, int[] strides) {
        int[] indices = new int[strides.length];
        for (int i = 0; i < strides.length; i++) {
            indices[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }
        return indices;
    }

    /**
     * The save method takes a file name as an input and writes model to that file.
     *
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
        }
    }
}