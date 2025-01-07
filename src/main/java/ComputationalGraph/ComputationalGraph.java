package ComputationalGraph;

import java.util.*;
import Math.*;

public class ComputationalGraph {

    private final HashMap<ComputationalNode, ArrayList<ComputationalNode>> nodeMap;
    private final HashMap<ComputationalNode, ArrayList<ComputationalNode>> reverseNodeMap;

    public ComputationalGraph() {
        this.nodeMap = new HashMap<>();
        this.reverseNodeMap = new HashMap<>();
    }

    public ComputationalNode addEdge(ComputationalNode first, ComputationalNode second) {
        ComputationalNode newNode = new ComputationalNode(false, second.getOperator());
        if (!nodeMap.containsKey(first)) {
            nodeMap.put(first, new ArrayList<>());
        }
        if (!nodeMap.containsKey(second)) {
            nodeMap.put(second, new ArrayList<>());
        }
        nodeMap.get(first).add(newNode);
        nodeMap.get(second).add(newNode);
        if (!reverseNodeMap.containsKey(newNode)) {
            reverseNodeMap.put(newNode, new ArrayList<>());
        }
        reverseNodeMap.get(newNode).add(first);
        reverseNodeMap.get(newNode).add(second);
        return newNode;
    }

    public ComputationalNode addEdge(ComputationalNode node, FunctionType type) {
        ComputationalNode newNode = new ComputationalNode(false, type);
        if (!nodeMap.containsKey(node)) {
            nodeMap.put(node, new ArrayList<>());
        }
        nodeMap.get(node).add(newNode);
        if (!reverseNodeMap.containsKey(newNode)) {
            reverseNodeMap.put(newNode, new ArrayList<>());
        }
        reverseNodeMap.get(newNode).add(node);
        return newNode;
    }

    private LinkedList<ComputationalNode> sort(ComputationalNode root, HashSet<ComputationalNode> visited) {
        LinkedList<ComputationalNode> queue = new LinkedList<>();
        visited.add(root);
        if (nodeMap.containsKey(root)) {
            for (ComputationalNode node : nodeMap.get(root)) {
                if (!visited.contains(node)) {
                    queue.addAll(sort(node, visited));
                }
            }
        }
        queue.add(root);
        return queue;
    }

    private LinkedList<ComputationalNode> topologicalSort() {
        LinkedList<ComputationalNode> list = new LinkedList<>();
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                LinkedList<ComputationalNode> queue = sort(node, visited);
                while (!queue.isEmpty()) {
                    list.add(queue.getFirst());
                    queue.removeFirst();
                }
            }
        }
        return list;
    }

    private void clear() {
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!node.isLearnable()) {
                node.setValue(null);
            }
            node.setBackward(null);
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!node.isLearnable()) {
                    child.setValue(null);
                }
                child.setBackward(null);
            }
        }
    }

    private void update(HashSet<ComputationalNode> visited, ComputationalNode node) throws MatrixDimensionMismatch {
        visited.add(node);
        if (node.isLearnable()) {
            node.updateValue();
        }
        if (nodeMap.containsKey(node)) {
            for (ComputationalNode child : nodeMap.get(node)) {
                if (!visited.contains(child)) {
                    update(visited, child);
                }
            }
        }
    }

    private void updateValues() throws MatrixDimensionMismatch {
        HashSet<ComputationalNode> visited = new HashSet<>();
        for (ComputationalNode node : nodeMap.keySet()) {
            if (!visited.contains(node)) {
                update(visited, node);
            }
        }
    }

    private Matrix calculateDerivative(ComputationalNode node, ComputationalNode child) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        ComputationalNode left = reverseNodeMap.get(child).get(0);
        if (reverseNodeMap.get(child).size() == 1) {
            Function function;
            switch (child.getFunctionType()) {
                case SIGMOID:
                    function = new Sigmoid();
                    return child.getBackward().elementProduct(function.derivative(child.getValue()));
                case TANH:
                    function = new Tanh();
                    return child.getBackward().elementProduct(function.derivative(child.getValue()));
                case RELU:
                    function = new ReLU();
                    return child.getBackward().elementProduct(function.derivative(child.getValue()));
                case SOFTMAX:
                    function = new Softmax();
                    return child.getBackward().elementProduct(function.derivative(child.getValue()));
                default:
                    return null;
            }
        } else {
            ComputationalNode right = reverseNodeMap.get(child).get(1);
            switch (child.getOperator()) {
                case '*':
                    if (left.equals(node)) {
                        return child.getBackward().multiply(right.getValue().transpose());
                    }
                    return left.getValue().transpose().multiply(child.getBackward());
                case '+':
                    return child.getBackward().clone();
                case '-':
                    if (left.equals(node)) {
                        return child.getBackward().clone();
                    }
                    Matrix result = child.getBackward().clone();
                    for (int i = 0; i < result.getRow(); i++) {
                        for (int j = 0; j < result.getColumn(); j++) {
                            result.setValue(i, j, -result.getValue(i, j));
                        }
                    }
                    return result;
            }
        }
        return null;
    }

    private void calculateRMinusY(ComputationalNode output, double learningRate, ArrayList<Integer> classLabelIndex) {
        Matrix backward = new Matrix(output.getValue().getRow(), output.getValue().getColumn());
        for (int i = 0; i < output.getValue().getRow(); i++) {
            for (int j = 0; j < output.getValue().getColumn(); j++) {
                if (classLabelIndex.get(i).equals(j)) {
                    backward.setValue(i, j, (1 - output.getValue().getValue(i, j)) * learningRate);
                } else {
                    backward.setValue(i, j, (-output.getValue().getValue(i, j)) * learningRate);
                }
            }
        }
        output.setBackward(backward);
    }

    public void backpropagation(double learningRate, ArrayList<Integer> classLabelIndex) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        LinkedList<ComputationalNode> sortedNodes = topologicalSort();
        ComputationalNode output = sortedNodes.removeFirst();
        calculateRMinusY(output, learningRate, classLabelIndex);
        sortedNodes.removeFirst().setBackward(output.getBackward().clone());
        while (!sortedNodes.isEmpty()) {
            ComputationalNode node = sortedNodes.removeFirst();
            for (ComputationalNode child : nodeMap.get(node)) {
                if (node.getBackward() == null) {
                    node.setBackward(calculateDerivative(node, child));
                } else {
                    node.getBackward().add(calculateDerivative(node, child));
                }
            }
        }
        updateValues();
        clear();
    }

    public ArrayList<Integer> forwardCalculation() throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LinkedList<ComputationalNode> sortedNodes = topologicalSort();
        ComputationalNode output = sortedNodes.getFirst();
        while (sortedNodes.size() != 1) {
            ComputationalNode currentNode = sortedNodes.removeLast();
            for (ComputationalNode child : nodeMap.get(currentNode)) {
                if (child.getValue() == null) {
                    if (child.getFunctionType() != null) {
                        Function function;
                        switch (child.getFunctionType()) {
                            case TANH:
                                function = new Tanh();
                                child.setValue(function.calculate(currentNode.getValue()));
                                break;
                            case SIGMOID:
                                function = new Sigmoid();
                                child.setValue(function.calculate(currentNode.getValue()));
                                break;
                            case RELU:
                                function = new ReLU();
                                child.setValue(function.calculate(currentNode.getValue()));
                                break;
                            case SOFTMAX:
                                function = new Softmax();
                                child.setValue(function.calculate(currentNode.getValue()));
                                break;
                            default:
                                break;
                        }
                    } else {
                        child.setValue(currentNode.getValue().clone());
                    }
                } else {
                    if (child.getFunctionType() == null) {
                        Matrix result;
                        switch (child.getOperator()) {
                            case '*':
                                if (child.getValue().getColumn() == currentNode.getValue().getRow()) {
                                    child.setValue(child.getValue().multiply(currentNode.getValue()));
                                } else {
                                    child.setValue(currentNode.getValue().multiply(child.getValue()));
                                }
                                break;
                            case '+':
                                result = child.getValue().clone();
                                result.add(currentNode.getValue());
                                child.setValue(result);
                                break;
                            case '-':
                                result = child.getValue().clone();
                                result.subtract(currentNode.getValue());
                                child.setValue(result);
                                break;
                            default:
                                break;
                        }
                    }
                }
            }
        }
        ArrayList<Integer> classLabelIndex = new ArrayList<>();
        for (int i = 0; i < output.getValue().getRow(); i++) {
            double max = Integer.MIN_VALUE;
            int labelIndex = -1;
            for (int j = 0; j < output.getValue().getColumn(); j++) {
                if (max < output.getValue().getValue(i, j)) {
                    max = output.getValue().getValue(i, j);
                    labelIndex = j;
                }
            }
            classLabelIndex.add(labelIndex);
        }
        return classLabelIndex;
    }
}
