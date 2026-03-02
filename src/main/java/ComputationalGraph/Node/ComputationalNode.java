package ComputationalGraph.Node;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class ComputationalNode implements Serializable {

    protected Tensor value;
    protected Tensor backward;
    protected final boolean isBiased;
    protected final boolean learnable;
    private final ArrayList<ComputationalNode> children;
    private final ArrayList<ComputationalNode> parents;

    /**
     * Initializes a ComputationalNode.
     * @param isBiased Indicates whether the node is biased
     * @param value The tensor value associated with the node (optional)
     */
    public ComputationalNode(boolean learnable, boolean isBiased, Tensor value) {
        this.value = value;
        this.backward = null;
        this.isBiased = isBiased;
        this.learnable = learnable;
        children = new ArrayList<>();
        parents = new ArrayList<>();
    }

    /**
     * Constructor overload for function type initialization
     */
    public ComputationalNode(boolean learnable, boolean isBiased) {
        this(learnable, isBiased, null);
    }

    public ComputationalNode getChild(int index) {
        return children.get(index);
    }

    public void addChild(ComputationalNode child) {
        children.add(child);
    }

    public void addParent(ComputationalNode parent) {
        parents.add(parent);
    }

    public void add(ComputationalNode child) {
        children.add(child);
        child.addParent(this);
    }

    public ComputationalNode getParent(int index) {
        return parents.get(index);
    }

    public int childrenSize() {
        return children.size();
    }

    public int parentsSize() {
        return parents.size();
    }

    public boolean isLearnable() {
        return learnable;
    }

    /**
     * @return a string representation of the node.
     */
    @Override
    public String toString() {
        StringBuilder details = new StringBuilder();
        if (value != null) {
            if (details.length() > 0) {
                details.append(", ");
            }
            details.append("Value Shape: [").append(value.getShape()[0]);
            for (int i = 1; i < value.getShape().length; i++) {
                details.append(", ").append(value.getShape()[i]);
            }
            details.append("]");
        }
        if (details.length() > 0) details.append(", ");
        details.append("is learnable: ").append(learnable);
        details.append(", is biased: ").append(isBiased);
        return "Node(" + details + ")";
    }

    public boolean isBiased() {
        return isBiased;
    }

    public Tensor getValue() {
        return value;
    }

    public void setValue(Tensor value) {
        this.value = value;
    }

    public void updateValue() {
        this.setValue(value.add(backward));
    }

    public Tensor getBackward() {
        return backward;
    }

    public void setBackward(Tensor backward) {
        this.backward = backward;
    }
}