package ComputationalGraph;

import Math.*;

public class ComputationalNode {

    private final FunctionType functionType;
    private char operator;
    private Matrix value;
    private Matrix backward;
    private final boolean isLearnable;
    private final boolean isBiased;

    public ComputationalNode(boolean learnable, FunctionType functionType, boolean isBiased) {
        this.value = null;
        this.backward = null;
        this.isLearnable = learnable;
        this.functionType = functionType;
        this.isBiased = isBiased;
    }

    public ComputationalNode(boolean learnable, char operator, boolean isBiased) {
        this.value = null;
        this.backward = null;
        this.isLearnable = learnable;
        this.operator = operator;
        this.functionType = null;
        this.isBiased = isBiased;
    }

    public ComputationalNode(Matrix value, char operator) {
        this.value = value;
        this.backward = null;
        this.isLearnable = true;
        this.operator = operator;
        this.functionType = null;
        this.isBiased = false;
    }

    protected boolean isBiased() {
        return isBiased;
    }

    protected FunctionType getFunctionType() {
        return functionType;
    }

    protected char getOperator() {
        return operator;
    }

    protected Matrix getValue() {
        return value;
    }

    public void setValue(Matrix value) {
        this.value = value;
    }

    protected void updateValue() {
        for (int i = 0; i < value.getRow(); i++) {
            for (int j = 0; j < value.getColumn(); j++) {
                value.setValue(i, j, value.getValue(i, j) + backward.getValue(i, j));
            }
        }
    }
    protected boolean isLearnable() {
        return isLearnable;
    }

    protected Matrix getBackward() {
        return backward;
    }

    protected void setBackward(Matrix backward) {
        this.backward = backward;
    }
}
