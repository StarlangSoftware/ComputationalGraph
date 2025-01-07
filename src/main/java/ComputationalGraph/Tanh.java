package ComputationalGraph;

import Math.Matrix;

public class Tanh implements Function {

    @Override
    public Matrix calculate(Matrix matrix) {
        Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int i = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++) {
                result.setValue(i, j, Math.tanh(matrix.getValue(i, j)));
            }
        }
        return result;
    }

    @Override
    public Matrix derivative(Matrix matrix) {
        Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int i = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++) {
                result.setValue(i, j, 1 - (matrix.getValue(i, j) * matrix.getValue(i, j)));
            }
        }
        return result;
    }
}
