package ComputationalGraph;

import Math.Matrix;

public class ReLU implements Function {

    @Override
    public Matrix calculate(Matrix matrix) {
        Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int i = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++) {
                if (matrix.getValue(i, j) > 0) {
                    result.setValue(i, j, matrix.getValue(i, j));
                } else {
                    result.setValue(i, j, 0.0);
                }
            }
        }
        return result;
    }

    @Override
    public Matrix derivative(Matrix matrix) {
        Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int i = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++) {
                if (matrix.getValue(i, j) != 0) {
                    result.setValue(i, j, 1.0);
                } else {
                    result.setValue(i, j, 0.0);
                }
            }
        }
        return result;
    }
}
