import numpy as np

class Matrix:
    @staticmethod
    def invert_matrix(matrix):
        real_matrix = np.array(matrix)
        inverted_matrix = np.linalg.inv(real_matrix)
        return inverted_matrix

    @staticmethod
    def multiply_matrix_vector(matrix, vector):
        real_matrix = np.array(matrix)
        real_vector = np.array(np.array(vector)).reshape(-1, 1)
        result = np.dot(real_matrix, real_vector)
        return result

    @staticmethod
    def create_identity_matrix(size):
        identity_matrix = np.eye(size)
        return identity_matrix

    @staticmethod
    def multiply_scalar_matrix(scalar, matrix):
        real_matrix = np.array(matrix)
        result = scalar * real_matrix
        return result

    @staticmethod
    def multiply_scalar_vector(scalar, vector):
        real_vector = np.array(vector).reshape(-1, 1)
        result = scalar * real_vector
        return result

    @staticmethod
    def subtract_matrix(matrix1, matrix2):
        real_matrix1 = np.array(matrix1)
        real_matrix2 = np.array(matrix2)
        result = real_matrix1 - real_matrix2
        return result

    @staticmethod
    def add_matrix(matrix1, matrix2):
        real_matrix1 = np.array(matrix1)
        real_matrix2 = np.array(matrix2)
        result = real_matrix1 + real_matrix2
        return result

    @staticmethod
    def add_vector(vector1, vector2):
        real_vector1 = np.array(vector1)
        real_vector2 = np.array(vector2)
        result = real_vector1 + real_vector2
        return result