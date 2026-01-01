import numpy as np
import math

class AIPlayer:
    def __init__(self):
        # Wagi: [Empty, Mono, Smooth, Max, Corner]
        self.weights = np.array([10.0, 1.0, 0.1, 1.0, 2.0])
        self.alpha = 0.001 #Learning rate

    def get_features(self, board):
        empty = len(board[board == 0])
        m_max = np.max(board)
        log_max = math.log2(m_max) if m_max > 0 else 0


        # Smoothness (różnica sąsiadów) i Monotonicity
        smoothness = 0
        monotonicity = 0

        for current_board in [board, board.T]:  # Wiersze, potem kolumny
            for row in current_board:
                # 1. Monotonicity: Analiza logiczna (ignoruje zera)
                # Promuje porządek w całym rzędzie np. [16, 0, 4, 2] -> [16, 4, 2]
                vals = [math.log2(x) for x in row if x > 0]
                if len(vals) >= 2:
                    diffs = [vals[i] - vals[i+1] for i in range(len(vals)-1)]
                    inc = sum(abs(d) for d in diffs if d < 0)
                    dec = sum(abs(d) for d in diffs if d > 0)
                    monotonicity -= min(inc, dec)

                # 2. Smoothness: Analiza fizyczna (tylko bezpośredni sąsiedzi)
                # Promuje trzymanie klocków blisko siebie (zwarte grupy)
                for i in range(len(row) - 1):
                    if row[i] > 0 and row[i+1] > 0:
                        smoothness -= abs(math.log2(row[i]) - math.log2(row[i+1]))

        corners = [board[0, 0], board[0, -1], board[-1, 0], board[-1, -1]]
        corner = 1 if m_max in corners else 0

        return np.array([empty, monotonicity, smoothness, log_max, corner])

    def evaluate(self, board):
        return np.dot(self.weights, self.get_features(board))

    def update_weights(self, gradient, delta):
        self.weights += self.alpha * delta * gradient

    def get_expected_value(self, board):
        """Oblicza E[V(s'')] - średnią wartość po losowym pojawieniu się kafelka."""
        empty_cells = list(zip(*np.where(board == 0)))
        if not empty_cells:
            # Brak miejsca = koniec gry lub stan ustalony
            return self.evaluate(board)

        total_val = 0
        n_empty = len(empty_cells)

        # Prawdopodobieństwo wystąpienia konkretnego pola = 1 / n_empty
        # Ważona suma: 0.9 * V(z '2') + 0.1 * V(z '4')
        for r, c in empty_cells:
            # Przypadek 1: Pojawia się 2
            board[r, c] = 2
            val2 = self.evaluate(board)

            # Przypadek 2: Pojawia się 4
            board[r, c] = 4
            val4 = self.evaluate(board)

            # Przywracamy stan (backtracking)
            board[r, c] = 0

            total_val += 0.9 * val2 + 0.1 * val4

        return total_val / n_empty