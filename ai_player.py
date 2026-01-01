import numpy as np
import math


class AIPlayer:
    def __init__(self):
        # Wagi startowe (małe, losowe lub zerowe)
        # Kolejność: [Empty, MaxTile, Snake, Merge]
        self.weights = np.array([0.5, 0.5, 0.5, 0.5])
        self.alpha = 0.00025  # Znacznie mniejsza alpha dla stabilności

        # Macierz Gradientu (Snake)
        base_gradient = np.array([
            [15, 14, 13, 12],
            [ 8,  9, 10, 11],
            [ 7,  6,  5,  4],
            [ 0,  1,  2,  3]
        ])

        self.gradients = []
        for k in range(4):
            rot = np.rot90(base_gradient, k=k)
            self.gradients.append(rot)
            self.gradients.append(np.fliplr(rot))

    def get_features(self, board):
        # 1. Logarytmy planszy
        board_log = np.zeros_like(board, dtype=float)
        mask = board > 0
        board_log[mask] = np.log2(board[mask])

        # --- NORMALIZACJA CECH (Klucz do naprawy eksplozji) ---

        # Cecha 1: Puste Pola (0-16) -> Skalujemy do 0-1
        empty = len(board[board == 0]) / 16.0

        # Cecha 2: Max Tile Log (0-11 dla 2048) -> Skalujemy do 0-1
        max_val = np.max(board_log) / 11.0

        # Cecha 3: Snake Gradient
        # Max teoretyczny wynik to ok. 1500 (gdy cała plansza pełna idealnie)
        # Dzielimy przez 1000, żeby rząd wielkości był podobny do reszty
        gradient_scores = [np.sum(board_log * g) for g in self.gradients]
        best_gradient = max(gradient_scores) / 1000.0



        #1.1 change wektoryzacja
        # --- ZMIANA TUTAJ: Cecha 4: Merges (Wektoryzacja) ---
        # Zamiast wolnych pętli for, używamy szybkiego porównywania macierzy numpy.
        # Porównujemy planszę z jej wersją przesuniętą o 1 w prawo/dół.

        # Czy element [i] == element [i+1] (poziomo) i nie są zerami?
        merges_h = (board[:, :-1] == board[:, 1:]) & (board[:, :-1] != 0)

        # Czy element [i] == element [i+1] (pionowo) i nie są zerami?
        merges_v = (board[:-1, :] == board[1:, :]) & (board[:-1, :] != 0)

        merges = np.sum(merges_h) + np.sum(merges_v)
        merges_norm = merges / 48.0

        # Zwracamy znormalizowany wektor
        return np.array([empty, max_val, best_gradient, merges_norm])
        '''
        # Cecha 4: Merges (0-48) -> Skalujemy do 0-1
        merges = 0
        for r in range(4):
            for c in range(4):
                if c < 3 and board[r, c] != 0 and board[r, c] == board[r, c+1]:
                    merges += 1
                if r < 3 and board[r, c] != 0 and board[r, c] == board[r+1, c]:
                    merges += 1
        merges_norm = merges / 48.0

        # Zwracamy znormalizowany wektor
        return np.array([empty, max_val, best_gradient, merges_norm])
'''

    def evaluate(self, board):
        return np.dot(self.weights, self.get_features(board))

    def update_weights(self, features_state, td_error):
        # GRADIENT CLIPPING: Zapobiegamy zbyt dużej zmianie w jednym kroku
        # Jeśli błąd jest gigantyczny, przycinamy go do zakresu [-10, 10]
        td_error_clipped = np.clip(td_error, -10, 10)

        delta = self.alpha * td_error_clipped * features_state
        self.weights += delta

    def get_expected_value(self, board):
        empty_cells = list(zip(*np.where(board == 0)))
        if not empty_cells:
            return self.evaluate(board)

        if len(empty_cells) > 3:
            # Losujemy 3 próbki dla szybkości
            indices = np.random.choice(len(empty_cells), 3, replace=False)
            sample_cells = [empty_cells[i] for i in indices]
        else:
            sample_cells = empty_cells

        total_val = 0
        for r, c in sample_cells:
            # 2 (90%)
            board[r, c] = 2
            v2 = self.evaluate(board)
            # 4 (10%)
            board[r, c] = 4
            v4 = self.evaluate(board)
            board[r, c] = 0 # Backtrack

            total_val += (0.9 * v2 + 0.1 * v4)

        return total_val / len(sample_cells)