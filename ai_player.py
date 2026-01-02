import numpy as np
import math
import pickle
import os


class AIPlayer:
    def __init__(self):
        # --- ZMIANA: Dwa zestawy wag ---
        # [Empty, MaxTile, Snake, Merge]
        self.weights_normal = np.array([0.5, 0.5, 0.5, 0.5]) # Gdy > 4 puste
        self.weights_panic  = np.array([0.5, 0.5, 0.5, 0.5]) # Gdy <= 4 puste

        self.alpha = 0.00025

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
        merges_norm = merges / 10.0

        merges_norm = min(merges_norm, 1.0)

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

    def _calculate_smoothness(self, board):
        """
        Wersja ZWEKTORYZOWANA (Błyskawiczna).
        Zamiast pętli, używamy operacji na całych macierzach.
        """
        # 1. Logarytmy (unikanie zer)
        # Tworzymy maskę dla wartości > 0
        mask = board > 0
        if not np.any(mask):
            return 0

        board_log = np.zeros_like(board, dtype=float)
        # Log2 liczymy tylko tam, gdzie są liczby.
        # Operacja wektorowa - C robi to za jednym zamachem.
        board_log[mask] = np.log2(board[mask])

        smoothness = 0

        # 2. Różnice w poziomie (Kolumny 0,1,2 vs 1,2,3)
        # board_log[:, :-1] to cała plansza bez ostatniej kolumny
        # board_log[:, 1:]  to cała plansza bez pierwszej kolumny
        # Odejmujemy je od siebie równolegle
        diff_x = np.abs(board_log[:, :-1] - board_log[:, 1:])

        # Maska: liczymy tylko tam, gdzie OBA klocki są niezerowe
        mask_x = (board[:, :-1] > 0) & (board[:, 1:] > 0)
        smoothness -= np.sum(diff_x[mask_x])

        # 3. Różnice w pionie (Wiersze 0,1,2 vs 1,2,3)
        diff_y = np.abs(board_log[:-1, :] - board_log[1:, :])
        mask_y = (board[:-1, :] > 0) & (board[1:, :] > 0)
        smoothness -= np.sum(diff_y[mask_y])

        return smoothness

    def _calculate_isolation_penalty(self, board):
        """
        Wersja ZWEKTORYZOWANA.
        Sprawdza izolację bez ani jednej pętli for.
        """
        # Tworzymy mapę boolowską "czy mam sąsiada?"
        has_neighbor = np.zeros(board.shape, dtype=bool)

        # 1. Sprawdzenie poziome (Horizontal Matches)
        # Czy klocek[i] == klocek[i+1] (i nie są zerami)
        match_h = (board[:, :-1] == board[:, 1:]) & (board[:, :-1] != 0)

        # Jeśli nastąpiło dopasowanie, to OBA klocki mają sąsiada
        # Używamy operatora |= (OR), żeby nanosić prawdę na mapę
        has_neighbor[:, :-1] |= match_h  # Lewy klocek z pary ma sąsiada
        has_neighbor[:, 1:]  |= match_h  # Prawy klocek z pary ma sąsiada

        # 2. Sprawdzenie pionowe (Vertical Matches)
        match_v = (board[:-1, :] == board[1:, :]) & (board[:-1, :] != 0)

        has_neighbor[:-1, :] |= match_v # Górny klocek
        has_neighbor[1:, :]  |= match_v # Dolny klocek

        # 3. Kto jest samotny?
        # Klocek musi być niezerowy I (AND) nie mieć sąsiada
        isolated_mask = (board != 0) & (~has_neighbor)

        # Sumujemy True (True = 1, False = 0)
        return np.sum(isolated_mask)

    # --- ZAKTUALIZOWANA METODA: Evaluate ---
    def evaluate(self, board):
        """
        Ocena z wyborem zestawu wag w zależności od fazy gry.
        """
        features = self.get_features(board)
        empty_cells_count = len(board[board == 0])

        # --- WYBÓR MÓZGU ---
        if empty_cells_count < 4:
            # === TRYB PANIKI ===
            # Używamy wag dedykowanych do ratowania sytuacji
            base_score = np.dot(self.weights_panic, features)

            # Instynkty w panice (z poprzedniego kroku)
            smoothness_weight = 2
            #wzrost z 0.8 do 15
            #wzrost z 2.0 do 80
            isolation_weight = 10
        else:
            # === TRYB NORMALNY ===
            # Używamy wag do budowania wyniku
            base_score = np.dot(self.weights_normal, features)

            # Instynkty w spokoju
            #wzrost z 0.1 do 3
            #wzrost z 0.5 do 15
            smoothness_weight = 1
            isolation_weight = 5

        # Obliczenie instynktów (Smoothness + Isolation)
        smoothness = self._calculate_smoothness(board)
        isolation = self._calculate_isolation_penalty(board)

        final_score = base_score + \
                      (smoothness * smoothness_weight) - \
                      (isolation * isolation_weight)

        return final_score

    def update_weights(self, features_state, td_error):
        """Aktualizuje odpowiedni zestaw wag na podstawie stanu."""

        # Odzyskujemy liczbę pustych pól z cech (cecha nr 0 to znormalizowane puste pola)
        # features[0] = empty_count / 16.0
        empty_ratio = features_state[0]
        empty_count = empty_ratio * 16.0

        td_error_clipped = np.clip(td_error, -10, 10)
        delta = self.alpha * td_error_clipped * features_state

        # --- DECYZJA: Który mózg się uczył? ---
        # Musimy użyć progu np. 3.99 zamiast 4, żeby uniknąć błędów zmiennoprzecinkowych
        if empty_count < 3.99:
            self.weights_panic += delta
            self.weights_panic = np.maximum(self.weights_panic, 0.0)
        else:
            self.weights_normal += delta
            self.weights_normal = np.maximum(self.weights_normal, 0.0)

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


    def save_model(self, filename, episode_count):
        """Zapisuje OBA zestawy wag."""
        data = {
            'weights_normal': self.weights_normal,
            'weights_panic': self.weights_panic,
            'episode': episode_count
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"--> Zapisano checkpoint (Epizod: {episode_count})")

    def load_model(self, filename):
        """Ładuje OBA zestawy wag."""
        if not os.path.exists(filename):
            return 0
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

                # Obsługa wstecznej kompatybilności (gdyby stary plik miał tylko 'weights')
                if 'weights_normal' in data:
                    self.weights_normal = data['weights_normal']
                    self.weights_panic = data['weights_panic']
                else:
                    # Stary save -> przypisz stary zestaw do obu
                    old_weights = data['weights']
                    self.weights_normal = old_weights.copy()
                    self.weights_panic = old_weights.copy()
                    print("Konwersja starego zapisu na Dual-Weights...")

                return data['episode']
        except Exception as e:
            print(f"Błąd odczytu zapisu: {e}")
            return 0

