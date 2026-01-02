import tkinter as tk
from game_2048 import Game2048
import time
from ai_player import AIPlayer

class Game2048App:
    def __init__(self, root, size=4):
        self.root = root
        self.size = size
        self.game = Game2048(size)

        # --- DODANO: Inicjalizacja AI ---
        self.ai = AIPlayer()

        # Próba wczytania modelu
        loaded_episode = self.ai.load_model("ai_2048_save.pkl")

        # --- TEST PRAWDY: Co siedzi w głowie AI? ---
        print("-" * 40)
        if loaded_episode > 0:
            print(f"✅ SUKCES: Wczytano WYTRENOWANY model (Epizod {loaded_episode})")
        else:
            print("⚠️ UWAGA: Gram na DOMYŚLNYCH ustawieniach (Brak pliku)")

        # Wyświetlamy oba zestawy wag (Dual-Weights)
        # Dzięki temu widzisz, czy AI ma "dwubiegunowość"
        print(f"🧠 Wagi NORMAL (Spokój): {self.ai.weights_normal}")
        print(f"🔥 Wagi PANIC  (Kryzys): {self.ai.weights_panic}")
        print("-" * 40)
        # -------------------------------------------

        self.ai_running = False
        self.sim_game = Game2048(size)  # Do symulacji ruchów

        self.root.title("2048 - Tkinter (AI Dual-Mode)")
        self.root.resizable(False, False)

        # --- Konfiguracja i stałe ---
        self.TILE_SIZE = 100
        self.PADDING = 12
        self.CANVAS_SIZE = self.size * (self.TILE_SIZE + self.PADDING) + self.PADDING
        self.BG_COLOR = '#bbada0'
        self.GAME_FRAME_BG = '#faf8ef'
        self.TILE_COLORS = self._get_colors()

        self.animation_in_progress = False
        self.game_over_shown = False
        self.game_over_popup = None

        main_container = tk.Frame(root, bg=self.GAME_FRAME_BG)
        main_container.pack(fill="both", expand=True, padx=0, pady=0)

        score_frame = tk.Frame(main_container, bg=self.GAME_FRAME_BG, padx=10, pady=10)
        score_frame.pack(side="top", fill="x")

        tk.Label(score_frame, text="WYNIK:", font=('Helvetica', 16), fg='#776e65', bg=self.GAME_FRAME_BG).pack(side="left")
        self.score_label = tk.Label(score_frame, text="0", font=('Helvetica', 20, 'bold'), fg='#776e65', bg=self.GAME_FRAME_BG)
        self.score_label.pack(side="left", padx=10)

        self.canvas = tk.Canvas(main_container,
                                width=self.CANVAS_SIZE,
                                height=self.CANVAS_SIZE,
                                bg=self.BG_COLOR,
                                highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        self.tile_objects = {}
        self.tile_count = 0

        ctrl_frame = tk.Frame(main_container, bg=self.GAME_FRAME_BG)
        ctrl_frame.pack(pady=(0, 10))

        self._create_controls(ctrl_frame)

        self.root.bind("<Key>", self.key_handler)
        self.draw_grid()
        self.update_board(animate=False)

    # --- KONFIGURACJA KOLORÓW ---
    def _get_colors(self):
        return {
            0: {'bg': '#cdc1b4', 'fg': '#776e65', 'font_size': 32},
            2: {'bg': '#eee4da', 'fg': '#776e65', 'font_size': 32},
            4: {'bg': '#ede0c8', 'fg': '#776e65', 'font_size': 32},
            8: {'bg': '#f2b179', 'fg': 'white', 'font_size': 32},
            16: {'bg': '#f59563', 'fg': 'white', 'font_size': 32},
            32: {'bg': '#f67c5f', 'fg': 'white', 'font_size': 32},
            64: {'bg': '#f65e3b', 'fg': 'white', 'font_size': 32},
            128: {'bg': '#edcf72', 'fg': 'white', 'font_size': 28},
            256: {'bg': '#edcc61', 'fg': 'white', 'font_size': 28},
            512: {'bg': '#edc850', 'fg': 'white', 'font_size': 28},
            1024: {'bg': '#edc53f', 'fg': 'white', 'font_size': 24},
            2048: {'bg': '#edc22e', 'fg': 'white', 'font_size': 24},
            4096: {'bg': '#a2d149', 'fg': 'white', 'font_size': 24},
            8192: {'bg': '#3c3a32', 'fg': 'white', 'font_size': 24}
        }

    # --- TWORZENIE KONTROLEK ---
    def _create_controls(self, ctrl_frame):
        self.restart_btn = tk.Button(
            ctrl_frame, text="🔄 Restart", command=self.restart_game,
            font=("Helvetica", 14, "bold"), bg="#8f7a66", fg="white",
            activebackground="#a68a72", activeforeground="white",
            relief="flat", bd=0, padx=15, pady=5
        )
        self.restart_btn.pack(side="left", padx=10)

        # --- Przycisk AI ---
        self.ai_btn = tk.Button(
            ctrl_frame, text="🤖 Gra AI", command=self.toggle_ai,
            font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="white",
            activebackground="#45a049", activeforeground="white",
            relief="flat", bd=0, padx=15, pady=5
        )
        self.ai_btn.pack(side="left", padx=10)

        self.quit_btn = tk.Button(
            ctrl_frame, text="❌ Wyjście", command=self.root.destroy,
            font=("Helvetica", 14, "bold"), bg="#8f7a66", fg="white",
            activebackground="#a68a72", activeforeground="white",
            relief="flat", bd=0, padx=15, pady=5
        )
        self.quit_btn.pack(side="left", padx=10)

    # --- RYSOWANIE SIATKI ---
    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.size):
            for j in range(self.size):
                x1, y1 = self._get_coords(j, i)
                x2 = x1 + self.TILE_SIZE
                y2 = y1 + self.TILE_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill='#cdc1b4', outline='')

    # --- NARZĘDZIA KOORDYNATÓW ---
    def _get_coords(self, col, row):
        x = col * (self.TILE_SIZE + self.PADDING) + self.PADDING
        y = row * (self.TILE_SIZE + self.PADDING) + self.PADDING
        return x, y

    # --- RYSOWANIE KAFELKA ---
    def _draw_tile(self, value, col, row, tile_id=None, is_new=False):
        x, y = self._get_coords(col, row)
        colors = self.TILE_COLORS.get(value, self.TILE_COLORS[0])

        if tile_id is not None and self.canvas.find_withtag(f"tile_{tile_id}"):
            self.canvas.delete(f"tile_{tile_id}")

        rect_id = self.canvas.create_rectangle(
            x, y, x + self.TILE_SIZE, y + self.TILE_SIZE,
            fill=colors['bg'], outline='',
            tags=f"tile_{tile_id}" if tile_id is not None else "new_tile"
        )

        text_content = str(value) if value else ""
        text_id = self.canvas.create_text(
            x + self.TILE_SIZE / 2, y + self.TILE_SIZE / 2,
            text=text_content, fill=colors['fg'],
            font=('Helvetica', colors['font_size'], 'bold'),
            tags=f"tile_{tile_id}" if tile_id is not None else "new_tile"
        )

        if tile_id is None:
            self.tile_count += 1
            tile_id = self.tile_count
            self.tile_objects[tile_id] = [value, rect_id, text_id, row, col]

        self.canvas.tag_raise(rect_id)
        self.canvas.tag_raise(text_id)

        return tile_id

    # --- ANIMACJA ---
    def animate_move(self, start_pos, end_pos, tile_id, duration=100):
        if self.animation_in_progress:
            return

        self.animation_in_progress = True

        start_x, start_y = self._get_coords(start_pos[1], start_pos[0])
        end_x, end_y = self._get_coords(end_pos[1], end_pos[0])

        bbox = self.canvas.bbox(self.tile_objects[tile_id][1])
        current_x = (bbox[0] + bbox[2]) / 2 - self.TILE_SIZE / 2
        current_y = (bbox[1] + bbox[3]) / 2 - self.TILE_SIZE / 2

        dx = end_x - current_x
        dy = end_y - current_y

        steps = 10
        delay = duration // steps

        def step(current_step):
            if current_step < steps:
                move_x = dx / steps
                move_y = dy / steps

                self.canvas.move(f"tile_{tile_id}", move_x, move_y)

                self.root.after(delay, lambda: step(current_step + 1))
            else:
                self.animation_in_progress = False
                self.update_board(animate=False)

        step(0)

    # --- ODSWIEŻANIE PLANSZY ---
    def update_board(self, animate=True):
        self.score_label.config(text=str(self.game.score))

        new_tile_objects = {}

        self.canvas.delete("all")
        self.draw_grid()
        self.tile_objects = {}
        self.tile_count = 0

        for r in range(self.size):
            for c in range(self.size):
                value = int(self.game.board[r][c])
                if value != 0:
                    tile_id = self._draw_tile(value, c, r)
                    new_tile_objects[(r, c)] = tile_id

        self.tile_objects = new_tile_objects

    # --- HANDLER KLUCZY ---
    def key_handler(self, event):
        # Blokada klawiszy gdy gra AI
        if self.game_over_shown or self.animation_in_progress or self.ai_running:
            return

        mapping = {
            'Up': 'up', 'Down': 'down', 'Left': 'left', 'Right': 'right',
            'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'
        }

        key = event.keysym
        if key not in mapping and event.char in mapping:
            key = event.char

        if key in mapping:
            _, _, done, changed = self.game.move(mapping[key])

            if changed:
                self.update_board()

            if done:
                self.game_over_shown = True
                self.show_popup()

    # --- RESET GRY ---
    def restart_game(self):
        # Stop AI przy restarcie
        self.ai_running = False
        self.ai_btn.config(text="🤖 Gra AI", bg="#4CAF50")

        if self.game_over_popup:
            self.game_over_popup.destroy()
            self.game_over_popup = None

        self.game_over_shown = False
        self.game.reset()
        self.root.bind("<Key>", self.key_handler)
        self.update_board(animate=False)

    # --- GAME OVER POPUP ---
    def show_popup(self):
        if self.game_over_popup:
            return

        popup = tk.Toplevel(self.root)
        popup.title("Koniec gry")
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()

        tk.Label(
            popup, text="Koniec gry!",
            font=("Helvetica", 22, "bold")
        ).pack(padx=20, pady=10)

        tk.Label(
            popup, text=f"Wynik: {self.game.score}",
            font=("Helvetica", 14)
        ).pack(pady=5)

        btn = tk.Button(
            popup, text="🔄 Restart",
            font=("Helvetica", 14, "bold"),
            bg="#8f7a66", fg="white",
            activebackground="#a68a72",
            relief="flat",
            command=lambda: (popup.destroy(), self.restart_game())
        )
        btn.pack(pady=10)

        self.game_over_popup = popup


    # --- METODY AI ---
    def toggle_ai(self):
        if self.ai_running:
            # Zatrzymaj
            self.ai_running = False
            self.ai_btn.config(text="🤖 Gra AI", bg="#4CAF50")
        else:
            # Uruchom
            self.ai_running = True
            self.ai_btn.config(text="⏹ Stop AI", bg="#f44336")
            self.run_ai_step()

    def run_ai_step(self):
        if not self.ai_running or self.game_over_shown:
            self.ai_running = False
            self.ai_btn.config(text="🤖 Gra AI", bg="#4CAF50")
            return

        # 1. Pobierz dostępne ruchy
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            self.game_over_shown = True
            self.show_popup()
            return

        # 2. Wybierz najlepszy ruch (1-step Lookahead)
        state = self.game.board.copy()
        best_move, best_v = None, -float('inf')

        for move in valid_moves:
            self.sim_game.board = state.copy()
            # AI używa move_without_random, by zobaczyć przyszły stan
            next_s_sim, _, _ = self.sim_game.move_without_random(move)

            # AIPlayer (get_expected_value -> evaluate) sam decyduje,
            # których wag (Normal/Panic) użyć na podstawie planszy next_s_sim
            v = self.ai.get_expected_value(next_s_sim)

            if v > best_v:
                best_v = v
                best_move = move

        # 3. Wykonaj ruch w prawdziwej grze
        if best_move:
            _, _, done, changed = self.game.move(best_move)
            if changed:
                self.update_board(animate=False)

            if done:
                self.ai_running = False
                self.ai_btn.config(text="🤖 Gra AI", bg="#4CAF50")
                self.game_over_shown = True
                self.show_popup()
                return

        # 4. Zaplanuj kolejny krok (50ms)
        self.root.after(50, self.run_ai_step)


# --------------- program start ---------------

if __name__ == "__main__":
    root = tk.Tk()
    app = Game2048App(root)
    root.mainloop()