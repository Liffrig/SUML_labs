import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
import pickle

# ===========================
# load_linear_regression_model:
# 1. Otwieramy plik pickle i przypisujemy do zmiennej, która będzie przechowywała model
# 2. Zwracamy model
#
# linear_predict:
# 1. Dokonujemy przewidywania wartości y na podstawie dostarczonego x
# 2. Zwracamy y
# ===========================

def load_linear_regression_model(file_path: str = 'our_model.pkl') -> LinearRegression:
	path = Path(file_path)

	if path.exists() and file_path.endswith('.pkl'):
		with open(path, 'rb') as f:
			imported_model = pickle.load(f)
		return imported_model
	else:
		raise FileNotFoundError("Nie znaleziono pliku .pkl")


def linear_predict(model:LinearRegression, x: int | list[int] ) -> np.ndarray:
	if isinstance(x, int):
		X = np.array([[x]])
	else:
		X = np.array(x).reshape(-1, 1)
	return model.predict(X)

# ================
# 1. Otwieramy plik csv i przypisujemy zawartość do zmiennej w formie DataFrame
# 2. Dodajemy na końcu nowy wiersz z parametrami x i y, zapisujemy zaktualizowany CSV
# 3. Przekształcamy kolumny w łańcuchy numpy o odpowiednim kształcie
# 4. Przypisujemy do zmiennej model regresji liniowej
# 5. Dokonujemy treningu (fit) przy pomocy nowych danych (łańcuchów Numpy)
# 6. Zmodyfikowany model zapisujemy w pliku pickle o nazwie our_model.pkl (istniejący nadpisujemy)
# ================

def update_and_retrain(file_path: str, x: int | float, y: int | float, model_path: str = 'our_model.pkl') -> None:
	csv_path = Path(file_path)
	if not csv_path.exists() or not file_path.endswith('.csv') :
		raise FileNotFoundError(f"Nie znaleziono pliku CSV: {file_path}")

	pkl_path = Path(model_path)
	if not model_path.endswith('.pkl') or not pkl_path.parent.exists():
		raise ValueError(f"Oczekiwano pliku .pkl, otrzymano: {model_path}")

	df = pd.read_csv(csv_path)
	new_row = pd.DataFrame({df.columns[0]: [x], df.columns[1]: [y]})
	df = pd.concat([df, new_row], ignore_index=True)

	df.to_csv(csv_path, index=False)

	x_data = df.iloc[:, 0].to_numpy().reshape(-1, 1)
	y_data = df.iloc[:, 1].to_numpy()

	model = LinearRegression()
	model.fit(x_data, y_data)

	with open(pkl_path, 'wb') as f:
		pickle.dump(model, f)


def main() -> None:
	MODEL_PATH = 'our_model.pkl'
	CSV_PATH = '10_points.csv'

	# --- Scenariusz 1: wczytanie modelu ---
	print("=== Scenariusz 1: wczytanie modelu ===")
	try:
		model = load_linear_regression_model(MODEL_PATH)
		print(f"Model wczytany poprawnie: {model}")
	except FileNotFoundError as e:
		print(f"Błąd: {e}")

	# --- Scenariusz 2: błędna ścieżka do modelu ---
	print("\n=== Scenariusz 2: błędna ścieżka do modelu ===")
	try:
		load_linear_regression_model('nieistniejacy.pkl')
	except FileNotFoundError as e:
		print(f"Oczekiwany błąd: {e}")

	# --- Scenariusz 3: predykcja dla pojedynczej wartości int ---
	print("\n=== Scenariusz 3: predykcja dla pojedynczej wartości ===")
	model = load_linear_regression_model(MODEL_PATH)
	wynik = linear_predict(model, 5)
	print(f"Predykcja dla x=5: {wynik}")

	# --- Scenariusz 4: predykcja dla listy wartości ---
	print("\n=== Scenariusz 4: predykcja dla listy wartości ===")
	wyniki = linear_predict(model, [1, 5, 10, 20])
	print(f"Predykcja dla x=[1, 5, 10, 20]: {wyniki}")

	# --- Scenariusz 5: aktualizacja CSV i retrain modelu ---
	print("\n=== Scenariusz 5: aktualizacja danych i retrain ===")
	try:
		update_and_retrain(CSV_PATH, x=11.0, y=22.0, model_path=MODEL_PATH)
		print("Dane dodane, model przetrenowany i zapisany.")
		model = load_linear_regression_model(MODEL_PATH)
		wynik = linear_predict(model, 11)
		print(f"Predykcja po retrain dla x=11: {wynik}")
	except (FileNotFoundError, ValueError) as e:
		print(f"Błąd: {e}")

	# --- Scenariusz 6: błędna ścieżka CSV ---
	print("\n=== Scenariusz 6: błędna ścieżka CSV ===")
	try:
		update_and_retrain('nieistniejacy.csv', x=1, y=2)
	except FileNotFoundError as e:
		print(f"Oczekiwany błąd: {e}")

	# --- Scenariusz 7: błędna ścieżka pkl w update_and_retrain ---
	print("\n=== Scenariusz 7: błędne rozszerzenie pliku modelu ===")
	try:
		update_and_retrain(CSV_PATH, x=1, y=2, model_path='model.txt')
	except ValueError as e:
		print(f"Oczekiwany błąd: {e}")



if __name__ == '__main__':
	main()


