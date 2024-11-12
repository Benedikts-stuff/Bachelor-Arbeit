import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import pylab as pl
import seaborn as sns
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from tqdm import tqdm
import time  # Beispielweise zum Simulieren von Berechnungszeit

from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
#from c_b_thompson import ThompsonSamplingContextualBandit
from lin_ucb import LinUCB
from olr_e_greedy import EpsilonGreedyContextualBandit
from w_ucb import OmegaUCB

from multiprocessing import Pool
from time import sleep
import time





class Paralell_Experiment:
    def __init__(self, functions, arguments, jobs):
       self.functions = functions
       self.arguments = arguments
       self.njobs = jobs

    # Funktion, die einen bestimmten Zeitraum wartet und dann den Wert zurückgibt
    def _wait(i):
        sleep(i)  # Warte für 'i' Sekunden
        return i  # Gib den Wert 'i' zurück

    # Funktion zur asynchronen Ausführung einer beliebigen Funktion in mehreren Prozessen
    def run_async(self, function, args_list, njobs, sleep_time_s=0.01):
        # Erstelle einen Pool von 'njobs' parallelen Prozessen
        pool = Pool(njobs)

        # Starte asynchrone Ausführung der Funktion mit verschiedenen Argumenten
        # `pool.apply_async` führt `function` asynchron mit den jeweiligen Argumenten aus
        results = [pool.apply_async(function, args=args) for args in args_list]

        # Überprüfe wiederholt, ob alle asynchronen Aufgaben abgeschlossen sind
        while not all(future.ready() for future in results):
            sleep(sleep_time_s)  # Warte eine kurze Zeit bevor erneut geprüft wird

        # Hole die Ergebnisse aller asynchronen Aufgaben ab, nachdem sie abgeschlossen sind
        results = [result.get() for result in results]

        # Schließe den Pool, sodass keine weiteren Aufgaben hinzugefügt werden können
        pool.close()

        return results  # Gib die Ergebnisse der asynchronen Aufgaben zurück

    def execute_parallel(self):
        # Hauptprogramm
        if __name__ == '__main__':
            #njobs = 3  # Anzahl der parallelen Prozesse, die genutzt werden sollen
            # Liste von Argumenten (hier: Wartezeiten), die an `_wait` übergeben werden
            delay = [[i] for i in range(4)]
            # Führe die Funktion `_wait` asynchron mit den Argumenten aus und speichere das Ergebnis
            result = self.run_async(self._wait, delay, self.njobs)
            print(result)  # Ausgabe: [0, 1, 2] (da `_wait(i)` jeweils den Wert `i` zurückgibt)
