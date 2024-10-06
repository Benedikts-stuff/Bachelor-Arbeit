import matplotlib.pyplot as plt

class PlotUtils:
    @staticmethod
    def plot_all(cumulative_reward, arm_counts, arm_reward_means, regret):
        plt.figure(figsize=(15, 10))

        # Subplot 1: Kumulative Belohnung
        plt.subplot(3, 2, 1)
        plt.plot(cumulative_reward)
        plt.title("Kumulative Belohnung über die Zeit")
        plt.xlabel("Runden")
        plt.ylabel("Kumulative Belohnung")
        plt.grid()

        # Subplot 2: Häufigkeit der Arm-Auswahl
        plt.subplot(3, 2, 2)
        plt.bar(range(len(arm_counts)), arm_counts)
        plt.title("Häufigkeit der Arm-Auswahl")
        plt.xlabel("Arm (Anzeige)")
        plt.ylabel("Anzahl der Ziehungen")
        plt.grid()

        # Subplot 3: Durchschnittliche Belohnung pro Arm
        plt.subplot(3, 2, 3)
        plt.bar(range(len(arm_reward_means)), arm_reward_means)
        plt.title("Durchschnittliche Belohnung pro Arm")
        plt.xlabel("Arm (Anzeige)")
        plt.ylabel("Durchschnittliche Belohnung")
        plt.grid()

        # Subplot 4: Kumulative Regret über die Zeit
        plt.subplot(3, 2, 4)
        plt.plot(regret)
        plt.title("Kumulative Regret über die Zeit")
        plt.xlabel("Runden")
        plt.ylabel("Regret")
        plt.grid()

        plt.tight_layout()
        plt.show()