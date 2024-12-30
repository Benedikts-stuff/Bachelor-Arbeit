import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Daten laden
with open('./plot_data.json', 'r') as file:
    list_all = json.load(file)

# Anzahl der Runden und Arme ermitteln
num_rounds = len(list_all[0])  # Anzahl der Runden (1000)
num_arms = len(list_all)  # Anzahl der Arme (z. B. 3)

# Interaktiven Plot erstellen
fig = make_subplots(rows=1, cols=1)

# Initiale Daten für die Kerzen (erste Runde)
initial_data = []
for arm_idx, arm_data in enumerate(list_all):
    mean = arm_data[0][0]
    offset = arm_data[0][1]
    initial_data.append(go.Candlestick(
        x=[arm_idx],
        open=[mean + offset],
        close=[mean - offset],
        high=[mean + offset],
        low=[mean - offset],
        name=f'Arm {arm_idx + 1}'
    ))

# Frames für Animation erstellen
frames = []
for round_idx in range(1, num_rounds):  # Beginne bei Runde 1, da Runde 0 die Initialisierung ist
    data = []
    for arm_idx, arm_data in enumerate(list_all):
        mean = arm_data[round_idx][0]
        offset = arm_data[round_idx][1]

        # Kerzenplot erstellen (Kerzen nebeneinander verschieben)
        data.append(go.Candlestick(
            x=[arm_idx],  # Position des Arms (nebeneinander)
            open=[mean + offset],  # Upper bound
            close=[mean - offset],  # Lower bound
            high=[mean + offset],  # Upper bound
            low=[mean - offset],  # Lower bound
            name=f'Arm {arm_idx + 1}'
        ))
    frames.append(go.Frame(data=data, name=str(round_idx)))

# Layout des Plots
fig.update_layout(
    xaxis=dict(
        tickvals=list(range(num_arms)),  # Position der Arme auf der x-Achse
        ticktext=[f'Arm {i + 1}' for i in range(num_arms)],
        title='Arme',
        range=[-0.5, num_arms - 0.5]  # Fixiere die x-Achse, damit sie nicht verschiebt
    ),
    yaxis=dict(title='Wert',
               range=[0,2]),
    title='Konfidenzintervalle der Arme über die Runden',
    updatemenus=[dict(
        type='buttons',
        active=True,
        buttons=[
            dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=1, redraw=True),
                                 fromcurrent=True, mode="immediate")]
            ),
            dict(
                label='Pause',
                method='animate',
                args=[[None], dict(frame=dict(duration=0, redraw=False))]
            )
        ]
    )],
    sliders=[dict(
        steps=[dict(
            method='animate',
            args=[[str(i)], dict(mode='immediate', frame=dict(duration=1, redraw=True))],
            label=f'Runde {i}'
        ) for i in range(num_rounds)],
        currentvalue=dict(prefix='Runde: '),
        pad=dict(t=50),
        active=True,
        len=0.9,  # Slider Länge
        xanchor="center",
        yanchor="bottom"
    )]
)

# Initiale Daten und Frames hinzufügen
fig.add_traces(initial_data)
fig.frames = frames

# Plot als HTML speichern
fig.write_html("interactive_plot.html")

# Den Plot im Standardbrowser öffnen
import webbrowser

webbrowser.open("interactive_plot.html")

# Plot anzeigen
fig.show()
