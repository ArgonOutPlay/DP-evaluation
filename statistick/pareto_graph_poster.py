import matplotlib.pyplot as plt


# PLOT PARETO FRONT GRAPH

def pareto_front(points):
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if (q[0] >= p[0] and q[1] <= p[1]) and (q != p):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return sorted(pareto, key=lambda x: x[0])


def plot_multi_metric_pareto():
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "legend.fontsize": 20
    })

    data = {
        "Relevance odpovědi": [
            ("4.1-mini", 0.0757, 36),
            ("4o-mini", 0.0004, 62),
            ("5.1", 0.1159, 33),
            ("5.4-mini", 0.1046, 32),
            ("5.4-nano", 0.0970, 24),
        ],
        "Kontextová dostatečnost": [
            ("4.1-mini", -0.0030, 15),
            ("4o-mini", 0.0109, 17),
            ("5.1", 0.0030, 15),
            ("5.4-mini", 0.0366, 32),
            ("5.4-nano", -0.0129, 12),
        ],
        "Věrohodnost": [
            ("4.1-mini", 0.0226, 50),
            ("4o-mini", 0.0421, 36),
            ("5.1", 0.0089, 51),
            ("5.4-mini", 0.0012, 54),
            ("5.4-nano", -0.0450, 65),
        ]
    }

    colors = {
        "Relevance odpovědi": "tab:blue",
        "Kontextová dostatečnost": "tab:red",
        "Věrohodnost": "tab:green"
    }

    marker_map = {
        "4.1-mini": "o",
        "4o-mini": "s",
        "5.1": "^",
        "5.4-mini": "D",
        "5.4-nano": "x"
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for metric, values in data.items():
        points = [(d, n) for _, d, n in values]
        pareto = pareto_front(points)

        for label, d, n in values:
            ax.scatter(d, n,
                       color=colors[metric],
                       marker=marker_map[label],
                       s=90)

        if len(pareto) > 1:
            px, py = zip(*pareto)
            ax.plot(px, py,
                    linestyle='--',
                    linewidth=2.5,
                    color=colors[metric])

    metric_handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=metric)
        for metric, color in colors.items()
    ]

    legend1 = ax.legend(handles=metric_handles, title="Metrika", loc="upper right")
    ax.add_artist(legend1)
    ax.set_xlabel("Efekt (Δ)")
    ax.set_ylabel("Počet vzorků N")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_multi_metric_pareto()