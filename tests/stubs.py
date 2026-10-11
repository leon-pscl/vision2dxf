"""Lightweight stand-ins for an OpenPattern instance (no heavy deps)."""


class FakePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakePattern:
    """Minimal object exposing the attributes ``extract_pattern_data`` reads."""

    def __init__(self):
        self.Front_vertices = [
            [0, 0],
            [10, 0],
            [10, 20],
            [0, 20],
            [0, 0],
        ]
        self.Back_vertices = [
            [20, 0],
            [30, 0],
            [30, 20],
            [20, 20],
            [20, 0],
        ]
        self.Front_dic = {
            "A": FakePoint(0, 0),
            "Front_Armhole": [[0, 0], [5, 5], [10, 0]],
        }
        self.Back_dic = {"B": FakePoint(30, 20)}
        self.m = {"tour_taille": 62.0, "tour_bassin": 88.0}


class FakeDrawPattern(FakePattern):
    """Adds the drawing methods used by :func:`export_pdf`."""

    def draw_pattern(self, dic_list, vertices_list):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for vertices in vertices_list:
            xs = [p[0] for p in vertices]
            ys = [p[1] for p in vertices]
            ax.plot(xs, ys)
        return fig, ax

    def print_info(self, ax, model=None):
        return ax

    def add_legends(self, ax):
        return None

    def add_scales(self, ax, val=5):
        return None
