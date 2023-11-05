import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx


class GridWorld:
    def __init__(
        self,
        grid_size=(100, 100),
        n_locations=10,
        n_locales=5,
        locale_method="rectangular",
    ):
        """Initialize the GridWorld.

        Parameters
        ----------
        grid_size : tuple of int, optional
            Dimensions of the grid. Default is (100, 100).
        n_locations : int, optional
            Number of random locations to be selected. Default is 10.
        n_locales : int, optional
            Number of locales. Default is 5.
        locale_method : str, optional
            Method for determining locale boundaries. Default is 'rectangular'.

        Attributes
        ----------
        grid_size : tuple of int
            Dimensions of the grid.
        n_locations : int
            Number of random locations.
        n_locales : int
            Number of locales.
        locales : list of tuple
            List of locales' boundaries.
        locations : list of tuple
            List of selected random locations.
        """
        self.grid_size = grid_size
        self.n_locations = n_locations
        self.n_locales = n_locales
        self.locales = self._create_locales(locale_method)
        self.locations = self._select_random_locations()

    def _create_locales(self, method):
        """Create locales based on the specified method.

        Parameters
        ----------
        method : str
            Method to create locales. Supported methods: 'rectangular'.

        Returns
        -------
        list of tuple
            List of locales' boundaries.

        Raises
        ------
        NotImplementedError
            If an unsupported locale creation method is provided.
        """
        if method == "rectangular":
            return self._create_rectangular_locales()
        else:
            raise NotImplementedError

    def _create_rectangular_locales(self):
        """Private method to create rectangular-shaped locales.

        Divides the grid into rectangles based on the specified number of locales.

        Returns
        -------
        list of tuple
            List of rectangular locales' boundaries.
        """
        locales = []
        x_divisions = int(np.sqrt(self.n_locales))
        y_divisions = self.n_locales // x_divisions

        x_step = self.grid_size[0] // x_divisions
        y_step = self.grid_size[1] // y_divisions

        for i in range(x_divisions):
            for j in range(y_divisions):
                x_start, x_end = i * x_step, (i + 1) * x_step
                y_start, y_end = j * y_step, (j + 1) * y_step
                locales.append(((x_start, y_start), (x_end, y_end)))

        return locales

    def _select_random_locations(self):
        """Private method to select random locations on the grid.

        Randomly select n_locations on the grid.

        Returns
        -------
        list of tuple
            List of selected random locations.
        """
        locations = [
            (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
            for _ in range(self.n_locations)
        ]
        return locations

    def get_locale(self, coordinate):
        """
        Given a coordinate, determine which locale it belongs to.

        Parameters:
        - coordinate: tuple(int, int) representing the x and y coordinates.

        Returns:
        - tuple representing the boundary of the locale the coordinate belongs to.
        """
        for locale in self.locales:
            if (
                locale[0][0] <= coordinate[0] <= locale[1][0]
                and locale[0][1] <= coordinate[1] <= locale[1][1]
            ):
                return locale
        return None

    def display_grid(self):
        """
        Display the grid with the selected locations.
        """
        grid = np.zeros(self.grid_size)

        for x, y in self.locations:
            grid[x][y] = 1
        return grid

    def generate_events(
        self, n_events, destination_prob_method="uniform", return_distances=False
    ):
        """Generate events representing trips between locations.

        Parameters
        ----------
        n_events : int
            Number of events/trips to generate.
        destination_prob_method : str, optional
            Method to generate destination probabilities. Options are 'uniform' or 'inverse_distance'. Default is 'uniform'.
        return_distances : bool, optional
            If True, return distances between events instead of their coordinates. Default is False.

        Returns
        -------
        list of tuple or list of float
            List of events or distances between events.
        """
        events = []
        for _ in range(n_events):
            source_idx = np.random.choice(self.n_locations)
            source = self.locations[source_idx]

            if destination_prob_method == "uniform":
                probs = np.ones(self.n_locations) / self.n_locations
            elif destination_prob_method == "inverse_distance":
                distances = np.array(
                    [
                        np.linalg.norm(np.array(source) - np.array(dest))
                        for dest in self.locations
                    ]
                )
                probs = 1 / (1 + distances)
            else:
                raise ValueError(
                    f"Unknown destination_prob_method: {destination_prob_method}"
                )

            probs[source_idx] = 0  # source should not be the destination
            probs /= probs.sum()

            dest_idx = np.random.choice(self.n_locations, p=probs)
            dest = self.locations[dest_idx]

            if return_distances:
                distance = np.linalg.norm(np.array(source) - np.array(dest))
                events.append(distance)
            else:
                events.append((source, dest))

        return events

    def get_locale_distance_range(self, locale1, locale2):
        """Determine the minimum and maximum Euclidean distance between two locales.

        Parameters
        ----------
        locale1, locale2 : tuple
            Tuples representing the boundaries of the locales.

        Returns
        -------
        tuple of float
            Minimum and maximum distances between the two locales.
        """
        if locale1 == locale2:
            # min distance is 0, max distance is the diagonal
            width = locale1[1][0] - locale1[0][0]
            height = locale1[1][1] - locale1[0][1]
            return (0, np.sqrt(width**2 + height**2))

        x_distances = self._get_boundary_distances(
            locale1[0][0], locale1[1][0], locale2[0][0], locale2[1][0]
        )
        y_distances = self._get_boundary_distances(
            locale1[0][1], locale1[1][1], locale2[0][1], locale2[1][1]
        )

        min_distance = np.sqrt(min(x_distances) ** 2 + min(y_distances) ** 2)
        max_distance = np.sqrt(max(x_distances) ** 2 + max(y_distances) ** 2)

        return (min_distance, max_distance)

    def _get_boundary_distances(self, a_start, a_end, b_start, b_end):
        """Private helper method to determine the distances between boundaries of two intervals.

        Parameters
        ----------
        a_start, a_end : float
            Start and end points of the first interval.
        b_start, b_end : float
            Start and end points of the second interval.

        Returns
        -------
        list of float
            Minimum and maximum distances between the boundaries of the two intervals.
        """
        if a_end < b_start:
            return [b_start - a_end, b_end - a_start]
        elif b_end < a_start:
            return [a_start - b_end, a_end - b_start]
        else:
            # overlapping intervals
            return [0, max(a_end, b_end) - min(a_start, b_start)]

    def sample_event_distances(self, n_events, destination_prob_method="uniform"):
        """Sample events and get distances and locale distance intervals.

        Parameters
        ----------
        n_events : int
            Number of events/trips to generate.
        destination_prob_method : str, optional
            Method to generate destination probabilities.
            Default is 'uniform'

        Returns
        -------
        tuple of list
            List of actual event distances, list of possible distance intervals for each event,
            and optionally, list of distances between locale centroids.
        """
        events = self.generate_events(
            n_events, destination_prob_method, return_distances=False
        )
        actual_distances = []
        locale_intervals = []

        for event in events:
            source, dest = event
            distance = np.linalg.norm(np.array(source) - np.array(dest))
            actual_distances.append(distance)

            source_locale = self.get_locale(source)
            dest_locale = self.get_locale(dest)
            min_dist, max_dist = self.get_locale_distance_range(
                source_locale, dest_locale
            )
            locale_intervals.append((min_dist, max_dist))

        return actual_distances, locale_intervals

    def average_locale_area_ratio(self):
        """Compute the ratio of the average locale area to the total area of GridWorld.

        Returns
        -------
        float
            Ratio of average locale area to total GridWorld area.
        """
        total_area = self.grid_size[0] * self.grid_size[1]
        locale_areas = [
            (locale[1][0] - locale[0][0]) * (locale[1][1] - locale[0][1])
            for locale in self.locales
        ]
        avg_locale_area = sum(locale_areas) / len(locale_areas)
        return avg_locale_area / total_area

    def visualize_grid(self, fname=None):
        """Visualize the GridWorld using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])

        for locale in self.locales:
            x_start, y_start = locale[0]
            width = locale[1][0] - x_start
            height = locale[1][1] - y_start

            rect = patches.Rectangle(
                (x_start, y_start),
                width,
                height,
                linewidth=1,
                edgecolor="blue",
                facecolor="none",
            )
            ax.add_patch(rect)

        x_coords, y_coords = zip(*self.locations)
        ax.scatter(x_coords, y_coords, c="red", s=50)

        ax.set_title("GridWorld Visualization")

        if fname is not None:
            plt.savefig(fname)

        plt.show()

    def generate_event_network(self, n_events, destination_prob_method="uniform"):
        """Generate a network (graph) using networkx where nodes are locations and edges
        represent events.

        Parameters
        ----------
        n_events : int
            Number of events/trips to generate.
        destination_prob_method : str, optional
            Method to generate destination probabilities. Default is 'uniform'.

        Returns
        -------
        networkx.Graph
            Graph representing events. The nodes are locations. Each edge represents
            the number of events between two locations, and the edge weight is the count of these events.
        """
        events = self.generate_events(
            n_events, destination_prob_method, return_distances=False
        )
        G = nx.Graph()

        for location in self.locations:
            G.add_node(location)

        for source, dest in events:
            if G.has_edge(source, dest):
                G[source][dest]["weight"] += 1
            else:
                G.add_edge(source, dest, weight=1)

        return G

    def visualize_grid_on_ax(self, ax):
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])

        for locale in self.locales:
            x_start, y_start = locale[0]
            width = locale[1][0] - x_start
            height = locale[1][1] - y_start

            rect = patches.Rectangle(
                (x_start, y_start),
                width,
                height,
                linewidth=1,
                edgecolor="blue",
                facecolor="none",
            )
            ax.add_patch(rect)

        x_coords, y_coords = zip(*self.locations)
        ax.scatter(x_coords, y_coords, c="red", s=50)
