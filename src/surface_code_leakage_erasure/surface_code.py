# This file defines the surface code layout, including helper classes
# SurfaceCodeLayout is very slow but only need to do it once per distance

class Coord(tuple):
    # qubit or plaquette coordinate
    def __new__(cls, x, y=None):
        # Accept either Coord(x, y) or Coord((x, y)) to be pickle-friendly.
        if y is None:
            try:
                x0, y0 = x
            except Exception as e:
                raise TypeError("Coord requires two values (x, y) or a single iterable of length 2") from e
        else:
            x0, y0 = x, y
        return super().__new__(cls, (x0, y0))

    def __repr__(self):
        return f"Coord({self[0]}, {self[1]})"

    def __add__(self, other):
        if isinstance(other, Coord):
            return Coord(self[0] + other[0], self[1] + other[1])
        else:
            raise TypeError(f"Unsupported operand type(s) {type(self)} and {type(other)} for +")

    def __sub__(self, other):
        if isinstance(other, Coord):
            return Coord(self[0] - other[0], self[1] - other[1])
        else:
            raise TypeError(f"Unsupported operand type(s) {type(self)} and {type(other)} for -")

    def __mul__(self, other):
        if isinstance(other, int):
            return Coord(self[0]*other, self[1]*other)
        else:
            raise TypeError(f"Unsupported operand type(s) {type(self)} and {type(other)} for *")

    def __rmul__(self, other):
        return self * other

    def __lt__(self, other):
        return (self[0] < other[0]) or (self[0] == other[0] and self[1] < other[1])

    def isinside(self, top_left: 'Coord', bottom_right: 'Coord') -> bool:
        return (top_left[0] <= self[0] <= bottom_right[0]) and (top_left[1] <= self[1] <= bottom_right[1])

class Qubit:
    def __init__(self, index: int, coord: Coord):
        self.index = index
        self.coord = coord

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.index == other.index) and (self.coord == other.coord)

    def __hash__(self):
        # Hash must be consistent with __eq__. Use the immutable attributes.
        return hash((self.index, self.coord))

    def __lt__(self, other):
        return self.index < other.index

    def __repr__(self):
        return f"Qubit(index={self.index}, coord={self.coord})"

class DataQubit(Qubit):
    def __init__(self, index: int, coord: Coord):
        super().__init__(index, coord)
        self.ordered_plaquettes: list[Plaquette | None] = [None, None, None, None]

    def __repr__(self):
        return f"DataQubit(index={self.index}, coord={self.coord})"

class Plaquette:
    def __init__(self, s_type: str, ordered_data_qubits: list[Qubit | None], ancilla: Qubit, coord: Coord):
        assert s_type in ['X','Z']
        self.type = s_type
        self.ordered_data_qubits = ordered_data_qubits
        self.ancilla = ancilla
        self.index = ancilla.index
        self.coord = coord

    def __eq__(self, other):
        # Compare plaquette type, ordered data qubit indexes (or None), ancilla index, and coord.
        # We avoid comparing objects directly to prevent recursive equality issues.
        def dq_key(dq):
            return None if dq is None else dq.index

        return (
            type(self) == type(other) and
            self.type == other.type and
            [dq_key(dq) for dq in self.ordered_data_qubits] == [dq_key(dq) for dq in other.ordered_data_qubits] and
            self.ancilla.index == other.ancilla.index and
            self.index == other.index and
            self.coord == other.coord
        )

    def __hash__(self):
        # Build a tuple of immutable keys consistent with __eq__.
        # Deliberately hash the plaquette type as a bool rather than the string
        # 'X'/'Z': Python randomizes string hashing per process, which would make
        # iteration order of any set of Plaquettes differ between runs, and hence
        # detector numbering non-reproducible. All other components are ints.
        dq_indexes = tuple(None if dq is None else dq.index for dq in self.ordered_data_qubits)
        return hash((self.type == 'X', dq_indexes, self.ancilla.index, self.index, self.coord))

    def __lt__(self, other):
        return (self.type < other.type) or (self.type == other.type and self.index < other.index)

    def __repr__(self):
        return f"Plaquette(type={self.type}, ancilla={self.ancilla.index}, coord={self.coord})"

class SurfaceCodeLayout:
    ### constructs an object which contains all the information about the surface
    # code layout, including the plaquette types and gate order
    Z_DELTAS = (Coord(1,1), Coord(-1,1), Coord(1,-1), Coord(-1,-1))
    X_DELTAS = (Coord(1,1), Coord(1,-1), Coord(-1,1), Coord(-1,-1))

    def __init__(self, d):
        d = int(d)
        self.d: int = d
        self.define_layout()

    def __eq__(self, other):
        return (type(self) == type(other) and self.d == other.d and
                self.data_qubits == other.data_qubits and
                self.qubits == other.qubits and
                self.plaquettes == other.plaquettes)

    def define_layout(self):
        ## write qubits and coord_to_qubit dictionary
        data_qubits, dq_coords = self.layout_data_qubits()
        ancillas, aq_coords = self.layout_ancillas(qubit_index=len(data_qubits))

        self.data_qubits = frozenset(data_qubits)
        self.qubits = frozenset(data_qubits + ancillas)
        self.coord_to_qubit = {**dq_coords, **aq_coords}
        self.index_to_qubit = {q.index: q for q in self.qubits}

        ## define plaquettes including gate orders
        z_plaqs, x_plaqs, coord_to_plaq, index_to_plaq = self.layout_plaquettes()

        self.plaquettes = frozenset(z_plaqs + x_plaqs)
        self.z_plaquettes = frozenset(z_plaqs)
        self.x_plaquettes = frozenset(x_plaqs)

        self.coord_to_plaquette = coord_to_plaq
        self.index_to_plaquette = index_to_plaq

        ## define logical observables
        z_logical, x_logical = self.get_logicals()
        self.z_logical = frozenset(z_logical)
        self.x_logical = frozenset(x_logical)

    def layout_data_qubits(self, qubit_index = 0, top_left = Coord(1,1)):
        ## define the data qubits in a grid
        d = self.d
        qubits : list[Qubit] = []
        coord_to_qubit: dict[Coord, Qubit]= {}
        for i in range(d):
            for j in range(d):
                coord = top_left + Coord(2*i, 2*j)
                qubit = DataQubit(qubit_index, coord)
                qubits.append(qubit)
                coord_to_qubit[coord] = qubit
                qubit_index += 1

        return qubits, coord_to_qubit

    def layout_ancillas(self, qubit_index: int):
        ## define the ancilla qubits
        d = self.d
        coord_list = []

        # the bulk ancillas
        top_left = Coord(2,2)
        for i in range(d-1):
            for j in range(d-1):
                coord_list.append(top_left + Coord(2*i, 2*j))

        # the edge ancillas
        top_left = Coord(0,2)
        top_right = Coord(d*2-2, 0)
        bottom_left = Coord(2, d*2)
        bottom_right = Coord(d*2, d*2-2)
        starts = [top_left, top_right, bottom_left, bottom_right]
        directions = [Coord(0,4), Coord(-4,0), Coord(4,0), Coord(0,-4)]
        for i in range(4):
            direction = directions[i]
            if (d%2==0) and ((i==1) or (i==2)):
                ran = (d-1)//2
                start = starts[i] + Coord(direction[0]//2, direction[1]//2)
            else:
                ran = d//2
                start = starts[i]
            for j in range(ran):
                coord_list.append(start + direction*j)

        coord_list.sort()
        qubits = [Qubit(qubit_index + i, coord) for i, coord in enumerate(coord_list)]
        coord_to_qubit = {coord: qubits[i] for i, coord in enumerate(coord_list)}

        return qubits, coord_to_qubit

    def layout_plaquettes(self, top_left = Coord(0,0)):
        d = self.d
        ## make plaquettes and coord_to_plaquette dictionary
        z_plaquettes: list[Plaquette] = []
        x_plaquettes: list[Plaquette] = []
        coord_to_plaquette: dict[Coord, Plaquette] = {}
        index_to_plaquette: dict[int, Plaquette] = {}

        z_deltas = self.Z_DELTAS
        x_deltas = self.X_DELTAS
        for i in range(d+1):
            for j in range(d+1):
                coord = top_left + Coord(j*2, i*2)
                if coord in self.coord_to_qubit:
                    s_type = 'Z' if ((i+j)%2)==0 else 'X'
                    ancilla = self.coord_to_qubit[coord]
                    deltas = z_deltas if s_type == 'Z' else x_deltas
                    ordered_data_qubits = [self.coord_to_qubit.get(coord+delta) for delta in deltas]
                    plaquette = Plaquette(s_type, ordered_data_qubits, ancilla, coord)
                    for (t, dq) in enumerate(ordered_data_qubits):
                        if dq is not None and isinstance(dq, DataQubit):
                            dq.ordered_plaquettes[t] = plaquette
                    coord_to_plaquette[coord] = plaquette
                    index_to_plaquette[ancilla.index] = plaquette
                    if s_type == 'Z':
                        z_plaquettes.append(plaquette)
                    else:
                        x_plaquettes.append(plaquette)

        return z_plaquettes, x_plaquettes, coord_to_plaquette, index_to_plaquette

    def get_logicals(self):
        z_logical = []
        x_logical = []
        for qubit in self.data_qubits:
            if qubit.coord[0] == 1:
                z_logical.append(qubit)
            if qubit.coord[1] == 1:
                x_logical.append(qubit)

        return z_logical, x_logical

    def __repr__(self):
        return f"SurfaceCodeLayout(d={self.d})"
