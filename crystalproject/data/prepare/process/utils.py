from molmod import MolecularGraph, CustomPattern

class CustomMolecularPattern(CustomPattern):
    '''
    In a GraphSearch, the atom numbers are forced to be the same
    '''
    def compare(self, vertex0, vertex1, subject_graph):
        return self.pattern_graph.numbers[vertex0] == subject_graph.numbers[vertex1]


patterns = {
        'Boronate': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 0]), ([8, 5, 8, 6, 6]))), [0, 1, 2]),
        'Boroxine': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]), [5, 8, 5, 8, 5, 8])), [0, 1, 2, 3, 4, 5]),
        'Borosilicate': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 10], [0, 4], [4, 5], [5, 6], [6, 10], [0, 7], [7, 8], [8, 9], [9, 10], [0, 11], [10, 12]), ([14, 8, 5, 8, 8, 5, 8, 8, 5, 8, 14, 1, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        'Imine': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 3], [2, 4]), ([6, 1, 7, 6, 6]))), [0, 1, 2]),
        'Hydrazone': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [3, 4], [3, 5], [5, 6], [0, 7], [5, 8]), ([6, 1, 7, 7, 1, 6, 8, 6, 6]))), [0, 1, 2, 3, 4, 5, 6]),
        'Azine': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [3, 4], [4, 5], [0, 6], [4, 7]), ([6, 1, 7, 7, 6, 1, 6, 6]))), [0, 1, 2, 3, 4, 5]),
        'Benzobisoxazole': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 0]), ([7, 6, 8, 6, 6]))), [0, 1, 2]),
        'Ketoenamine': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [2, 4], [4, 5], [4, 6], [6, 7]), ([7, 1, 6, 1, 6, 1, 6, 8]))), [0, 1, 2, 3, 4, 5, 6, 7]),
        'Triazine': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]), ([6, 7, 6, 7, 6, 7]))), [0, 1, 2, 3, 4, 5]),
        'Borazine': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [1, 6], [3, 7], [5, 8]), ([7, 5, 7, 5, 7, 5, 1, 1, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        'Imide': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [2, 6]), ([6, 7, 6, 6, 6, 8, 8]))), [0, 1, 2, 5, 6]),
        # Not included in database
        'AzineH': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [3, 4], [4, 5], [0, 6], [4, 7], [2, 8], [3, 9]), ([6, 1, 7, 7, 6, 1, 6, 6, 1, 1]))), [0, 1, 2, 3, 4, 5, 8, 9]),
        'HydrazoneH': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [3, 4], [3, 5], [5, 6], [0, 7], [5, 8], [2, 9]), ([6, 1, 7, 7, 1, 6, 8, 6, 6, 1]))), [0, 1, 2, 3, 4, 5, 6, 9]),
        'Benzimidazole': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5]), ([7, 6, 7, 6, 6, 1]))), [0, 1, 2, 5]),
        'Borosilicate-tBu': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 10], [0, 4], [4, 5], [5, 6], [6, 10], [0, 7], [7, 8], [8, 9], [9, 10], [0, 11], [11, 12], [11, 13], [11, 14], [12, 15], [12, 16], [12, 17], [13, 18], [13, 19], [13, 20], [14, 21], [14, 22], [14, 23], [10, 24], [24, 25], [24, 26], [24, 27], [25, 28], [25, 29], [25, 30], [26, 31], [26, 32], [26, 33], [27, 34], [27, 35], [27, 36]), ([14, 8, 5, 8, 8, 5, 8, 8, 5, 8, 14, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]),
        'Borosilicate-Me': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 10], [0, 4], [4, 5], [5, 6], [6, 10], [0, 7], [7, 8], [8, 9], [9, 10], [0, 11], [11, 12], [11, 13], [11, 14], [10, 15], [15, 16], [15, 17], [15, 18]), ([14, 8, 5, 8, 8, 5, 8, 8, 5, 8, 14, 6, 1, 1, 1, 6, 1, 1, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
        'Borosilicate-PropanoicAcid': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 10], [0, 4], [4, 5], [5, 6], [6, 10], [0, 7], [7, 8], [8, 9], [9, 10], [0, 11], [11, 12], [11, 13], [11, 14], [14, 15], [14, 16], [14, 17], [17, 18], [17, 19], [19, 20], [10, 21], [21, 22], [21, 23], [21, 24], [24, 25], [24, 26], [24, 27], [27, 28], [27, 29], [29, 30]), ([14, 8, 5, 8, 8, 5, 8, 8, 5, 8, 14, 6, 1, 1, 6, 1, 1, 6, 8, 8, 1, 6, 1, 1, 6, 1, 1, 6, 8, 8, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        'Thiazole': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 0]), ([7, 6, 16, 6, 6]))), [0, 1, 2]),
        'ImineCH2': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [3, 4], [3, 5], [0, 6], [6, 7], [6, 8]), ([6, 1, 7, 6, 6, 6, 6, 1, 1]))), [0, 1, 2]),
        'ImineNC': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 3], [2, 4], [3, 5], [3, 6], [4, 7]), ([6, 1, 7, 6, 7, 6, 6, 6]))), [0, 1, 2]),
        'ImineTG+': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 4], [2, 5], [4, 6], [4, 7], [5, 8], [5, 9], [2, 3]), ([6, 1, 7, 17, 6, 7, 6, 6, 1, 6]))), [0, 1, 2, 3]),
        'ImineCo': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 4], [2, 5], [4, 6], [4, 7], [5, 8], [5, 9], [2, 3]), ([6, 1, 7, 27, 6, 6, 6, 7, 6, 6]))), [0, 1, 2, 3]),
        'ImineUnphysical': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [2, 4], [2, 5], [3, 6], [3, 7]), ([6, 7, 6, 6, 6, 6, 6, 6]))), [0, 1]),
        'ImineUnphysical2': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [0, 4], [4, 5], [4, 6], [3, 7], [3, 8]), ([7, 1, 6, 6, 6, 6, 6, 6, 6]))), [0, 1, 2]),
        'Imide6': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [2, 7]), ([6, 7, 6, 6, 6, 6, 8, 8]))), [0, 1, 2, 6, 7]),
        'Amide': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [0, 4], [1, 5]), ([6, 7, 8, 1, 6, 6]))), [0, 1, 2, 3]),
        'Azodioxy': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [0, 4], [1, 5]), ([7, 7, 8, 8, 6, 6]))), [0, 1, 2, 3]),
        'BOImine': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 3], [3, 4], [3, 5], [2, 6], [6, 7], [6, 8]), ([8, 1, 5, 6, 6, 6, 6, 6, 6]))), [0, 1, 2]),
        'Enamine': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [0, 4], [1, 5]), ([6, 7, 1, 1, 6, 6]))), [0, 1, 2, 3]),
        'Amine': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [0, 4], [0, 5], [1, 6]), ([6, 7, 1, 1, 1, 6, 6]))), [0, 1, 2, 3, 4]),
        'Carbamate': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [2, 7]), ([6, 7, 6, 6, 6, 8, 8, 1]))), [0, 1, 2, 3, 4, 5, 6, 7]),
        'ThioCarbamate': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [2, 7]), ([6, 7, 6, 6, 6, 8, 16, 1]))), [0, 1, 2, 3, 4, 5, 6, 7]),
        'Dioxin': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]), ([6, 6, 8, 6, 6, 8]))), [2, 5]),
        'Phosphazene': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]), [15, 7, 15, 7, 15, 7])), [0, 1, 2, 3, 4, 5]),
        'BP-cube': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 8], [0, 16], [8, 16], [2, 17], [10, 17], [4, 18], [12, 18], [6, 19], [14, 19]), ([15, 8, 5, 8, 15, 8, 5, 8, 5, 8, 15, 8, 5, 8, 15, 8, 8, 8, 8, 8]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        'BP-rod': (CustomMolecularPattern(MolecularGraph(([0, 2], [1, 2], [2, 3], [3, 4], [4, 5]), ([8, 8, 15, 8, 5, 8]))), [0, 1, 2, 3, 4, 5]),
        'Spiroborate-Li': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [2, 5]), ([5, 8, 8, 8, 8, 3]))), [0, 1, 2, 3, 4, 5]),
        'Spiroborate': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 3], [0, 4]), ([5, 8, 8, 8, 8]))), [0, 1, 2, 3, 4]),
        'Olefin': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [1, 4], [4, 5], [4, 6], [0, 7], [7, 8], [7, 9], [5, 10], [5, 14], [6, 11], [6, 15], [8, 12], [8, 16], [9, 13], [9, 17]), ([6, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6]))), [0, 1, 2, 3]),
        'Olefin-CNterm': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [1, 4], [4, 5], [4, 6], [0, 7], [7, 8], [7, 9], [5, 10], [10, 11], [6, 12], [12, 13], [8, 14], [9, 15], [5, 16], [6, 17], [8, 18], [9, 19]), ([6, 6, 1, 1, 6, 6, 6, 6, 6, 6, 6, 7, 6, 7, 1, 1, 6, 6, 6, 6]))), [0, 1, 2, 3]),
        'Olefin(CN)': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [4, 6], [3, 7], [3, 8], [5, 9], [5, 10]), ([6, 6, 1, 6, 6, 6, 7, 6, 6, 6, 6]))), [0, 1, 2, 4, 6]),
        'C-C(CN)': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [0, 4], [4, 5], [4, 6], [4, 7], [7, 8], [7, 9], [0, 10], [10, 11], [10, 12]), ([6, 1, 6, 7, 6, 1, 1, 6, 6, 6, 6, 6, 6]))), [0, 1, 2, 3, 4, 5, 6]),
        'Furonitrile': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [5, 6]), ([6, 6, 8, 6, 6, 6, 7]))), [0, 1, 2, 5, 6]),
        'Ester': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [1, 3], [3, 4], [3, 5], [2, 6], [6, 7], [6, 8]), ([8, 6, 8, 6, 6, 6, 6, 6, 6]))), [0, 1, 2]),
        'EnamineN': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [3, 6], [3, 7]), ([6, 7, 1, 6, 1, 7, 6, 6]))), [0, 1, 2, 4]),
        'Phenazine': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [1, 6], [6, 7], [7, 8], [8, 9], [9, 2], [4, 10], [10, 11], [11, 12], [12, 13], [13, 5], [6, 14], [9, 15], [13, 16], [10, 17]), ([7, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1]))), [0, 3]),
        'Silicate-Na': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [5, 6], [6, 7], [7, 8], [8, 0], [0, 9], [9, 10], [10, 11], [11, 12], [12, 0], [1, 13], [5, 13], [9, 13]), ([14, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 6, 8, 11]))), [0, 1, 4, 5, 8, 9, 12, 13]),
        'Silicate-Li': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9]), ([14, 8, 3, 8, 8, 3, 8, 8, 3, 8]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        'Pyrimidazole1_LZU-561': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2], [6, 7], [7, 8], [8, 9], [9, 10], [10, 5], [7, 11], [10, 12], [8, 13], [9, 14]), ([1, 7, 6, 6, 7, 6, 7, 6, 6, 6, 6, 1, 1, 1, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        'Pyrimidazole2_LZU-562': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2], [6, 7], [7, 8], [8, 9], [9, 10], [10, 5], [7, 11], [10, 12], [8, 13], [9, 14]), ([1, 7, 6, 6, 7, 6, 7, 6, 6, 6, 6, 1, 8, 1, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        'Pyrimidazole3_LZU-563': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2], [6, 7], [7, 8], [8, 9], [9, 10], [10, 5], [7, 11], [10, 12], [8, 13], [9, 14]), ([1, 7, 6, 6, 7, 6, 7, 6, 6, 6, 6, 1, 1, 1, 17]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        'Pyrimidazole4_LZU-564': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2], [6, 7], [7, 8], [8, 9], [9, 10], [10, 5], [7, 11], [10, 12], [8, 13], [7, 13], [9, 13], [9, 14], [14, 15], [14, 16], [14, 17]), ([1, 7, 6, 6, 7, 6, 7, 6, 6, 6, 6, 1, 1, 35, 6, 1, 1, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
        'Aminal': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [2, 4], [1, 5], [5, 6], [5, 7], [0, 8], [8, 9], [9, 10], [9, 11], [8, 12], [12, 13], [12, 14], [0, 15], [0, 16], [16, 17], [16, 18]), ([6, 7, 6, 1, 1, 6, 1, 1, 7, 6, 1, 1, 6, 1, 1, 1, 6, 6, 6]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        'Squaraine': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [2, 5], [1, 6], [6, 7], [3, 8], [8, 9]), ([6, 6, 6, 6, 8, 8, 7, 1, 7, 1]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        'Salen': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [7, 8], [7, 9], [9, 10], [10, 11], [10, 12], [10, 13], [13, 14], [13, 15], [13, 16], [16, 17], [17, 18], [17, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 19], [20, 25], [6, 26], [25, 27]), ([6, 6, 6, 6, 6, 6, 8, 6, 1, 7, 6, 1, 1, 6, 1, 1, 7, 6, 1, 6, 6, 6, 6, 6, 6, 8, 1, 1]))), [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26, 27]),
        'Salen-Zn': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [7, 8], [7, 9], [9, 10], [10, 11], [10, 12], [10, 13], [13, 14], [13, 15], [13, 16], [16, 17], [17, 18], [17, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 19], [20, 25], [9, 26], [16, 26], [6, 26], [25, 26]), ([6, 6, 6, 6, 6, 6, 8, 6, 1, 7, 6, 1, 1, 6, 1, 1, 7, 6, 1, 6, 6, 6, 6, 6, 6, 8, 30]))), [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26]),
        '2015': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [3, 6], [6, 7], [7, 8], [8, 9], [9, 4], [9, 10], [10, 11], [11, 12], [12, 8], [12, 13], [11, 14], [14, 15], [14, 16], [14, 17]), ([6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 16, 6, 6, 1, 6, 1, 1, 1]))), [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
        '2106': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [3, 6], [6, 7], [7, 8], [8, 9], [9, 4], [8, 10], [9, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 11], [12, 16], [14, 17], [15, 18]), ([6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 1, 7, 6, 7, 6, 6, 1, 1, 1]))), [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
        'TetraHydroPyranQuinoline': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [3, 6], [6, 7], [7, 8], [8, 9], [9, 4], [9, 10], [10, 11], [11, 12], [12, 13], [13, 8], [11, 14], [11, 15], [12, 16], [12, 17], [13, 18], [13, 19]), ([6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 8, 6, 6, 6, 1, 1, 1, 1, 1, 1]))), [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        'PhenylQuinoline': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [3, 6], [6, 7], [7, 8], [8, 9], [9, 4], [8, 10], [9, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21]), ([6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1]))), [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
        'Alpha-AminoNitrile': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [2, 4], [4, 5], [0, 6], [6, 7], [6, 8], [2, 9], [9, 10], [10, 11]), ([7, 1, 6, 1, 6, 7, 6, 6, 6, 6, 6, 6]))), [0, 1, 2, 3, 4, 5]),
        'Propargylamine': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [2, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6], [7, 12], [8, 13], [9, 14], [10, 15], [11, 16], [0, 17], [17, 18], [17, 19], [2, 20], [20, 21], [20, 22]), ([7, 1, 6, 1, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6]))), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        # 自己添加的
        'NHNH': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [0, 4], [1, 5]), ([7, 7, 1, 1, 6, 6]))), [0, 1, 2, 3]),
        # 'COCO': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [0, 4], [1, 5]), ([6, 6, 8, 8, 6, 6]))), [0, 1, 2, 3]),
        # 'CH2CH2': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3], [0, 4], [1, 5], [0, 6], [1, 7]), ([6, 6, 1, 1, 1, 1, 6, 6]))), [0, 1, 2, 3, 4, 5]),
        # 'CHCH': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [2, 3], [0, 4], [2, 5]), ([6, 1, 6, 1, 6, 6]))), [0, 1, 2, 3]),
        'NN': (CustomMolecularPattern(MolecularGraph(([0, 1], [0, 2], [1, 3]), ([7, 7, 6, 6]))), [0, 1]),
        'cc3cc3cc': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5]), ([6, 6, 6, 6, 6, 6]))), [1, 2, 3, 4]),
        'ccnhccnh': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [2, 6], [6, 8], [3, 7], [7, 13], [6, 14], [7, 15], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 8]), ([6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6, 6, 6, 1, 1]))), [2, 3, 6, 7, 8, 13, 14, 15]),
        'cconnchc': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [5, 6], [5, 7]), ([6, 6, 8, 7, 7, 6, 1, 6]))), [1, 2, 3, 4, 5, 6]),
        }

# SBUs of mercado with connectivity of three or more
cc = {
        'linker91': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [2, 7], [4, 8]), ([6, 7, 6, 7, 6, 7, 6, 6, 6]))), [[0, 6], [2, 7], [4, 8]]),
        'linker92': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]), ([6, 6, 6, 6, 6, 6, 1, 6, 1, 6, 1, 6]))), [[1, 7], [3, 9], [5, 11]]),
        'linker93': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1], [2, 7], [3, 8], [4, 9], [5, 10], [6, 11], [0, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 12], [13, 18], [14, 19], [15, 20], [16, 21], [17, 22], [0, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 23], [24, 29], [25, 30], [26, 31], [27, 32], [28, 33]), ([7, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1]))), [[4, 9], [15, 20], [26, 31]]),
        'linker94': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [2, 7], [4, 8], [1, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 9], [3, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 15], [5, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 21], [10, 27], [11, 28], [12, 29], [13, 30], [14, 31], [16, 32], [17, 33], [18, 34], [19, 35], [20, 36], [22, 37], [23, 38], [24, 39], [25, 40], [26, 41]), ([6, 6, 6, 6, 6, 6, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 1, 1, 6, 1, 1, 1, 1, 6, 1, 1]))), [[12, 29], [18, 34], [24, 39]]),
        'linker95': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [2, 7], [4, 8], [1, 9], [9, 10], [9, 11], [11, 12], [11, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 13], [14, 19], [15, 20], [16, 21], [17, 22], [18, 23], [3, 24], [24, 25], [24, 26], [26, 27], [26, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 28], [29, 34], [30, 35], [31, 36], [32, 37], [33, 38], [5, 39], [39, 40], [39, 41], [41, 42], [41, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 43], [44, 49], [45, 50], [46, 51], [47, 52], [48, 53]), ([6, 6, 6, 6, 6, 6, 1, 1, 1, 6, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1]))), [[16, 21], [31, 36], [46, 51]]),
        'linker96': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6], [7, 12], [8, 13], [9, 14], [10, 15], [11, 16], [2, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 17], [18, 23], [19, 24], [20, 25], [21, 26], [22, 27], [4, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 28], [29, 34], [30, 35], [31, 36], [32, 37], [33, 38]), ([6, 7, 6, 7, 6, 7, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1]))), [[9, 14], [20, 25], [31, 36]]),        
        'linker97': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1], [2, 7], [3, 8], [5, 9], [6, 10], [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 11], [14, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 17], [18, 23], [19, 24], [21, 25], [22, 26], [16, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 27], [30, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 33], [34, 39], [35, 40], [37, 41], [38, 42], [32, 43], [43, 44], [44, 45], [45, 46], [46, 47], [46, 48], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 49], [50, 55], [51, 56], [53, 57], [54, 58], [48, 59], [59, 60], [60, 61], [61, 62], [62, 63], [63, 59], [63, 64], [31, 65], [62, 0], [4, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 71], [71, 66], [67, 72], [68, 73], [69, 74], [70, 75], [71, 76], [20, 77], [77, 78], [78, 79], [79, 80], [80, 81], [81, 82], [82, 77], [78, 83], [79, 84], [80, 85], [81, 86], [82, 87], [36, 88], [88, 89], [89, 90], [90, 91], [91, 92], [92, 93], [93, 88], [89, 94], [90, 95], [91, 96], [92, 97], [93, 98], [52, 99], [99, 100], [100, 101], [101, 102], [102, 103], [103, 104], [104, 99], [100, 105], [101, 106], [102, 107], [103, 108], [104, 109]), ([6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1]))), [[69, 74], [80, 85], [91, 96], [102, 107]]),
        'linker98': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1], [2, 7], [3, 8], [5, 9], [6, 10], [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 11], [14, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 17], [18, 23], [19, 24], [21, 25], [22, 26], [16, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 27], [30, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 33], [34, 39], [35, 40], [37, 41], [38, 42], [32, 43], [43, 44], [44, 45], [45, 46], [46, 47], [46, 48], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 49], [50, 55], [51, 56], [53, 57], [54, 58], [48, 59], [59, 60], [60, 61], [61, 62], [62, 63], [63, 59], [63, 64], [31, 65], [62, 0], [4, 66], [20, 67], [36, 68], [52, 69]), ([6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 1, 1, 6, 6, 6, 6]))), [[4, 66], [20, 67], [36, 68], [52, 69]]),
        'linker99': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21]), ([6, 6, 6, 6, 6, 6, 1, 6, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 6, 1, 6, 1]))), [[1, 7], [3, 9], [13, 18], [15, 20]]),
        'linker100': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21], [10, 22], [21, 23]), ([6, 6, 6, 6, 6, 6, 1, 6, 1, 6, 8, 6, 6, 6, 6, 6, 6, 1, 6, 1, 6, 8, 1, 1]))), [[1, 7], [3, 9], [13, 18], [15, 20]]),
        'linker101': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21], [10, 22], [10, 23], [21, 24], [21, 25]), ([6, 6, 6, 6, 6, 6, 1, 6, 1, 6, 7, 6, 6, 6, 6, 6, 6, 1, 6, 1, 6, 7, 1, 1, 1, 1]))), [[1, 7], [3, 9], [13, 18], [15, 20]]),
        'linker102': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 1], [2, 7], [3, 8], [4, 9], [5, 10], [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [0, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 21], [22, 27], [23, 28], [24, 29], [25, 30], [0, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 31], [32, 37], [33, 38], [34, 39], [35, 40], [6, 41], [16, 41], [41, 42], [41, 43], [26, 44], [36, 44], [44, 45], [44, 46]), ([6, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1]))), [[3, 8], [13, 18], [23, 28], [33, 38]]),
        'linker103': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 1], [2, 7], [3, 8], [4, 9], [5, 10], [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [0, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 21], [22, 27], [23, 28], [24, 29], [25, 30], [0, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 31], [32, 37], [33, 38], [34, 39], [35, 40], [6, 16], [26, 36]), ([6, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1]))), [[3, 8], [13, 18], [23, 28], [33, 38]]),
        'linker104': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21], [8, 22], [10, 23], [19, 24], [21, 25]), ([6, 6, 6, 6, 6, 6, 1, 6, 8, 6, 8, 6, 6, 6, 6, 6, 6, 1, 6, 8, 6, 8, 1, 1, 1, 1]))), [[1, 7], [3, 9], [13, 18], [15, 20]]),
        'linker105': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21], [8, 22], [19, 23]), ([6, 6, 6, 6, 6, 6, 1, 6, 8, 6, 1, 6, 6, 6, 6, 6, 6, 1, 6, 8, 6, 1, 1, 1]))), [[1, 7], [3, 9], [13, 18], [15, 20]]),
        'linker106': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21], [8, 22], [8, 23], [10, 24], [10, 25], [19, 26], [19, 27], [21, 28], [21, 29]), ([6, 6, 6, 6, 6, 6, 1, 6, 7, 6, 7, 6, 6, 6, 6, 6, 6, 1, 6, 7, 6, 7, 1, 1, 1, 1, 1, 1, 1, 1]))), [[1, 7], [3, 9], [13, 18], [15, 20]]),
        'linker107': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21], [8, 22], [8, 23], [19, 24], [19, 25]), ([6, 6, 6, 6, 6, 6, 1, 6, 7, 6, 1, 6, 6, 6, 6, 6, 6, 1, 6, 7, 6, 1, 1, 1, 1, 1]))), [[1, 7], [3, 9], [13, 18], [15, 20]]),
        'linker108': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [2, 6], [6, 7], [7, 8], [8, 9], [9, 3], [9, 10], [10, 11], [11, 12], [12, 4], [8, 13], [13, 14], [14, 15], [15, 10], [0, 16], [1, 17], [6, 18], [7, 19], [13, 20], [14, 21], [15, 22], [11, 23], [12, 24], [5, 25]), ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 6, 1, 1, 6, 1, 6, 1, 1, 6]))), [[1, 17], [13, 20], [15, 22], [5, 25]]),
        'linker109': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 1], [2, 7], [3, 8], [4, 9], [5, 10], [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [0, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 21], [22, 27], [23, 28], [24, 29], [25, 30], [0, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 31], [32, 37], [33, 38], [34, 39], [35, 40], [6, 41], [16, 42], [26, 43], [36, 44]), ([6, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 1, 1, 1]))), [[4, 9], [14, 19], [24, 29], [34, 39]]),
        'linker110': (CustomMolecularPattern(MolecularGraph(([0, 4], [4, 5], [4, 6], [4, 1], [0, 7], [7, 8], [7, 9], [7, 3], [0, 10], [10, 11], [10, 12], [10, 2], [1, 13], [13, 14], [13, 15], [13, 2], [2, 16], [16, 17], [16, 18], [16, 3], [1, 19], [19, 20], [19, 21], [19, 3], [0, 22], [1, 23], [2, 24], [3, 25]), ([6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 6, 6, 6]))), [[0, 22], [1, 23], [2, 24], [3, 25]]),
        'linker111': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 1], [2, 7], [3, 8], [4, 9], [5, 10], [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 11], [12, 17], [13, 18], [14, 19], [15, 20], [0, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 21], [22, 27], [23, 28], [24, 29], [25, 30], [0, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 31], [32, 37], [33, 38], [34, 39], [35, 40], [6, 41], [16, 42], [26, 43], [36, 44]), ([14, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 1, 6, 1, 1, 1, 1, 1]))), [[4, 9], [14, 19], [24, 29], [34, 39]])
        }

# Porphyrin ring used in Red-PV-COF
cn = {'P-N': (CustomMolecularPattern(MolecularGraph(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1], [2, 7], [3, 8], [5, 9], [6, 10], [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 11], [14, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 17], [18, 23], [19, 24], [21, 25], [22, 26], [16, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 27], [30, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 33], [34, 39], [35, 40], [37, 41], [38, 42], [32, 43], [43, 44], [44, 45], [45, 46], [46, 47], [46, 48], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 49], [50, 55], [51, 56], [53, 57], [54, 58], [48, 59], [59, 60], [60, 61], [61, 62], [62, 63], [63, 59], [63, 64], [31, 65], [62, 0], [4, 66], [20, 67], [36, 68], [52, 69]), ([6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 6, 6, 6, 6, 7, 1, 1, 7, 7, 7, 7]))), [[4, 66], [20, 67], [36, 68], [52, 69]])}

# Criteria to identify patterns
crit1 = {
        'Imine': [[3, 1, 2, 3, 3], [3, 1, 2, 3, 4], [3, 1, 2, 4, 3], [3, 1, 2, 4, 4]],
        'Amide': [[3, 3, 1, 1, 3, 3], [3, 3, 1, 1, 3, 4], [3, 3, 1, 1, 4, 3], [3, 3, 1, 1, 4, 4]],
        'Azodioxy': [[3, 3, 1, 1, 3, 3], [3, 3, 1, 1, 3, 4], [3, 3, 1, 1, 4, 3], [3, 3, 1, 1, 4, 4]],
        'Enamine': [[3, 3, 1, 1, 3, 3], [3, 3, 1, 1, 3, 4], [3, 3, 1, 1, 4, 3], [3, 3, 1, 1, 4, 4]],
        'Amine': [[4, 3, 1, 1, 1, 3, 3], [4, 3, 1, 1, 1, 3, 4], [4, 3, 1, 1, 1, 4, 3], [4, 3, 1, 1, 1, 4, 4]],
        'Ester': [[1, 3, 2]],
        'ImineUnphysical': [[2, 2, 3, 3]],
        'ImineUnphysical2': [[3, 1, 2, 3, 3]],
        'ImineNC': [[3, 1, 2, 3, 2]],
        'ImineTG+': [[3, 1, 3, 1, 3, 3]],
        'ImineCo': [[3, 1, 3, 6, 3, 3]],
        'Imide': [[3, 3, 3, 3, 3, 1, 1]],
        'Imide6': [[3, 3, 3, 3, 3, 3, 1, 1]],
        'Olefin': [[3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1]],
        'Olefin-CNterm': [[3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 2, 1, 2, 1, 1, 1]],
        'Olefin(CN)': [[3, 3, 1, 3, 2, 3, 1]],
        'BP-rod': [[1, 2, 4, 2, 4, 1]],
        'Spiroborate': [[4, 2, 2, 2, 2]],
        'Ketoenamine': [[3, 1, 3, 1, 3, 1, 3, 1]],
        'Silicate-Li': [[6, 3, 2, 3, 3, 2, 3, 3, 2, 3]],
        'BOImine': [[3, 1, 2, 3, 3]],
        # 自己添加的
        'cc3cc3cc': [[3, 2, 2, 2, 2, 3], [3, 2, 2, 2, 2, 4], [4, 2, 2, 2, 2, 3], [4, 2, 2, 2, 2, 4]],
        'ccnhccnh': [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1]],
        'cconnchc': [[3, 3, 1, 2, 2, 3, 1, 3]],
    }

crit2 = {
        'Imine': [],
        'Amide': [],
        'Azodioxy': [],
        'Enamine': [],
        'Amine': [],
        'ImineUnphysical': [4, 5, 6, 7],
        'ImineUnphysical2': [5, 6, 7, 8],
        'ImineNC': [5, 6, 7],
        'ImineTG+': [6, 7, 8, 9],
        'ImineCo': [6, 7, 8, 9],
        'Olefin': [5, 6, 8, 9],
        'Olefin-CNterm': [5, 6, 8, 9],
        'Olefin(CN)': [7, 8, 9, 10],
        'BOImine': [4, 5, 7, 8]
    }


# Electronegativity (Pauling) by atom symbol
endict = {"H": 2.20, "He": 4.16,
          "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
          "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
          "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
          "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81,
          "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22,
          "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 2.10, "Ru": 2.20, "Rh": 2.28,
          "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "I": 2.66,
          "Cs": 0.79, "Ba": 0.89, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20,
          "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02,
          "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17,
          "Gd": 1.20, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Lu": 1.27,
          "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28,
          "Am": 1.3, "Cm": 1.3, "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3,
          "Yb": 1.1, "Eu": 1.2, "Tb": 1.1, "Te": 2.10}

# Polarizability (alpha) by atom symbol
# From https://www.tandfonline.com/doi/full/10.1080/00268976.2018.1535143
# Last accessed 4/28/20

poldict = {"H": 4.50711, "He": 1.38375,
           "Li": 164.1125, "Be": 37.74, "B": 20.5, "C": 11.3, "N": 7.4,
           "O":5.3, "F": 3.74, "Ne": 2.66, "Na": 162.7, "Mg":71.2, "Al": 57.8, "Si": 37.3, "P": 25,
           "S": 19.4, "Cl": 14.6, "Ar": 11.083, "K": 289.7, "Ca": 160.8, "Sc": 97, "Ti": 100,
           "V": 87, "Cr": 83, "Mn": 68, "Fe": 62, "Co": 55, "Ni": 49, "Cu": 46.5, "Zn": 38.67,
           "Ga": 50, "Ge": 40, "As": 30, "Se": 28.9, "Br": 21, "Kr": 16.78, "Rb": 319.8, "Sr": 197.2,
           "Y": 162, "Zr": 112, "Nb": 98, "Mo": 87, "Tc": 79, "Ru": 72, "Rh": 66, "Pd": 26.14,
           "Ag": 55, "Cd": 46, "In": 65, "Sn": 53, "Sb": 43, "Te": 38, "I": 32.9, "Xe": 27.32,
           "Cs": 400.9, "Ba": 272, "La": 215, "Ce": 205, "Pr": 216, "Nd": 208, "Pm": 200, "Sm": 192,
           "Eu": 184, "Gd": 158, "Tb": 170, "Dy": 163, "Ho": 156, "Er": 150, "Tm": 144,
           "Yb": 139, "Lu": 137, "Hf": 103, "Ta": 74, "W": 68, "Re": 62, "Os": 57, "Ir": 54,
           "Pt": 48, "Au": 36, "Hg": 33.91, "Tl": 50, "Pb": 47, "Bi": 48, "Po": 44, "At": 42,
           "Rn": 35, "Fr": 317.8, "Ra": 246, "Ac": 203, "Pa": 154, "U": 129, "Np": 151, "Pu": 132,
           "Am": 131, "Cm": 144, "Bk": 125, "Cf": 122, "Es": 118, "Fm": 113, "Md": 109, "No": 110,
           "Lr": 320, "Rf": 112, "Db": 42, "Sg": 40, "Bh": 38, "Hs": 36, "Mt": 34, "Ds": 32,
           "Rg": 32, "Cn": 28, "Nh": 29, "Fl": 31, "Mc": 71, "Ts": 76, "Og": 58}

